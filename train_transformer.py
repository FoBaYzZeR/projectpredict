#!/usr/bin/env python3
"""
train_transformer.py — stable edition

- Windows-friendly DataLoader
- Updated AMP API
- Gradient clipping
- Label smoothing
- NaN/Inf guard
- TensorBoard, CSV logs, confusion matrices
"""

from __future__ import annotations

import argparse, json, platform, warnings 
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch
import os
os.environ["PYTORCH_SDPA_ENABLED"] = "0"


# --- warnings cleanup ---
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def load_meta(meta_path: Path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def time_mask_from_ranges(dt: pd.Series, start: str, end: str):
    dt = pd.to_datetime(dt, utc=True)
    return (dt >= pd.to_datetime(start, utc=True)) & (dt <= pd.to_datetime(end, utc=True))

def save_confusion_matrix_png(cm: np.ndarray, outpath: Path, title: str):
    plt.figure(figsize=(4, 4), dpi=140)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    ticks = [0,1,2]
    plt.xticks(ticks, ["Down","Flat","Up"])
    plt.yticks(ticks, ["Down","Flat","Up"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_curves(history: dict, outdir: Path):
    e = history["epoch"]
    plt.figure(figsize=(6,3), dpi=140)
    plt.plot(e, history["train_loss"], label="train_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "curve_train_loss.png"); plt.close()

    plt.figure(figsize=(6,3), dpi=140)
    plt.plot(e, history["val_macroF1"], label="val_macroF1")
    plt.plot(e, history["val_acc"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.title("Validation Scores"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "curve_val_scores.png"); plt.close()

# -----------------------------
# Dataset
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, dt: np.ndarray, seq_len: int):
        assert len(X) == len(y) == len(dt)
        self.X = X
        self.y = y.astype(np.int64)
        self.dt = dt
        self.seq_len = seq_len
        idx = np.arange(len(X), dtype=np.int64)
        self.valid_idx = idx[idx >= (seq_len - 1)]

    def __len__(self): return len(self.valid_idx)

    def __getitem__(self, i):
        end = self.valid_idx[i]
        start = end - self.seq_len + 1
        x = self.X[start:end+1]
        y = self.y[end]
        dt = self.dt[end]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), str(pd.Timestamp(dt, tz="UTC"))

# -----------------------------
# Model
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # [B, L, D]
        return x + self.pe[:, :x.size(1), :]

class ConvStem(nn.Module):
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d_model//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model//2, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):  # [B, L, F] -> [B, L, D]
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)

class TinyTransformer(nn.Module):
    def __init__(self, in_feats: int, d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 dim_ff: int = 256, dropout: float = 0.1, n_classes: int = 3, max_len: int = 4096):
        super().__init__()
        self.stem = ConvStem(in_feats, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=False, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(d_model, max_len=max_len)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
    def forward(self, x):
        h = self.stem(x)
        h = self.posenc(h)
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        return self.head(h)

# -----------------------------
# Train / Eval
# -----------------------------
def eval_epoch(model, loader, device):
    model.eval()
    ys, yh, dts = [], [], []
    with torch.no_grad():
        for xb, yb, dtb in loader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            ys.append(yb.cpu().numpy()); yh.append(pred.cpu().numpy()); dts.extend(dtb)
    if not ys: return 0.0, 0.0, np.zeros((3,3), dtype=int), np.array([]), np.array([]), dts
    y_true = np.concatenate(ys); y_pred = np.concatenate(yh)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    return acc, mf1, cm, y_true, y_pred, dts

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    feats_path = Path(args.data)
    meta_path = Path(args.meta) if args.meta else feats_path.with_suffix(".meta.json")
    meta = load_meta(meta_path)

    df = pd.read_parquet(feats_path)
    ignore = {"datetime", "close", "target_return_1h", "label_cls"}
    feat_cols = [c for c in df.columns if c not in ignore]
    print(f"Using {len(feat_cols)} feature columns")

    dt = pd.to_datetime(df["datetime"], utc=True).values
    y = df["label_cls"].to_numpy(dtype=np.int64)
    X = df[feat_cols].to_numpy(dtype=np.float32)

    # Splits
    tr = meta["splits"]["train_range"]; vr = meta["splits"]["val_range"]; te = meta["splits"]["test_range"]
    m_train = time_mask_from_ranges(df["datetime"], tr["start"], tr["end"]).to_numpy()
    m_val   = time_mask_from_ranges(df["datetime"], vr["start"], vr["end"]).to_numpy()
    m_test  = time_mask_from_ranges(df["datetime"], te["start"], te["end"]).to_numpy()

    # Standardize (train only)
    mu = X[m_train].mean(axis=0); sd = X[m_train].std(axis=0); sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    np.savez(feats_path.with_suffix(".scaler.npz"), mean=mu, std=sd, features=np.array(feat_cols, dtype=object))
    print("Saved scaler:", feats_path.with_suffix(".scaler.npz").name)

    # Datasets
    train_ds = SeqDataset(Xs[m_train], y[m_train], dt[m_train], seq_len=args.seq_len)
    val_ds   = SeqDataset(Xs[m_val],   y[m_val],   dt[m_val],   seq_len=args.seq_len)
    test_ds  = SeqDataset(Xs[m_test],  y[m_test],  dt[m_test],  seq_len=args.seq_len)

    # Class weights
    classes, counts = np.unique(train_ds.y, return_counts=True)
    inv = 1.0 / (counts + 1e-6); w = inv / inv.sum() * len(classes)
    class_weights = torch.tensor([w[classes.tolist().index(k)] if k in classes else 1.0 for k in range(3)], dtype=torch.float32).to(device)
    print("Class distribution (train):", {int(k): int(v) for k, v in zip(classes, counts)})
    print("Class weights:", class_weights.cpu().numpy())

    # Loaders (Windows-safe)
    win = platform.system().lower().startswith("win")
    nw_train, nw_eval = (0, 0) if win else (4, 2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw_train, pin_memory=(device.type=="cuda" and not win),
                              drop_last=True, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw_eval, pin_memory=(device.type=="cuda" and not win),
                            persistent_workers=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=nw_eval, pin_memory=(device.type=="cuda" and not win),
                             persistent_workers=False)

    # Model/opt
    model = TinyTransformer(in_feats=len(feat_cols), d_model=args.d_model, nhead=args.nhead,
                            num_layers=args.layers, dim_ff=args.ffn, dropout=args.dropout,
                            n_classes=3, max_len=max(args.seq_len, 4096)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    # Logging
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(outdir / "tb"))
    csv_log = outdir / "metrics.csv"
    with open(csv_log, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_acc,val_macroF1,lr\n")

    history = {"epoch": [], "train_loss": [], "val_acc": [], "val_macroF1": []}
    best_val = -np.inf; patience_left = args.patience

    # Loop
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", mininterval=0.5)
        for xb, yb, _ in pbar:
            xb = xb.to(device, dtype=torch.float32); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(xb); loss = criterion(logits, yb)
            if not torch.isfinite(loss):  # guard
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
            running += loss.item(); pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        # Eval
        val_acc, val_f1, val_cm, _, _, _ = eval_epoch(model, val_loader, device)
        train_loss = running / max(1, len(train_loader)); lr_now = scheduler.get_last_lr()[0]
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("metric/val_acc", val_acc, epoch)
        writer.add_scalar("metric/val_macroF1", val_f1, epoch)
        writer.add_scalar("lr", lr_now, epoch)
        save_confusion_matrix_png(val_cm, outdir / f"cm_val_epoch{epoch:03d}.png", f"Val CM (epoch {epoch})")
        with open(csv_log, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_acc:.6f},{val_f1:.6f},{lr_now:.8f}\n")
        history["epoch"].append(epoch); history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc); history["val_macroF1"].append(val_f1)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_macroF1={val_f1:.4f}")
        if val_f1 > best_val + 1e-5:
            best_val = val_f1; patience_left = args.patience
            torch.save({"state_dict": model.state_dict(),
                        "meta": {"feat_cols": feat_cols, "seq_len": args.seq_len,
                                 "d_model": args.d_model, "nhead": args.nhead,
                                 "layers": args.layers, "ffn": args.ffn, "dropout": args.dropout}},
                       outdir / "best_model.pt")
            print("  ↳ Saved new best: best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    plot_curves(history, outdir)

    # Final test
    best = torch.load(outdir / "best_model.pt", map_location=device)
    model.load_state_dict(best["state_dict"])
    test_acc, test_f1, test_cm, y_true, y_pred, dts = eval_epoch(model, test_loader, device)
    print("\nTEST RESULTS"); print(f"test_acc={test_acc:.4f} | test_macroF1={test_f1:.4f}")
    print("Test CM:\n", test_cm)
    save_confusion_matrix_png(test_cm, outdir / "cm_test.png", "Test CM")

    # Save preds
    model.eval(); all_probs, all_dt = [], []
    with torch.no_grad():
        for xb, _, dtb in tqdm(test_loader, desc="Predicting test"):
            xb = xb.to(device, dtype=torch.float32)
            probs = torch.softmax(model(xb), dim=1).cpu().numpy()
            all_probs.append(probs); all_dt.extend(dtb)
    all_probs = np.vstack(all_probs)
    pd.DataFrame({
        "datetime": all_dt, "true": y_true[:len(all_dt)], "pred": y_pred[:len(all_dt)],
        "p_down": all_probs[:,0], "p_flat": all_probs[:,1], "p_up": all_probs[:,2],
    }).to_csv(outdir / "test_predictions.csv", index=False)
    print("Saved:", outdir / "test_predictions.csv")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to btc_1m_feats.parquet")
    ap.add_argument("--meta", default=None, help="Path to .meta.json (defaults alongside data)")
    ap.add_argument("--outdir", default="./runs", help="Output directory for checkpoints & logs")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    train(args)
