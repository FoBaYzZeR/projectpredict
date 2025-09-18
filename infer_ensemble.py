#!/usr/bin/env python3
"""
infer_ensemble.py

Run inference with both small + medium models and produce a consensus decision.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from train_transformer import TinyTransformer

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -----------------------------
# Helpers
# -----------------------------
def load_model_and_scaler(model_path: Path, scaler_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    meta = ckpt["meta"]

    scaler = np.load(scaler_path, allow_pickle=True)
    mu, sd, feat_cols = scaler["mean"], scaler["std"], scaler["features"]

    model = TinyTransformer(
        in_feats=len(feat_cols),
        d_model=meta["d_model"],
        nhead=meta["nhead"],
        num_layers=meta["layers"],
        dim_ff=meta["ffn"],
        dropout=meta["dropout"],
        n_classes=3,
        max_len=4096,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, mu, sd, feat_cols, meta["seq_len"]


def preprocess_input(df: pd.DataFrame, feat_cols, mu, sd, seq_len: int):
    X = df[feat_cols].to_numpy(dtype=np.float32)
    Xs = (X - mu) / sd
    if len(Xs) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {len(Xs)}")
    return torch.from_numpy(Xs[-seq_len:]).unsqueeze(0)  # [1, L, F]


# -----------------------------
# Main
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load both models
    model_small, mu_s, sd_s, feat_s, seq_s = load_model_and_scaler(
        Path(args.model_small), Path(args.scaler), device
    )
    model_med, mu_m, sd_m, feat_m, seq_m = load_model_and_scaler(
        Path(args.model_medium), Path(args.scaler), device
    )

    # Load input data
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    # Preprocess for both
    xb_s = preprocess_input(df, feat_s, mu_s, sd_s, seq_s).to(device, dtype=torch.float32)
    xb_m = preprocess_input(df, feat_m, mu_m, sd_m, seq_m).to(device, dtype=torch.float32)

    # Predictions
    with torch.no_grad():
        probs_s = torch.softmax(model_small(xb_s), dim=1).cpu().numpy()[0]
        probs_m = torch.softmax(model_med(xb_m), dim=1).cpu().numpy()[0]

    labels = ["Down", "Flat", "Up"]
    pred_s, pred_m = np.argmax(probs_s), np.argmax(probs_m)

    print("=== Small model ===")
    for lbl, p in zip(labels, probs_s):
        print(f"  {lbl:<5} : {p:.3f}")
    print(f"→ Most likely: {labels[pred_s]}\n")

    print("=== Medium model ===")
    for lbl, p in zip(labels, probs_m):
        print(f"  {lbl:<5} : {p:.3f}")
    print(f"→ Most likely: {labels[pred_m]}\n")

    # Consensus rule
    if pred_s == pred_m and probs_s[pred_s] >= args.threshold and probs_m[pred_m] >= args.threshold:
        print(f"✅ Consensus trade: {labels[pred_s]} (confidence ≥ {args.threshold})")
    else:
        print("🚫 No trade (disagreement or low confidence)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-small", default="runs_small/best_model.pt")
    ap.add_argument("--model-medium", default="runs_medium_lr1e4/best_model.pt")
    ap.add_argument("--scaler", default="btc_1m_feats.scaler.npz")
    ap.add_argument("--input", required=True, help="CSV/Parquet with recent features")
    ap.add_argument("--threshold", type=float, default=0.97, help="Min prob for consensus trade")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
