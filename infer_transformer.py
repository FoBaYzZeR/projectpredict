#!/usr/bin/env python3
"""
infer_transformer.py

Load best_model.pt + scaler and predict the next 1h trend
given a CSV/Parquet of recent BTC candles.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from train_transformer import TinyTransformer  # reuse model definition

# -----------------------------
# Helper
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
        max_len=4096,  # ✅ match training setup
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, mu, sd, feat_cols, meta["seq_len"]

def preprocess_input(df: pd.DataFrame, feat_cols, mu, sd, seq_len: int):
    # select features in correct order
    X = df[feat_cols].to_numpy(dtype=np.float32)
    # normalize
    Xs = (X - mu) / sd
    # last seq_len window
    if len(Xs) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {len(Xs)}")
    return torch.from_numpy(Xs[-seq_len:]).unsqueeze(0)  # [1, L, F]

# -----------------------------
# Main
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, mu, sd, feat_cols, seq_len = load_model_and_scaler(
        Path(args.model), Path(args.scaler), device
    )

    # Load data (CSV/Parquet with the same features as training)
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    xb = preprocess_input(df, feat_cols, mu, sd, seq_len).to(device, dtype=torch.float32)

    with torch.no_grad():
        probs = torch.softmax(model(xb), dim=1).cpu().numpy()[0]

    labels = ["Down", "Flat", "Up"]
    print("Prediction for next 1h trend:")
    for lbl, p in zip(labels, probs):
        print(f"  {lbl:<5} : {p:.3f}")
    print("→ Most likely:", labels[np.argmax(probs)])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs_small_lr5e4/best_model.pt", help="Path to trained model")
    ap.add_argument("--scaler", default="btc_1m_feats.scaler.npz", help="Path to scaler .npz")
    ap.add_argument("--input", required=True, help="CSV/Parquet with recent features (same columns as training)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
