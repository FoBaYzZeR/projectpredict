#!/usr/bin/env python3
"""
make_features.py

Read a continuous 1-minute OHLCV parquet (from preprocess step) and produce
feature-rich data + 1h-ahead trend labels for transformer training.

Features (per minute, all past-looking to avoid leakage):
- log return (1-min)
- rolling volatility (std of log returns): w=5, 15, 60
- RSI(14)
- MACD(12,26,9): macd, signal, hist
- Bollinger %B (20)
- Volume z-scores: short=15, long=60  (z = (vol - mean)/std)
- VWAP distance (20): (close - vwap_20) / vwap_20
- Time encodings: minute-of-day sin/cos, day-of-week sin/cos
- Optional extras from source: num_trades, taker_buy_base, taker_buy_quote (scaled later)

Labels:
- 1-hour ahead (horizon_steps=60) **log return** r = ln(C[t+60]) - ln(C[t])
- Ternary classes with ±threshold_bps (default 10 bps = 0.001):
    Up   if r > +th
    Down if r < -th
    Flat otherwise
- Stored as `label_cls` in {0:Down, 1:Flat, 2:Up} and `target_return_1h`

Output:
- A Parquet with: datetime, close, features..., label_cls, target_return_1h
- A sidecar JSON with metadata (feature names, label mapping, horizon, thresholds, split ratios)
- (Optional) a CSV listing the train/val/test datetime ranges.

Usage example:
  python make_features.py \
    --in btc_1m_2022_2025.parquet \
    --out btc_1m_feats.parquet \
    --horizon 60 \
    --threshold-bps 10 \
    --train 0.70 --val 0.15 --test 0.15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder's RSI using exponential smoothing (no future info)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing ~ EMA with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss.replace(0.0, np.nan))

    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val.fillna(50.0)  # neutral where undefined


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal, histogram (EMA-based, no future info)."""
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_percent_b(series: pd.Series, window: int = 20):
    """%B = (price - ma) / (k*std); using k=2 (upper/lower bands +/- 2 std)."""
    ma = series.rolling(window=window, min_periods=window).mean()
    sd = series.rolling(window=window, min_periods=window).std()
    # k=2 bands → normalize by (2*sd) to get how far within bands; clip to [-inf, inf]
    percent_b = (series - ma) / (2.0 * sd.replace(0.0, np.nan))
    return percent_b.fillna(0.0)


def rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of returns (past only)."""
    return returns.rolling(window=window, min_periods=window).std().fillna(0.0)


def vwap_distance(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """VWAP over a rolling window; distance = (close - vwap)/vwap."""
    pv = (close * volume).rolling(window=window, min_periods=1).sum()
    v = volume.rolling(window=window, min_periods=1).sum()
    vwap = (pv / v.replace(0.0, np.nan)).fillna(method="ffill").fillna(close)
    return ((close - vwap) / vwap).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window=window, min_periods=window).mean()
    s = series.rolling(window=window, min_periods=window).std()
    z = (series - m) / s.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def make_time_features(dt: pd.Series) -> pd.DataFrame:
    """Sin/Cos encodings for minute-of-day and day-of-week."""
    # Ensure UTC and datetime64[ns, UTC]
    if not pd.api.types.is_datetime64tz_dtype(dt):
        dt = pd.to_datetime(dt, utc=True)
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    day_of_week = dt.dt.dayofweek

    mod = 1440.0
    mod_w = 7.0
    sod_sin = np.sin(2.0 * np.pi * (minute_of_day / mod))
    sod_cos = np.cos(2.0 * np.pi * (minute_of_day / mod))
    dow_sin = np.sin(2.0 * np.pi * (day_of_week / mod_w))
    dow_cos = np.cos(2.0 * np.pi * (day_of_week / mod_w))

    return pd.DataFrame(
        {
            "time_sin_m": sod_sin.astype("float32"),
            "time_cos_m": sod_cos.astype("float32"),
            "dow_sin": dow_sin.astype("float32"),
            "dow_cos": dow_cos.astype("float32"),
        }
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: columns = datetime, open, high, low, close, volume, num_trades, taker_buy_base, taker_buy_quote
    returns: features DataFrame aligned to df.index
    """
    out = pd.DataFrame(index=df.index.copy())

    # Basic 1-min log return
    lr1 = np.log(df["close"].astype("float64")).diff().fillna(0.0).astype("float32")
    out["ret_1"] = lr1

    # Rolling vols on returns (5, 15, 60)
    out["vol_5"] = rolling_vol(lr1, 5).astype("float32")
    out["vol_15"] = rolling_vol(lr1, 15).astype("float32")
    out["vol_60"] = rolling_vol(lr1, 60).astype("float32")

    # RSI(14) on close
    out["rsi_14"] = rsi(df["close"].astype("float64"), 14).astype("float32")

    # MACD(12,26,9) on close
    macd_line, signal_line, hist = macd(df["close"].astype("float64"), 12, 26, 9)
    out["macd"] = macd_line.astype("float32").fillna(0.0)
    out["macd_sig"] = signal_line.astype("float32").fillna(0.0)
    out["macd_hist"] = hist.astype("float32").fillna(0.0)

    # Bollinger %B (20)
    out["bb_percB_20"] = bollinger_percent_b(df["close"].astype("float64"), 20).astype("float32")

    # Volume z-scores (15, 60)
    vol = df["volume"].astype("float64")
    out["vol_z_15"] = zscore(vol, 15).astype("float32")
    out["vol_z_60"] = zscore(vol, 60).astype("float32")

    # VWAP distance (20)
    out["vwap_dist_20"] = vwap_distance(df["close"].astype("float64"), vol, 20).astype("float32")

    # Time encodings
    tf = make_time_features(df["datetime"])
    out = pd.concat([out, tf.set_index(df.index)], axis=1)

    # Optional raw microstructure-ish signals (kept raw; scale later in training)
    out["trades"] = df["num_trades"].astype("float32")
    out["taker_base"] = df["taker_buy_base"].astype("float32")
    out["taker_quote"] = df["taker_buy_quote"].astype("float32")

    # Replace any residual NaNs/Infs
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return out


def make_labels(df: pd.DataFrame, horizon_steps: int, threshold_bps: float):
    """
    Returns (target_return_1h, label_cls) aligned to df.index
    label_cls mapping: 0=Down, 1=Flat, 2=Up
    """
    # future log return over horizon_steps
    fut_close = df["close"].shift(-horizon_steps)
    target_return = np.log(fut_close) - np.log(df["close"])
    # thresholds in absolute log-return
    th = threshold_bps / 1e4  # bps→decimal (10 bps -> 0.001)
    # classes
    label_cls = pd.Series(np.int8(1), index=df.index)  # default Flat=1
    label_cls[target_return > +th] = 2  # Up
    label_cls[target_return < -th] = 0  # Down

    return target_return.astype("float32"), label_cls


def parse_args():
    ap = argparse.ArgumentParser(description="Build features + labels for BTC 1m transformer training.")
    ap.add_argument("--in", dest="infile", required=True, help="Input continuous parquet from preprocess step")
    ap.add_argument("--out", dest="outfile", required=True, help="Output features parquet")
    ap.add_argument("--horizon", type=int, default=60, help="Label horizon in steps (minutes), default 60")
    ap.add_argument("--threshold-bps", type=float, default=10.0, help="Ternary threshold in basis points, default 10")
    ap.add_argument("--train", type=float, default=0.70, help="Train split ratio")
    ap.add_argument("--val", type=float, default=0.15, help="Val split ratio")
    ap.add_argument("--test", type=float, default=0.15, help="Test split ratio")
    return ap.parse_args()


def main():
    args = parse_args()
    infile = Path(args.infile)
    outfile = Path(args.outfile)

    if not infile.exists():
        raise SystemExit(f"Input parquet not found: {infile}")

    print(f"Loading {infile} ...")
    df = pd.read_parquet(infile)

    # Ensure expected columns exist
    expected = {"datetime","open","high","low","close","volume","num_trades","taker_buy_base","taker_buy_quote"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"Missing expected columns: {sorted(missing)}")

    # Sort and set index
    df = df.sort_values("datetime").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index(pd.RangeIndex(len(df)), inplace=True)  # simple integer index for alignment speed

    # Build features
    feats = build_features(df)

    # Labels
    target_ret, label_cls = make_labels(df, horizon_steps=args.horizon, threshold_bps=args.threshold_bps)

    # Assemble output frame
    out = pd.DataFrame(index=feats.index)
    out["datetime"] = df["datetime"].values
    out["close"] = df["close"].astype("float32").values
    out = pd.concat([out, feats], axis=1)
    out["target_return_1h"] = target_ret.values
    out["label_cls"] = label_cls.values

    # Drop rows where label would be undefined (the last 'horizon' rows after shift)
    valid = out["target_return_1h"].notna()
    out = out.loc[valid].reset_index(drop=True)

    # Cast final dtypes tightly
    float_cols = out.select_dtypes(include=["float64"]).columns
    out[float_cols] = out[float_cols].astype("float32")
    out["label_cls"] = out["label_cls"].astype("int8")

    # Compute split cutoffs by ratio (time-based)
    n = len(out)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    n_test = n - n_train - n_val
    idx_train_end = n_train - 1
    idx_val_end = n_train + n_val - 1

    dt_train = (str(out.loc[0, "datetime"]), str(out.loc[idx_train_end, "datetime"]))
    dt_val   = (str(out.loc[idx_train_end + 1, "datetime"]), str(out.loc[idx_val_end, "datetime"]))
    dt_test  = (str(out.loc[idx_val_end + 1, "datetime"]), str(out.loc[n - 1, "datetime"]))

    # Save Parquet
    outfile.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(outfile, index=False, engine="pyarrow", compression="snappy")
    print(f"Saved {outfile}  shape={out.shape}")

    # Save metadata
    feature_cols = [c for c in out.columns if c not in ("datetime","close","target_return_1h","label_cls")]
    meta = {
        "source_file": str(infile),
        "output_file": str(outfile),
        "rows": int(out.shape[0]),
        "horizon_steps": int(args.horizon),
        "threshold_bps": float(args.threshold_bps),
        "label_mapping": {"0":"Down", "1":"Flat", "2":"Up"},
        "feature_cols": feature_cols,
        "splits": {
            "ratios": {"train": args.train, "val": args.val, "test": args.test},
            "train_range": {"start": dt_train[0], "end": dt_train[1], "rows": n_train},
            "val_range":   {"start": dt_val[0],   "end": dt_val[1],   "rows": n_val},
            "test_range":  {"start": dt_test[0],  "end": dt_test[1],  "rows": n_test},
        },
    }
    meta_path = outfile.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {meta_path}")

    # Optional: quick class balance report
    cls_counts = out["label_cls"].value_counts().sort_index()
    print("Class counts (0=Down,1=Flat,2=Up):", {int(k): int(v) for k, v in cls_counts.items()})


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 50)
    main()
