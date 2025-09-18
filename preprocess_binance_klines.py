#!/usr/bin/env python3
"""
Preprocess Binance daily 1m kline CSVs into a clean, continuous Parquet dataset.

- Robust timestamp normalization (sec/ms/µs) with sanity bounds (2010..2035 UTC).
- Drops invalid rows/timestamps, deduplicates by open_time.
- Enforces a continuous 1-minute time index; FFill OHLC, zero-fill volumes/trades.
- Efficient dtypes (float32/int32) for NN training.
- Saves a single Parquet (and optionally per-year Parquets).

Example:
  python preprocess_binance_klines.py \
    --in ./btc_1m_3y \
    --out ./btc_1m_2022_2025.parquet \
    --symbol BTCUSDC \
    --interval 1m \
    --start 2022-08-31 \
    --end 2025-08-31 \
    --workers 8 \
    --save-yearly
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    # fallback if tqdm not installed
    def tqdm(x, **kwargs):
        return x

SAFE_MIN = pd.Timestamp("2010-01-01", tz="UTC")
SAFE_MAX = pd.Timestamp("2035-01-01", tz="UTC")

FNAME_RE = re.compile(r"(?P<symbol>[A-Z0-9]+)-(?P<interval>[0-9a-zA-Z]+)-(?P<yyyy>\d{4})-(?P<mm>\d{2})-(?P<dd>\d{2})\.csv$")

OHLC_COLS = ["open", "high", "low", "close"]
VOL_COLS  = ["volume", "num_trades", "taker_buy_base", "taker_buy_quote"]

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess Binance daily 1m klines CSVs into a continuous Parquet.")
    p.add_argument("--in", dest="indir", required=True, help="Input directory with extracted daily CSVs")
    p.add_argument("--out", dest="outfile", required=True, help="Output Parquet file path")
    p.add_argument("--symbol", default="BTCUSDC", help="Symbol to include (file name prefix), e.g. BTCUSDC")
    p.add_argument("--interval", default="1m", help="Interval in file name, e.g. 1m")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD) inclusive; default = min found")
    p.add_argument("--end",   default=None, help="End date (YYYY-MM-DD) inclusive; default = max found")
    p.add_argument("--workers", type=int, default=8, help="Parallel CSV readers")
    p.add_argument("--save-yearly", action="store_true", help="Also save per-year Parquets next to --out")
    p.add_argument("--log-fixed", action="store_true", help="Write a CSV log of rows with normalized/fixed timestamps")
    return p.parse_args()

def _normalize_epoch_ms(x):
    """
    Normalize Binance kline 'open_time' values to milliseconds since epoch.
    Accepts string/int. Handles sec/ms/µs; returns np.nan on nonsense.
    """
    if pd.isna(x):
        return np.nan
    try:
        s = str(x).strip()
        if not s.isdigit():
            return np.nan
        v = int(s)
        # seconds ~ 1e9–1e10; ms ~ 1e12–1e13; µs ~ 1e15–1e16
        if v < 10**11:         # seconds
            v *= 1000
        elif v > 10**14:       # microseconds
            v //= 1000
        return v
    except Exception:
        return np.nan

def _coerce_datetime_from_ms(ms_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(ms_series, unit="ms", utc=True, errors="coerce")
    dt = dt.where((dt >= SAFE_MIN) & (dt < SAFE_MAX))
    return dt

def load_and_clean_csv(file: Path, log_fixed: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load one daily CSV, fix timestamps, return cleaned DF.
    Returns (df, fixes_log_df)
    """
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.read_csv(
        file,
        header=None,
        names=cols,
        dtype={
            "open_time": "string",
            "open": "float64", "high": "float64", "low": "float64", "close": "float64",
            "volume": "float64", "quote_asset_volume": "float64",
            "num_trades": "Int64",
            "taker_buy_base": "float64", "taker_buy_quote": "float64",
            "close_time": "string", "ignore": "string",
        },
        engine="c",
        low_memory=False
    )

    # Keep a copy for logging if requested
    fixes = None
    if log_fixed:
        fixes = df[["open_time"]].copy()
        fixes.rename(columns={"open_time":"open_time_raw"}, inplace=True)

    ms = df["open_time"].map(_normalize_epoch_ms)
    dt = _coerce_datetime_from_ms(ms)

    mask_valid = dt.notna()
    dropped = int((~mask_valid).sum())
    if dropped:
        print(f"[{file.name}] dropped {dropped} invalid timestamp row(s)")

    df = df.loc[mask_valid].copy()
    df["open_time"] = dt.loc[mask_valid]

    if log_fixed:
        fixes = fixes.loc[mask_valid].copy()
        fixes["open_time_ms_norm"] = ms.loc[mask_valid].values
        fixes["open_time_dt"] = df["open_time"].values
        fixes["file"] = file.name

    # Keep relevant columns
    df = df[["open_time", "open", "high", "low", "close", "volume",
             "num_trades", "taker_buy_base", "taker_buy_quote"]]

    # Dtypes
    df["num_trades"] = df["num_trades"].fillna(0).astype("int32")
    for c in ["open","high","low","close","volume","taker_buy_base","taker_buy_quote"]:
        df[c] = df[c].astype("float32")

    # Sort & de-dup
    df = df.sort_values("open_time").drop_duplicates("open_time", keep="last")

    # Drop fully zeroed OHLC rows (rare corruption)
    zero_ohlc = (df[OHLC_COLS].sum(axis=1) == 0)
    if zero_ohlc.any():
        n = int(zero_ohlc.sum())
        print(f"[{file.name}] dropped {n} row(s) with zeroed OHLC")
        df = df.loc[~zero_ohlc]

    return df, (fixes if log_fixed else pd.DataFrame())

def discover_files(indir: Path, symbol: str, interval: str, start: datetime|None, end: datetime|None) -> List[Path]:
    files = []
    for p in sorted(indir.glob(f"{symbol}-{interval}-*.csv")):
        m = FNAME_RE.search(p.name)
        if not m:
            continue
        yyyy, mm, dd = int(m["yyyy"]), int(m["mm"]), int(m["dd"])
        dt = datetime(yyyy, mm, dd, tzinfo=timezone.utc)
        if start and dt < start:
            continue
        if end and dt > end:
            continue
        files.append(p)
    return files

def main():
    args = parse_args()
    indir = Path(args.indir)
    outfile = Path(args.outfile)

    if not indir.exists():
        raise SystemExit(f"Input directory not found: {indir}")

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.start else None
    end   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end   else None

    files = discover_files(indir, args.symbol, args.interval, start, end)
    if not files:
        raise SystemExit("No matching CSV files found. Check --in, --symbol, --interval, --start/--end.")

    print(f"Found {len(files)} daily files between "
          f"{files[0].name.split(args.symbol + '-' + args.interval + '-')[1].rstrip('.csv')} and "
          f"{files[-1].name.split(args.symbol + '-' + args.interval + '-')[1].rstrip('.csv')}")

    dfs: List[pd.DataFrame] = []
    fix_logs: List[pd.DataFrame] = []

    # Parallel load
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(load_and_clean_csv, f, args.log_fixed): f for f in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Loading/cleaning"):
            f = futs[fut]
            try:
                df_part, fix_log = fut.result()
                if not df_part.empty:
                    dfs.append(df_part)
                if args.log_fixed and not fix_log.empty:
                    fix_logs.append(fix_log)
            except Exception as e:
                print(f"Skipping {f.name}: {e}")

    if not dfs:
        raise SystemExit("No data loaded after cleaning. Aborting.")

    # Merge all
    df = pd.concat(dfs, axis=0, ignore_index=True)
    # Deduplicate & sort (again, across files)
    df = df.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)

    # Audit continuity BEFORE reindexing
    first_dt = df["open_time"].iloc[0]
    last_dt  = df["open_time"].iloc[-1]
    full_index = pd.date_range(first_dt, last_dt, freq="1min", tz="UTC")
    missing_before = len(full_index.difference(df["open_time"]))
    print(f"Continuity audit: missing minutes BEFORE fix = {missing_before}")

    # Reindex continuous timeline
    df = df.set_index("open_time").reindex(full_index)

    # Fill gaps: forward-fill OHLC, zero-fill volumes/trades
    df[OHLC_COLS] = df[OHLC_COLS].ffill()
    df[VOL_COLS] = df[VOL_COLS].fillna(0)

    # After fix: how many NaNs left in OHLC (if start had NaNs, drop until first valid)
    nan_ohlc_rows = df[OHLC_COLS].isna().any(axis=1)
    if nan_ohlc_rows.any():
        n = int(nan_ohlc_rows.sum())
        print(f"Dropping {n} leading row(s) with NaN OHLC after reindex (pre-first valid candle).")
        df = df.loc[~nan_ohlc_rows]

    # Final tidy
    df = df.reset_index().rename(columns={"index": "datetime"})

    # Ensure dtypes (ffill might upcast)
    for c in ["open","high","low","close","volume","taker_buy_base","taker_buy_quote"]:
        df[c] = df[c].astype("float32")
    df["num_trades"] = df["num_trades"].astype("int32")

    # Audit continuity AFTER fix
    span = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds() / 60.0 + 1
    expected_rows = int(span)
    missing_after = expected_rows - len(df)
    print(f"Continuity audit: expected rows after fix = {expected_rows}, actual = {len(df)}, missing AFTER fix = {missing_after}")

    # Save Parquet
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outfile, index=False, engine="pyarrow", compression="snappy")
    print(f"Saved {outfile}  shape={df.shape}  size≈{outfile.stat().st_size/1e6:.1f} MB")
    print(f"Range: {df['datetime'].iloc[0]}  ->  {df['datetime'].iloc[-1]}")

    # Optional: per-year Parquets
    if args.save_yearly:
        outdir = outfile.parent
        df["year"] = df["datetime"].dt.year
        for y, dfy in df.groupby("year"):
            yfile = outdir / f"{outfile.stem}.{y}.parquet"
            dfg = dfy.drop(columns=["year"])
            dfg.to_parquet(yfile, index=False, engine="pyarrow", compression="snappy")
            print(f"Saved {yfile}  shape={dfg.shape}")

    # Optional: write fixed timestamp log
    if args.log_fixed and fix_logs:
        fixes = pd.concat(fix_logs, ignore_index=True)
        log_file = outfile.with_suffix(".fixed_timestamps.csv")
        fixes.to_csv(log_file, index=False)
        print(f"Wrote fixed timestamp log: {log_file} (rows={len(fixes)})")

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    main()
