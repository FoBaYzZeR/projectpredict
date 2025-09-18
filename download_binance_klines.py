#!/usr/bin/env python3
"""
Download & extract Binance daily 1m klines over a date range.
Defaults target: BTCUSDC 1m, 2022-08-31 .. 2025-08-31 (inclusive).

Example:
  python download_binance_klines.py \
    --symbol BTCUSDC --interval 1m \
    --start 2022-08-31 --end 2025-08-31 \
    --out ./binance_1m \
    --workers 8
"""

import argparse
import concurrent.futures as cf
import contextlib
import os
from pathlib import Path
from datetime import datetime, timedelta
import zipfile
import io
import sys
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{yyyy}-{mm}-{dd}.zip"

def daterange(start_date, end_date):
    # Inclusive date range
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)

def build_url(symbol, interval, day):
    return BASE_URL.format(
        symbol=symbol,
        interval=interval,
        yyyy=day.strftime("%Y"),
        mm=day.strftime("%m"),
        dd=day.strftime("%d"),
    )

def make_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "klines-downloader/1.0 (+https://github.com/)",
        "Accept": "*/*",
    })
    return session

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def target_csv_path(outdir: Path, symbol: str, interval: str, day: datetime):
    # Binance CSV inside ZIP is typically: {symbol}-{interval}-{yyyy}-{mm}-{dd}.csv
    return outdir / f"{symbol}-{interval}-{day.strftime('%Y-%m-%d')}.csv"

def head_exists(session: requests.Session, url: str) -> bool:
    # Quick existence check to skip 404s without downloading the body
    try:
        r = session.head(url, timeout=15)
        return r.status_code == 200
    except requests.RequestException:
        return False

def download_and_extract_one(args):
    (day, symbol, interval, outdir, keep_zips) = args
    session = make_session()

    url = build_url(symbol, interval, day)
    csv_path = target_csv_path(outdir, symbol, interval, day)
    zip_path = outdir / f"{symbol}-{interval}-{day.strftime('%Y-%m-%d')}.zip"

    # Resume: Skip if CSV already exists and looks sane (non-empty)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return (day, "skip_exists")

    # Skip fast if URL is missing
    if not head_exists(session, url):
        return (day, "not_found")

    try:
        with session.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                return (day, f"http_{r.status_code}")

            # Write ZIP to disk (optional; we could stream to memory, but this is safer on large ranges)
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                    if chunk:
                        f.write(chunk)

        # Validate & extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                # Corrupted zip; remove and signal retryable failure
                with contextlib.suppress(Exception):
                    zip_path.unlink()
                return (day, "corrupt_zip")

            # Extract expected CSV; if multiple, extract all (but usually one)
            members = zf.namelist()
            if not members:
                with contextlib.suppress(Exception):
                    zip_path.unlink()
                return (day, "empty_zip")

            # Extract to temp memory then write to target to avoid partial files
            extracted_any = False
            for name in members:
                if not name.lower().endswith(".csv"):
                    continue
                data = zf.read(name)
                if len(data) == 0:
                    continue
                tmp_path = csv_path.with_suffix(".csv.part")
                with open(tmp_path, "wb") as f:
                    f.write(data)
                os.replace(tmp_path, csv_path)
                extracted_any = True

            if not extracted_any:
                with contextlib.suppress(Exception):
                    zip_path.unlink()
                return (day, "no_csv_in_zip")

        # Delete ZIP if requested (default)
        if not keep_zips:
            with contextlib.suppress(Exception):
                zip_path.unlink()

        return (day, "ok")

    except requests.RequestException:
        # Network hiccup; caller may retry a few times if desired
        with contextlib.suppress(Exception):
            if zip_path.exists():
                zip_path.unlink()
        return (day, "network_err")
    except zipfile.BadZipFile:
        with contextlib.suppress(Exception):
            if zip_path.exists():
                zip_path.unlink()
        return (day, "bad_zip")
    except Exception as e:
        # Unknown error; clean up partials
        with contextlib.suppress(Exception):
            if zip_path.exists():
                zip_path.unlink()
            tmp = csv_path.with_suffix(".csv.part")
            if tmp.exists():
                tmp.unlink()
        return (day, f"error:{type(e).__name__}")

def main():
    parser = argparse.ArgumentParser(description="Download & extract Binance daily klines.")
    parser.add_argument("--symbol", default="BTCUSDC", help="Trading pair symbol (e.g., BTCUSDT or BTCUSDC)")
    parser.add_argument("--interval", default="1m", help="Kline interval (e.g., 1m, 5m, 1h)")
    parser.add_argument("--start", default="2022-08-31", help="Start date (YYYY-MM-DD), inclusive")
    parser.add_argument("--end", default="2025-08-31", help="End date (YYYY-MM-DD), inclusive")
    parser.add_argument("--out", default="./binance_klines", help="Output directory for extracted CSVs")
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads")
    parser.add_argument("--keep-zips", action="store_true", help="Keep ZIP files after extraction")
    parser.add_argument("--max-retries", type=int, default=2, help="Per-day task retry attempts on recoverable errors")

    args = parser.parse_args()

    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError:
        print("ERROR: Dates must be in YYYY-MM-DD format.", file=sys.stderr)
        sys.exit(1)

    if end_date < start_date:
        print("ERROR: end date must be >= start date.", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.out)
    ensure_dir(outdir)

    days = list(daterange(start_date, end_date))
    total = len(days)
    print(f"Downloading {total} day(s) for {args.symbol} {args.interval} into {outdir.resolve()}")

    # Prepare task args
    base_task_args = [
        (datetime.combine(day, datetime.min.time()), args.symbol, args.interval, outdir, args.keep_zips)
        for day in days
    ]

    # Simple retry loop around the thread pool
    remaining = base_task_args
    attempt = 0
    summary = {}
    while remaining and attempt <= args.max_retries:
        attempt += 1
        print(f"\nAttempt {attempt}: {len(remaining)} day(s) to process...")
        new_remaining = []
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(download_and_extract_one, task) for task in remaining]
            for fut in cf.as_completed(futures):
                day, status = fut.result()
                day_str = day.strftime("%Y-%m-%d")
                summary.setdefault(status, 0)
                summary[status] += 1

                if status in ("ok", "skip_exists", "not_found"):
                    print(f"[{day_str}] {status}")
                else:
                    print(f"[{day_str}] {status} -> will retry" if attempt <= args.max_retries else f"[{day_str}] {status}")

                # Decide if we should retry this day
                if status in ("network_err", "corrupt_zip", "bad_zip", "http_429", "http_500", "http_502", "http_503", "http_504", "error:TimeoutError"):
                    new_remaining.append((
                        day, args.symbol, args.interval, outdir, args.keep_zips
                    ))
        remaining = new_remaining

    # Final report
    print("\nSummary:")
    total_processed = sum(summary.values()) if summary else 0
    for k, v in sorted(summary.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:>12}: {v}")
    print(f"Processed {total_processed} task result(s).")
    print("Done.")

if __name__ == "__main__":
    main()
