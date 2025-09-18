import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- classification metrics ---
def classification_metrics(df):
    y_true = df["true"]
    y_pred = df["pred"]
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))
    print()
    print("Pred distribution (0=Down,1=Flat,2=Up):", dict(df["pred"].value_counts()))
    print("True distribution  (0=Down,1=Flat,2=Up):", dict(df["true"].value_counts()))

# --- load prices ---
def load_prices(parquet_path):
    dfp = pd.read_parquet(parquet_path, columns=["datetime","close"])
    dfp["datetime"] = pd.to_datetime(dfp["datetime"], utc=True)
    return dfp.set_index("datetime").sort_index()

# --- backtest ---
def backtest(preds, px, horizon=60, fee=0.0004, slippage=0.0001, long_only=False, short_only=False):
    trades = []
    for _, row in preds.iterrows():
        pred = row["pred"]
        conf = max(row["p_down"], row["p_flat"], row["p_up"])
        if pred == 1:  # Flat
            continue
        if long_only and pred != 2:
            continue
        if short_only and pred != 0:
            continue
        entry_time = row["datetime"]
        exit_time = entry_time + pd.Timedelta(minutes=horizon)
        if entry_time not in px.index or exit_time not in px.index:
            continue
        entry_price = px.at[entry_time, "close"]
        exit_price  = px.at[exit_time,  "close"]
        side = 1 if pred == 2 else -1
        ret = (exit_price / entry_price - 1) * side
        ret -= (fee + slippage) * 2
        trades.append({"entry": entry_time, "side": side, "ret": ret})
    trades = pd.DataFrame(trades)
    if trades.empty:
        return trades
    trades["equity"] = (1 + trades["ret"]).cumprod()
    return trades

# --- analyze ---
def analyze_trades(trades):
    if trades.empty:
        print("No trades after filtering.")
        return
    n = len(trades)
    win_rate = (trades["ret"] > 0).mean()
    sharpe = np.mean(trades["ret"]) / (np.std(trades["ret"]) + 1e-9) * np.sqrt(252*24*60)  # 252 trading days, minutes
    total_ret = trades["equity"].iloc[-1] - 1
    print("\n=== Backtest Results ===")
    print(f"Trades: {n}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Total return: {total_ret:.2%}")
    plt.figure(figsize=(10,5))
    plt.plot(trades["entry"], trades["equity"], label="Equity Curve")
    plt.axhline(1.0, ls="--", c="k", alpha=0.5)
    plt.title("Backtest Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity (start=1.0)")
    plt.legend()
    plt.show()

# --- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs_medium_lr1e4/test_predictions.csv")
    ap.add_argument("--prices", default="btc_1m_2022_2025.parquet")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slippage", type=float, default=0.0001)
    ap.add_argument("--min-prob", type=float, default=0.0)
    ap.add_argument("--long-only", action="store_true", help="Take only long (Up) trades")
    ap.add_argument("--short-only", action="store_true", help="Take only short (Down) trades")
    args = ap.parse_args()

    preds = pd.read_csv(args.pred, parse_dates=["datetime"])
    px = load_prices(args.prices)

    # confidence filter
    if args.min_prob > 0:
        mask = preds[["p_down","p_flat","p_up"]].max(axis=1) >= args.min_prob
        preds = preds[mask]
        print(f"Confidence filter: kept {len(preds)}/{len(mask)} rows (min_prob={args.min_prob}).")

    classification_metrics(preds)
    trades = backtest(preds, px, horizon=args.horizon, fee=args.fee, slippage=args.slippage,
                      long_only=args.long_only, short_only=args.short_only)
    analyze_trades(trades)

if __name__ == "__main__":
    main()
