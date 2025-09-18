#!/usr/bin/env python3
# grid_backtest.py (robust, non-overlapping, log compounding)

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Utilities ----------
def autodetect_cols(df: pd.DataFrame):
    for k in ["true_class", "true", "label"]:
        if k in df.columns: y_true_col = k; break
    else: raise KeyError("GT column not found (expected true_class/true/label).")

    for k in ["pred_class", "pred"]:
        if k in df.columns: y_pred_col = k; break
    else: raise KeyError("Pred column not found (expected pred_class/pred).")

    have_probs = all(c in df.columns for c in ["p_down", "p_flat", "p_up"])
    return y_true_col, y_pred_col, have_probs

def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    if "datetime" not in df.columns:
        raise KeyError("Predictions file must contain a 'datetime' column.")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)

def load_prices(parquet_path: str) -> pd.DataFrame:
    dfp = pd.read_parquet(parquet_path, columns=["datetime", "close"])
    dfp["datetime"] = pd.to_datetime(dfp["datetime"], utc=True)
    return dfp.set_index("datetime").sort_index()

def attach_prices(preds: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = preds.merge(
        prices.rename(columns={"close":"price"}),
        left_on="datetime", right_index=True, how="left"
    )
    miss = int(out["price"].isna().sum())
    if miss:
        print(f"Warning: {miss} rows had no matching price and were dropped.")
        out = out.dropna(subset=["price"])
    return out.sort_values("datetime").reset_index(drop=True)

def max_drawdown_from_equity(equity: pd.Series) -> float:
    if equity.empty: return 0.0
    peak = equity.cummax()
    dd = (equity/peak - 1.0).min()
    return -float(dd)

# ---------- Backtest (non-overlapping by default) ----------
def backtest_with_px(
    df: pd.DataFrame,
    prices_idx: pd.DataFrame,
    y_pred_col: str,
    horizon: int,
    fee: float,
    slippage: float,
    direction: str,         # "long", "short", or "both"
    min_prob: float,
    have_probs: bool,
    allow_overlap: bool,    # False => max 1 open trade at a time
):
    preds = df.copy()

    # confidence filter
    if have_probs and min_prob > 0.0:
        mask = preds[["p_down","p_flat","p_up"]].max(axis=1) >= min_prob
        preds = preds[mask]

    # iterate in time order
    trades = []
    open_until = pd.Timestamp.min.tz_localize("UTC")  # when we can open next trade
    max_possible_trades = int(len(preds))

    for _, row in preds.iterrows():
        cls = int(row[y_pred_col])
        if cls == 1:  # Flat -> skip
            continue
        if direction == "long" and cls != 2:  # only Up
            continue
        if direction == "short" and cls != 0: # only Down
            continue

        entry_time = row["datetime"]
        exit_time = entry_time + pd.Timedelta(minutes=horizon)

        # no-overlap enforcement
        if not allow_overlap and entry_time < open_until:
            continue

        try:
            entry = prices_idx.at[entry_time, "close"]
            exitp = prices_idx.at[exit_time, "close"]
        except KeyError:
            continue

        side = 1 if cls == 2 else -1
        gross = (exitp/entry - 1.0) * side
        cost = (fee + slippage) * 2.0
        net = gross - cost

        trades.append({"entry": entry_time, "ret": net, "side": side})
        if not allow_overlap:
            open_until = exit_time  # block new entries until we exit

    trades = pd.DataFrame(trades)
    if trades.empty:
        return {
            "trades": 0, "win_rate": np.nan, "sharpe": np.nan,
            "total_return": 0.0, "max_dd": 0.0, "unstable": False,
        }

    trades = trades.sort_values("entry").reset_index(drop=True)

    # log compounding (stable)
    log_rets = np.log1p(trades["ret"].clip(lower=-0.999999))  # avoid log(<=0)
    equity = np.exp(log_rets.cumsum())

    # minute-annualization for Sharpe: 252*24*60 minutes/year
    r_mean, r_std = trades["ret"].mean(), trades["ret"].std()
    sharpe = (r_mean / (r_std + 1e-12)) * np.sqrt(252*24*60 / horizon)
    win_rate = float((trades["ret"] > 0).mean())
    total_return = float(equity.iloc[-1] - 1.0)
    mdd = max_drawdown_from_equity(equity)

    # sanity checks for unrealistic results
    unstable = False
    if equity.max() / max(equity.min(), 1e-12) > 1e6:
        unstable = True
    if allow_overlap and len(trades) > max_possible_trades * 0.9:
        unstable = True

    return {
        "trades": int(len(trades)),
        "win_rate": win_rate,
        "sharpe": float(sharpe),
        "total_return": total_return,
        "max_dd": mdd,
        "unstable": unstable,
    }

# ---------- Grid search ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs_medium_lr1e4/test_predictions.csv")
    ap.add_argument("--prices", default="btc_1m_2022_2025.parquet")
    ap.add_argument("--direction", choices=["long","short","both"], default="long")
    ap.add_argument("--fees", type=float, default=0.0004)
    ap.add_argument("--slippage", type=float, default=0.0001)
    ap.add_argument("--horizons", type=str, default="60,120,240,480")
    ap.add_argument("--probs", type=str, default="0.95,0.97,0.975,0.98")
    ap.add_argument("--out", type=str, default="grid_results.csv")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--allow-overlap", action="store_true",
                    help="Allow overlapping trades (defaults to OFF for robustness).")
    args = ap.parse_args()

    preds = load_predictions(args.pred)
    _, y_pred_col, have_probs = autodetect_cols(preds)
    prices = load_prices(args.prices)
    preds = attach_prices(preds, prices)

    horizons = [int(x) for x in args.horizons.split(",")]
    thresholds = [float(x) for x in args.probs.split(",")]

    rows = []
    for h in horizons:
        for th in thresholds:
            res = backtest_with_px(
                preds, prices, y_pred_col,
                horizon=h, fee=args.fees, slippage=args.slippage,
                direction=args.direction, min_prob=th, have_probs=have_probs,
                allow_overlap=args.allow_overlap,
            )
            rows.append({
                "direction": args.direction, "horizon": h, "min_prob": th,
                "trades": res["trades"], "win_rate": res["win_rate"],
                "sharpe": res["sharpe"], "total_return": res["total_return"],
                "max_dd": res["max_dd"], "unstable": res["unstable"],
            })

    dfres = pd.DataFrame(rows).sort_values(["unstable","total_return"], ascending=[True, False])
    dfres.to_csv(args.out, index=False)

    # print top stable combos
    stable = dfres[dfres["unstable"] == False]
    print("\nTop stable results by total_return:")
    if not stable.empty:
        print(stable.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("(none — all runs flagged unstable)")

    # heatmap of total returns (stable only)
    if args.plot and not stable.empty:
        pivot = stable.pivot(index="horizon", columns="min_prob", values="total_return")
        plt.figure(figsize=(8,5))
        im = plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.colorbar(im, label="Total return")
        plt.xticks(range(len(pivot.columns)), [f"{c:.3f}" for c in pivot.columns], rotation=45)
        plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
        plt.title(f"Total Return Heatmap ({args.direction}-only, non-overlap)")
        plt.xlabel("min_prob")
        plt.ylabel("horizon (min)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
