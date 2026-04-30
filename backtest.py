import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from model import fetch_btc_data, predict_multi, score_prediction

WARMUP   = 250
N_PREDS  = 720
N_SIM    = 10_000
OUT_FILE = "backtest_results.jsonl"
TOTAL    = WARMUP + N_PREDS


def evaluate(predictions):
    def _m(k): return float(np.mean([p[k] for p in predictions]))
    return {
        "n_predictions":   len(predictions),
        "coverage_95":     _m("hit_95"),
        "avg_width":       _m("width_95"),
        "mean_winkler_95": _m("winkler_95"),
        "coverage_80":     _m("hit_80"),
        "coverage_50":     _m("hit_50"),
        "mean_winkler_80": _m("winkler_80"),
        "mean_winkler_50": _m("winkler_50"),
    }


def _row(label, cov, target, width, winkler):
    gap = cov - target
    sym = "OK" if abs(gap) <= 0.025 else ("wide" if gap > 0 else "narrow")
    return (f"  {label:<10} obs={cov:.4f}  target={target:.2f}"
            f"  delta={gap:+.4f}  {sym:<8}"
            f"  avg_width=${width:>8,.0f}  winkler={winkler:>10,.0f}")


def run_backtest(total=TOTAL, warmup=WARMUP, n_sim=N_SIM,
                 out_file=OUT_FILE):
    print()
    print("-"*62)
    print("  BTC Forecaster  -  Part A: 30-Day Walk-Forward Backtest")
    print("-"*62)
    print(f"  Fetching {total} BTCUSDT 1-h bars from Binance ...")

    df    = fetch_btc_data(limit=total)
    close = df["close"]
    times = df.index
    print(f"  Got {len(df)} bars  ({times[0].date()}  to  {times[-1].date()})")

    start  = max(warmup, len(df) - N_PREDS)
    end    = len(df) - 1
    n_iter = end - start
    print(f"  Walk-forward: predicting bars [{start}..{end-1}]  ({n_iter} steps)")
    print()

    predictions = []
    for i in tqdm(range(start, end), desc="  Backtesting", unit="bar",
                  ncols=68, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
        history = close.iloc[:i]
        actual  = float(close.iloc[i])
        try:
            pred = predict_multi(history, n_sim=n_sim)
        except ValueError:
            continue
        scores = score_prediction(pred, actual)
        predictions.append({
            "timestamp": times[i].isoformat(),
            "S0":        pred.current_price,
            "lower_95":  pred.lower_95,  "upper_95":  pred.upper_95,
            "lower_80":  pred.lower_80,  "upper_80":  pred.upper_80,
            "lower_50":  pred.lower_50,  "upper_50":  pred.upper_50,
            "regime":    pred.regime,    "nu":        pred.nu,
            "vol_ann":   pred.volatility_ann,
            "actual":    actual,
            **scores,
            "hit":    scores["hit_95"],
            "width":  scores["width_95"],
            "winkler":scores["winkler_95"],
        })

    predictions = predictions[-N_PREDS:]
    m = evaluate(predictions)

    w80 = float(np.mean([p["width_80"] for p in predictions]))
    w50 = float(np.mean([p["width_50"] for p in predictions]))

    print()
    print("="*62)
    print(f"  BACKTEST RESULTS  ({m['n_predictions']} predictions)")
    print("="*62)
    print(_row("95% band", m["coverage_95"], 0.95, m["avg_width"],    m["mean_winkler_95"]))
    print(_row("80% band", m["coverage_80"], 0.80, w80,               m["mean_winkler_80"]))
    print(_row("50% band", m["coverage_50"], 0.50, w50,               m["mean_winkler_50"]))
    print("-"*62)
    print("  PASTE INTO SUBMISSION FORM:")
    print(f"    coverage_95     = {m['coverage_95']:.4f}")
    print(f"    mean_winkler_95 = {m['mean_winkler_95']:,.2f}")
    print("="*62)
    print()

    out = Path(out_file)
    with out.open("w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(predictions)} rows to {out.resolve()}")
    print()
    return predictions, m


if __name__ == "__main__":
    run_backtest()
