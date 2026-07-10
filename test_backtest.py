"""
backfill_walkforward_backtest.py — Walk-forward backtest of StockWise's
PRODUCTION recommender logic against real historical NSE data.

Why this instead of test_backtest.py:
test_backtest.py can only score days that scheduler.py has already run
live in production — i.e. it needs weeks of the app running 24/7 before
it has anything to say. That's not realistic before a resume deadline.

This script instead downloads historical OHLC data once, then replays
recommender.py's *actual* scoring functions (_score_from_hist, the exact
top-5 ranking logic from generate_recommendations) day-by-day: for each
past trading day D, it only feeds the model data strictly BEFORE D —
exactly what the live 9:15 AM job would have seen — picks a top-5 the
same way production does, and compares against D's real open->close
move. No lookahead bias, no DB required. This is the honest way to get
a "backtested against N real trading days" number.

Usage:
    pip install yfinance pandas numpy --break-system-packages
    python backfill_walkforward_backtest.py --days 40 --lookback 10mo

Run this from the same folder as recommender.py and companies_india.csv.
"""
import argparse
import numpy as np
import yfinance as yf

from recommender import load_nifty50, _score_from_hist


def select_top5(results):
    """Exact copy of generate_recommendations()'s ranking logic in
    recommender.py, so the backtest picks are identical to what
    production would have chosen."""
    bullish = [r for r in results if r["predicted_gain"] > 0]

    if len(bullish) >= 5:
        for r in bullish:
            r["_rank_score"] = r["score"] * 0.6 + min(r["predicted_gain"] * 10, 40)
        bullish.sort(key=lambda x: x["_rank_score"], reverse=True)
        pool = bullish
    elif bullish:
        for r in results:
            r["_rank_score"] = r["score"] * 0.6 + min(r["predicted_gain"] * 10, 40)
        results.sort(key=lambda x: x["_rank_score"], reverse=True)
        pool = results
    else:
        for r in results:
            r["_rank_score"] = r["score"] - abs(r["predicted_gain"]) * 5
        results.sort(key=lambda x: x["_rank_score"], reverse=True)
        pool = results

    top5 = pool[:5]
    for r in top5:
        r.pop("_rank_score", None)
    return top5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=40,
                         help="number of past trading days to backtest")
    parser.add_argument("--lookback", type=str, default="10mo",
                         help="yfinance history window to download (needs to "
                              "cover --days plus ~25 warm-up days per symbol)")
    args = parser.parse_args()

    stocks = load_nifty50()
    symbols = [s for s, _ in stocks]
    company_map = {s: c for s, c in stocks}
    if not symbols:
        print("No symbols loaded from companies_india.csv — check the file exists here.")
        return

    print(f"Downloading {args.lookback} of history for {len(symbols)} symbols "
          f"(one batched call)...")
    raw = yf.download(symbols, period=args.lookback, group_by="ticker",
                       progress=False, threads=True)

    frames = {}
    for sym in symbols:
        try:
            df = raw[sym].dropna(how="any").copy()
            df.columns = [c.capitalize() for c in df.columns]
            if "Close" in df.columns and len(df) > 40:
                frames[sym] = df
        except Exception:
            continue
    print(f"Usable history for {len(frames)}/{len(symbols)} symbols.")
    if not frames:
        print("No usable data downloaded — check network/yfinance connectivity.")
        return

    ref_sym = max(frames, key=lambda s: len(frames[s]))
    test_dates = frames[ref_sym].index[-args.days:]

    paired_rows = []
    market_baseline_by_day = {}

    for D in test_dates:
        day_results = []
        day_actuals_all = []

        for sym, df in frames.items():
            if D not in df.index:
                continue
            hist_upto = df.loc[df.index < D]
            if len(hist_upto) < 25:
                continue

            scored = _score_from_hist(sym, company_map.get(sym, sym), hist_upto)
            if scored:
                day_results.append(scored)

            try:
                o, c = float(df.loc[D, "Open"]), float(df.loc[D, "Close"])
                if o:
                    day_actuals_all.append((c - o) / o * 100)
            except Exception:
                pass

        if not day_results:
            continue

        for r in select_top5(day_results):
            df = frames[r["symbol"]]
            try:
                o, c = float(df.loc[D, "Open"]), float(df.loc[D, "Close"])
                actual_pct = (c - o) / o * 100
            except Exception:
                continue
            paired_rows.append({
                "date": D, "symbol": r["symbol"],
                "predicted_gain": r["predicted_gain"],
                "target_price": r["target_price"],
                "actual_pct_change": actual_pct, "close_price": c,
            })

        if day_actuals_all:
            market_baseline_by_day[D] = float(np.mean(day_actuals_all))

    if len(paired_rows) < 10:
        print(f"Only {len(paired_rows)} paired rows generated — try a larger "
              f"--days / --lookback, or check that yfinance returned data.")
        return

    preds = np.array([r["predicted_gain"] for r in paired_rows])
    actuals = np.array([r["actual_pct_change"] for r in paired_rows])

    directional_accuracy = (np.sign(preds) == np.sign(actuals)).mean() * 100
    mae = np.mean(np.abs(preds - actuals))
    corr = np.corrcoef(preds, actuals)[0, 1] if len(preds) > 1 else float("nan")
    hits = sum(1 for r in paired_rows if r["close_price"] >= r["target_price"])
    target_hit_rate = hits / len(paired_rows) * 100

    by_day = {}
    for r in paired_rows:
        by_day.setdefault(r["date"], []).append(r)

    daily_top5_returns, daily_market_returns = [], []
    for day, rows in sorted(by_day.items()):
        daily_top5_returns.append(np.mean([r["actual_pct_change"] for r in rows]))
        if day in market_baseline_by_day:
            daily_market_returns.append(market_baseline_by_day[day])

    n_days = len(daily_top5_returns)
    cumulative_strategy = np.sum(daily_top5_returns)
    cumulative_market = np.sum(daily_market_returns) if daily_market_returns else float("nan")

    print("=" * 64)
    print(f"WALK-FORWARD BACKTEST — {len(paired_rows)} pick/outcome pairs "
          f"across {n_days} real trading days")
    print("=" * 64)
    print(f"Directional accuracy:        {directional_accuracy:.1f}%")
    print(f"Mean Absolute Error:         {mae:.2f} pct points")
    print(f"Predicted-vs-actual corr:    {corr:.3f}")
    print(f"Target price hit rate:       {target_hit_rate:.1f}%")
    print("-" * 64)
    print(f"Avg daily top-5 return:      {np.mean(daily_top5_returns):+.3f}%")
    print(f"Avg daily market baseline:   {np.mean(daily_market_returns):+.3f}%")
    print(f"Cumulative top-5 return:     {cumulative_strategy:+.2f}% over {n_days} days")
    print(f"Cumulative market baseline:  {cumulative_market:+.2f}% over {n_days} days")
    print("=" * 64)
    print("\nResume-ready line, e.g.:")
    print(f'  "Walk-forward backtested AI stock recommender against '
          f'{n_days} real NSE trading days with no lookahead bias: '
          f'{directional_accuracy:.0f}% directional accuracy, '
          f'{cumulative_strategy - cumulative_market:+.1f}pp cumulative excess '
          f'return vs. Nifty-50 baseline, {mae:.2f}pp MAE."')


if __name__ == "__main__":
    main()