"""
recommender.py — Fast AI Stock Recommender for Nifty 50
Runs every morning at 9:15 AM IST.

Ranking is driven by a TRAINED model (HistGradientBoostingRegressor)
whose target is exactly what backfill_walkforward_backtest.py measures:
next-day open->close % change. The model is trained on pooled
walk-forward samples across all Nifty 50 symbols (see
train_ranking_model()) and cached to disk, refreshed weekly like
ml_model.py's per-symbol models.

If no trained model is cached yet (cold start, or training data
unavailable), scoring falls back to the original hand-tuned heuristic
(_estimate_gain_heuristic) so the pipeline never breaks:
  1. Momentum score  (40%) — 5d / 10d / 20d price trend
  2. Volatility score(30%) — lower vol = safer, higher score
  3. Volume surge    (20%) — unusual buying activity vs 20d avg
  4. Gap score       (10%) — today's open vs yesterday's close
These heuristic scores are still computed and shown in "reason" text
and used as a secondary quality/tie-break signal even when the trained
model is driving predicted_gain.
"""

import os, csv, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import psycopg2.extras
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IST = pytz.timezone("Asia/Kolkata")

# -----------------------------------------------------------------------------
# Trained ranking model cache
# -----------------------------------------------------------------------------

RANKER_MODEL_DIR = os.getenv("RANKER_MODEL_DIR", "ranker_model_cache")
RANKER_MODEL_PATH = os.path.join(RANKER_MODEL_DIR, "global_ranker.joblib")
RANKER_META_PATH = os.path.join(RANKER_MODEL_DIR, "global_ranker_meta.txt")
RANKER_EXPIRY_DAYS = int(os.getenv("RANKER_EXPIRY_DAYS", "7"))

os.makedirs(RANKER_MODEL_DIR, exist_ok=True)

_RANKER_CACHE = {"bundle": None, "checked_at": None}


def _ranker_is_fresh() -> bool:
    if not os.path.exists(RANKER_MODEL_PATH) or not os.path.exists(RANKER_META_PATH):
        return False
    try:
        with open(RANKER_META_PATH, "r", encoding="utf-8") as f:
            saved_at = datetime.fromisoformat(f.read().strip())
        return (datetime.now() - saved_at).days < RANKER_EXPIRY_DAYS
    except Exception:
        return False


def _load_ranker_model():
    """
    Load the trained global ranker, cached in memory. Returns None if no
    model has been trained yet (cold start) — callers must fall back to
    the heuristic in that case. Does NOT auto-retrain; call
    train_ranking_model() explicitly (e.g. a weekly cron), same pattern
    as ml_model.py's get_or_train_model().
    """
    if _RANKER_CACHE["bundle"] is not None:
        return _RANKER_CACHE["bundle"]
    if not os.path.exists(RANKER_MODEL_PATH):
        return None
    try:
        bundle = joblib.load(RANKER_MODEL_PATH)
        _RANKER_CACHE["bundle"] = bundle
        return bundle
    except Exception as e:
        print(f"[Recommender] Could not load ranker model: {e}")
        return None

def _ist_now():
    return datetime.now(IST)

def _is_market_day():
    return _ist_now().weekday() < 5


# ── Load all Nifty 50 symbols ─────────────────────────────────────────────────

def load_nifty50():
    path = os.path.join(os.path.dirname(__file__), "companies_india.csv")
    out  = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "Symbol" in row and "Company" in row:
                    out.append((row["Symbol"].strip(), row["Company"].strip()))
    except Exception as e:
        print(f"[Recommender] Error loading companies: {e}")
    return out


# ── Fast batch data fetch via yfinance ───────────────────────────────────────
# TwelveData free plan does NOT support batch NSE symbols and has 8 req/min
# limit — completely unusable for 50 stocks. yfinance is used exclusively.

def _batch_fetch(symbols: list, period: str = "30d") -> dict:
    """
    Fetch historical data for all symbols using yfinance.
    Uses ThreadPoolExecutor for parallel fetching.
    """
    if not symbols:
        return {}

    import time

    def _fetch_one_yf(sym):
        for attempt in range(2):
            try:
                ticker = yf.Ticker(sym)
                df = ticker.history(period=period)
                if df.empty:
                    time.sleep(0.5)
                    continue
                df.columns = [c.capitalize() for c in df.columns]
                df.index = pd.to_datetime(df.index).tz_localize(None)
                if "Close" in df.columns and len(df) >= 5:
                    return sym, df
            except Exception as e:
                print(f"[yfinance] {sym} attempt {attempt+1}: {e}")
                time.sleep(1)
        return sym, None

    result = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_one_yf, s): s for s in symbols}
        for fut in as_completed(futures):
            sym, df = fut.result()
            if df is not None:
                result[sym] = df

    print(f"[yfinance] Fetched {len(result)}/{len(symbols)} symbols")
    return result


# ── Scoring functions ─────────────────────────────────────────────────────────

def _momentum_score(hist: pd.DataFrame) -> float:
    if len(hist) < 21:
        return 50.0
    close = hist["Close"].values
    try:
        m5  = (close[-1] - close[-6])  / close[-6]  * 100
        m10 = (close[-1] - close[-11]) / close[-11] * 100
        m20 = (close[-1] - close[-21]) / close[-21] * 100
        score = (m5 * 0.5) + (m10 * 0.3) + (m20 * 0.2)
        return float(min(100, max(0, 50 + score * 5)))
    except Exception:
        return 50.0


def _volatility_score(hist: pd.DataFrame) -> float:
    if len(hist) < 10:
        return 50.0
    try:
        returns = hist["Close"].pct_change().dropna()
        vol     = returns.std() * 100
        score   = max(10, 100 - (vol * 20))
        return float(min(100, score))
    except Exception:
        return 50.0


def _volume_score(hist: pd.DataFrame) -> float:
    if len(hist) < 5:
        return 50.0
    try:
        avg_vol  = hist["Volume"].iloc[:-1].mean()
        last_vol = hist["Volume"].iloc[-1]
        if avg_vol == 0:
            return 50.0
        ratio = last_vol / avg_vol
        return float(min(100, ratio * 50))
    except Exception:
        return 50.0


def _gap_score(hist: pd.DataFrame) -> float:
    if len(hist) < 2:
        return 50.0
    try:
        prev_close = float(hist["Close"].iloc[-2])
        today_open = float(hist["Open"].iloc[-1])
        gap_pct    = ((today_open - prev_close) / prev_close) * 100
        return float(min(100, max(0, 50 + gap_pct * 25)))
    except Exception:
        return 50.0


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index — >50 is bullish territory."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _extract_ranking_features(hist: pd.DataFrame) -> np.ndarray | None:
    """
    Build the numeric feature vector used by the TRAINED ranking model.
    All features are stationary (ratios/returns/percentages) rather than
    raw price levels, for the same reason as ml_model.py's features: a
    tree model trained on 2 years of history shouldn't see feature values
    that later drift outside the range it learned from.

    Uses the same underlying signals as the heuristic (momentum,
    volatility, volume, gap, RSI) plus a couple of extras, but leaves them
    as continuous numbers instead of collapsing each into a hand-mapped
    0-100 score — that mapping throws away information the model could
    otherwise use.
    """
    if len(hist) < 21:
        return None
    try:
        closes = hist["Close"].values
        current = closes[-1]

        m5 = (closes[-1] - closes[-6]) / closes[-6] * 100
        m10 = (closes[-1] - closes[-11]) / closes[-11] * 100
        m20 = (closes[-1] - closes[-21]) / closes[-21] * 100

        returns = pd.Series(closes).pct_change().dropna()
        vol20 = float(returns.tail(20).std() * 100) if len(returns) >= 20 else 0.0

        avg_vol = hist["Volume"].iloc[:-1].mean() if "Volume" in hist.columns else np.nan
        last_vol = hist["Volume"].iloc[-1] if "Volume" in hist.columns else np.nan
        vol_ratio = float(last_vol / avg_vol) if avg_vol and np.isfinite(avg_vol) and avg_vol > 0 else 1.0

        prev_close = float(hist["Close"].iloc[-2])
        today_open = float(hist["Open"].iloc[-1])
        gap_pct = ((today_open - prev_close) / prev_close) * 100 if prev_close else 0.0

        rsi_val = _rsi(closes)

        ma5 = closes[-5:].mean()
        ma20_val = closes[-20:].mean()
        ma_gap = (current / ma20_val - 1.0) * 100 if ma20_val else 0.0
        ma_cross = (ma5 / ma20_val - 1.0) * 100 if ma20_val else 0.0

        x20 = np.arange(20, dtype="float64")
        try:
            slope20 = np.polyfit(x20, closes[-20:], 1)[0] / current * 100 if current else 0.0
        except Exception:
            slope20 = 0.0

        recovery = ((closes[-1] - closes[-4]) / closes[-4]) * 100 if len(closes) >= 5 else 0.0

        feats = np.array([
            m5, m10, m20, vol20, vol_ratio, gap_pct, rsi_val,
            ma_gap, ma_cross, slope20, recovery,
        ], dtype="float64")
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.reshape(1, -1)
    except Exception:
        return None


def _estimate_gain_heuristic(hist: pd.DataFrame) -> float:
    """
    Multi-signal bullish gain estimate — the ORIGINAL hand-tuned rule set.
    Kept as a fallback for when no trained ranker model is cached yet.
    Uses:
      1. Short-term trend (5d slope vs 20d slope) — are we recovering?
      2. Distance from 20d MA — how far below/above are we?
      3. RSI — is momentum turning bullish?
    Returns positive % when signals are bullish, negative when bearish.
    """
    if len(hist) < 20:
        return 0.0
    try:
        closes = hist["Close"].values

        # 1. Trend direction — 5d slope vs 20d slope
        x5  = np.arange(5)
        x20 = np.arange(20)
        slope5,  _ = np.polyfit(x5,  closes[-5:],  1)
        slope20, _ = np.polyfit(x20, closes[-20:], 1)
        current = closes[-1]

        # Normalize slopes as % per day
        trend5  = (slope5  / current) * 100
        trend20 = (slope20 / current) * 100

        # 2. Mean reversion signal — price vs 20d MA
        ma20     = closes[-20:].mean()
        ma_gap   = ((current - ma20) / ma20) * 100   # +ve = above MA (bullish)

        # 3. RSI signal — map 0-100 RSI to -3..+3 gain signal
        rsi       = _rsi(closes)
        rsi_signal = (rsi - 50) / 50 * 3   # RSI 70 → +1.2, RSI 30 → -1.2

        # 4. Recent recovery — did last 3 days gain vs 5 days ago?
        recovery = ((closes[-1] - closes[-4]) / closes[-4]) * 100 if len(closes) >= 5 else 0

        # Weighted composite
        gain = (
            trend5   * 2.0  +   # short-term trend dominates
            trend20  * 1.0  +   # medium trend confirms
            ma_gap   * 0.3  +   # mean reversion component
            rsi_signal * 1.0 +  # RSI momentum
            recovery * 0.5      # very recent price action
        )

        # Cap at ±10% to keep projections realistic
        return float(round(max(-10.0, min(10.0, gain)), 2))
    except Exception:
        return 0.0


def _estimate_gain(hist: pd.DataFrame) -> float:
    """
    Predicted next-day open->close % gain. Uses the trained global ranker
    if one is cached; otherwise falls back to the hand-tuned heuristic.
    This is the exact value backfill_walkforward_backtest.py compares
    against actual returns, so improving the ranker model's accuracy
    (via train_ranking_model) directly improves the backtest metrics.
    """
    bundle = _load_ranker_model()
    if bundle is not None:
        feats = _extract_ranking_features(hist)
        if feats is not None:
            try:
                pred = float(bundle["model"].predict(feats)[0])
                if np.isfinite(pred):
                    return float(round(max(-10.0, min(10.0, pred)), 2))
            except Exception as e:
                print(f"[Recommender] Ranker predict failed, falling back to heuristic: {e}")
    return _estimate_gain_heuristic(hist)


def _score_from_hist(symbol: str, company: str, hist: pd.DataFrame):
    try:
        if hist.empty or len(hist) < 10:
            return None

        current_price = float(hist["Close"].iloc[-1])
        open_price    = float(hist["Open"].iloc[-1])
        closes        = hist["Close"].values

        mom_score = _momentum_score(hist)
        vol_score = _volatility_score(hist)
        vum_score = _volume_score(hist)
        gap_score = _gap_score(hist)
        est_gain  = _estimate_gain(hist)
        rsi_val   = _rsi(closes)

        total_score = (
            mom_score * 0.40 +
            vol_score * 0.30 +
            vum_score * 0.20 +
            gap_score * 0.10
        )

        reasons = []
        if mom_score > 65:   reasons.append("strong upward momentum")
        if vol_score > 70:   reasons.append("low volatility")
        if vum_score > 70:   reasons.append("high buying volume")
        if gap_score > 65:   reasons.append("gap-up open")
        if rsi_val > 55:     reasons.append(f"RSI bullish ({rsi_val:.0f})")
        if rsi_val < 35:     reasons.append(f"oversold RSI ({rsi_val:.0f}) — recovery potential")
        if est_gain > 0.5:   reasons.append(f"trend projects +{est_gain:.1f}%")

        reason       = ("Based on " + ", ".join(reasons)) if reasons else "Neutral technical signals"
        target_price = round(current_price * (1 + est_gain / 100), 2)

        return {
            "symbol":         symbol,
            "company":        company,
            "score":          round(total_score, 2),
            "predicted_gain": est_gain,
            "current_price":  current_price,
            "open_price":     open_price,
            "target_price":   target_price,
            "reason":         reason,
            "momentum":       round(mom_score, 1),
            "volatility":     round(vol_score, 1),
            "volume":         round(vum_score, 1),
            "rsi":            round(rsi_val, 1),
        }
    except Exception as e:
        print(f"[Recommender] Score error for {symbol}: {e}")
        return None


# ── Trained ranking model ─────────────────────────────────────────────────────

def _build_ranker_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=42,
        )),
    ])


def train_ranking_model(lookback: str = "2y", min_hist_days: int = 25) -> dict | None:
    """
    Train the global ranker on pooled walk-forward samples across all
    Nifty 50 symbols. For each symbol and each historical day D, builds
    features from data strictly BEFORE D (no lookahead — same discipline
    as backfill_walkforward_backtest.py) and labels the row with D's
    actual open->close % change. Pooling across ~50 symbols x ~2 years of
    days gives a much larger, more diverse training set than any single
    stock's own history could — that's what lets the model learn general
    momentum/volatility/volume patterns instead of memorizing one stock's
    quirks.

    Saves the fitted model to disk (RANKER_MODEL_PATH) and returns a
    dict of holdout validation metrics so you can see quality before
    trusting it in production. Run this periodically (e.g. weekly cron),
    mirroring ml_model.py's MODEL_EXPIRY_DAYS pattern.
    """
    stocks = load_nifty50()
    symbols = [s for s, _ in stocks]
    if not symbols:
        print("[Recommender] No symbols loaded — cannot train ranker.")
        return None

    print(f"[Ranker] Downloading {lookback} of history for {len(symbols)} symbols...")
    raw = yf.download(symbols, period=lookback, group_by="ticker", progress=False, threads=True)

    frames = {}
    for sym in symbols:
        try:
            df = raw[sym].dropna(how="any").copy()
            df.columns = [c.capitalize() for c in df.columns]
            if "Close" in df.columns and len(df) > min_hist_days + 30:
                frames[sym] = df
        except Exception:
            continue

    print(f"[Ranker] Usable history for {len(frames)}/{len(symbols)} symbols.")
    if not frames:
        print("[Ranker] No usable data — check network/yfinance connectivity.")
        return None

    ref_sym = max(frames, key=lambda s: len(frames[s]))
    all_dates = frames[ref_sym].index

    X_rows, y_rows, date_rows = [], [], []

    for D in all_dates:
        for sym, df in frames.items():
            if D not in df.index:
                continue
            hist_upto = df.loc[df.index < D]
            if len(hist_upto) < min_hist_days:
                continue
            feats = _extract_ranking_features(hist_upto)
            if feats is None:
                continue
            try:
                o, c = float(df.loc[D, "Open"]), float(df.loc[D, "Close"])
                if not o:
                    continue
                label = (c - o) / o * 100
            except Exception:
                continue
            X_rows.append(feats.flatten())
            y_rows.append(label)
            date_rows.append(D)

    if len(X_rows) < 500:
        print(f"[Ranker] Only {len(X_rows)} training rows — need more history/symbols to train reliably.")
        return None

    X = np.asarray(X_rows, dtype="float64")
    y = np.asarray(y_rows, dtype="float64")
    dates = np.asarray(date_rows)

    # Time-based split (not shuffled): train on the earlier ~80% of dates,
    # validate on the most recent ~20%. Shuffling here would leak future
    # information into training via same-day cross-sectional correlation.
    order = np.argsort(dates)
    X, y, dates = X[order], y[order], dates[order]
    split_date = np.quantile(np.unique(dates).astype("datetime64[ns]").view("int64"), 0.8)
    split_mask = dates.astype("datetime64[ns]").view("int64") <= split_date

    X_train, y_train = X[split_mask], y[split_mask]
    X_holdout, y_holdout = X[~split_mask], y[~split_mask]

    print(f"[Ranker] Training on {len(X_train)} rows, validating on {len(X_holdout)} holdout rows.")

    model = _build_ranker_pipeline()
    model.fit(X_train, y_train)

    metrics = {}
    if len(X_holdout) >= 20:
        pred_holdout = model.predict(X_holdout)
        metrics["holdout_mae"] = float(np.mean(np.abs(pred_holdout - y_holdout)))
        metrics["holdout_directional_accuracy"] = float(
            (np.sign(pred_holdout) == np.sign(y_holdout)).mean() * 100
        )
        metrics["holdout_corr"] = float(np.corrcoef(pred_holdout, y_holdout)[0, 1])
        metrics["n_holdout"] = len(X_holdout)
        print(f"[Ranker] Holdout MAE: {metrics['holdout_mae']:.3f}pp  "
              f"Directional accuracy: {metrics['holdout_directional_accuracy']:.1f}%  "
              f"Corr: {metrics['holdout_corr']:.3f}")
    else:
        print("[Ranker] Not enough holdout rows for validation metrics.")

    # Refit on the FULL pooled dataset for the deployed model, now that
    # holdout metrics have already been honestly measured.
    final_model = _build_ranker_pipeline()
    final_model.fit(X, y)

    bundle = {
        "model": final_model,
        "trained_at": datetime.now().isoformat(),
        "n_train_rows": len(X),
        "n_symbols": len(frames),
        "lookback": lookback,
        "metrics": metrics,
    }

    joblib.dump(bundle, RANKER_MODEL_PATH)
    with open(RANKER_META_PATH, "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat())

    _RANKER_CACHE["bundle"] = bundle
    print(f"[Ranker] Saved trained global ranker ({len(X)} rows, {len(frames)} symbols) to {RANKER_MODEL_PATH}")
    return metrics


# ── Main fast scoring ─────────────────────────────────────────────────────────

def generate_recommendations() -> list:
    print(f"[Recommender] Generating recommendations for {date.today()}...")
    stocks = load_nifty50()
    if not stocks:
        print("[Recommender] No stocks loaded.")
        return []

    symbols     = [s for s, _ in stocks]
    company_map = {s: c for s, c in stocks}

    print(f"[Recommender] Batch fetching {len(symbols)} symbols...")
    hist_map = _batch_fetch(symbols, period="60d")   # 60d for better RSI + MA signals
    print(f"[Recommender] Got data for {len(hist_map)}/{len(symbols)} symbols.")

    if not hist_map:
        print("[Recommender] No market data returned. Check yfinance connectivity.")
        return []

    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {
            ex.submit(_score_from_hist, sym, company_map[sym], hist_map[sym]): sym
            for sym in hist_map
        }
        for fut in as_completed(futures):
            scored = fut.result()
            if scored:
                results.append(scored)

    if not results:
        print("[Recommender] Scoring returned no results.")
        return []

    # Split into bullish (positive gain) and bearish candidates
    bullish = [r for r in results if r["predicted_gain"] > 0]
    bearish = [r for r in results if r["predicted_gain"] <= 0]

    print(f"[Recommender] {len(bullish)} bullish / {len(bearish)} bearish candidates")

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
        print("[Recommender] All stocks bearish today — picking least-bad options.")
        for r in results:
            r["_rank_score"] = r["score"] - abs(r["predicted_gain"]) * 5
        results.sort(key=lambda x: x["_rank_score"], reverse=True)
        pool = results

    top5 = pool[:5]
    for i, r in enumerate(top5):
        r["rank"] = i + 1
        r.pop("_rank_score", None)

    print(f"[Recommender] Done. Top 5: {[(r['symbol'], r['predicted_gain']) for r in top5]}")
    return top5


def score_stock(symbol: str, company: str):
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period="30d")
        if hist.empty:
            return None
        hist.columns = [c.capitalize() for c in hist.columns]
        return _score_from_hist(symbol, company, hist)
    except Exception as e:
        print(f"[Recommender] score_stock error for {symbol}: {e}")
        return None


# ── DB persistence ────────────────────────────────────────────────────────────

def save_recommendations(recommendations: list):
    import db
    conn = cur = None
    try:
        conn  = db.get_conn()
        cur   = conn.cursor()
        today = date.today()
        cur.execute("DELETE FROM ai_recommendations WHERE date=%s", (today,))
        for r in recommendations:
            cur.execute("""
                INSERT INTO ai_recommendations
                  (date, stock_symbol, company_name, score, predicted_gain,
                   current_price, target_price, reason, rank)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                today, r["symbol"], r["company"], r["score"],
                r["predicted_gain"], r["current_price"],
                r["target_price"], r["reason"], r["rank"]
            ))
        conn.commit()
        print(f"[Recommender] Saved {len(recommendations)} recommendations to DB.")
        return True
    except Exception as e:
        if conn: conn.rollback()
        print(f"[Recommender] DB save error: {e}")
        traceback.print_exc()
        return False
    finally:
        if cur:  cur.close()
        if conn: db.release_conn(conn)


def get_todays_recommendations() -> list:
    """
    Fetch today's cached recommendations from DB.
    Uses psycopg2 RealDictCursor and CURRENT_DATE (PostgreSQL).
    """
    import db
    conn = cur = None
    try:
        conn = db.get_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT * FROM ai_recommendations
            WHERE date = CURRENT_DATE
            ORDER BY rank ASC
        """)
        rows = cur.fetchall()
        result = []
        for r in rows:
            row = dict(r)
            if row.get("date"):       row["date"]       = str(row["date"])
            if row.get("created_at"): row["created_at"] = str(row["created_at"])
            for field in ("score", "predicted_gain", "current_price", "target_price"):
                if row.get(field) is not None:
                    row[field] = float(row[field])
            result.append(row)
        print(f"[Recommender] Fetched {len(result)} recommendations from DB for today.")
        return result
    except Exception as e:
        print(f"[Recommender] DB fetch error: {e}")
        traceback.print_exc()
        return []
    finally:
        if cur:  cur.close()
        if conn: db.release_conn(conn)


def track_daily_prices():
    """Batch-fetch open/close prices for all Nifty 50."""
    import db
    stocks  = load_nifty50()
    symbols = [s for s, _ in stocks]
    today   = date.today()

    hist_map = _batch_fetch(symbols, period="2d")

    for symbol, _ in stocks:
        hist = hist_map.get(symbol)
        if hist is None or hist.empty:
            continue
        conn = cur = None
        try:
            row  = hist.iloc[-1]
            conn = db.get_conn()
            cur  = conn.cursor()
            cur.execute("""
                INSERT INTO daily_prices
                  (date, stock_symbol, open_price, close_price, high_price, low_price, volume, pct_change)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (date, stock_symbol) DO UPDATE SET
                  close_price = EXCLUDED.close_price,
                  high_price  = EXCLUDED.high_price,
                  low_price   = EXCLUDED.low_price,
                  volume      = EXCLUDED.volume,
                  pct_change  = EXCLUDED.pct_change
            """, (
                today, symbol,
                float(row["Open"]),  float(row["Close"]),
                float(row["High"]),  float(row["Low"]),
                int(row["Volume"]),
                round(((float(row["Close"]) - float(row["Open"])) / float(row["Open"])) * 100, 2)
            ))
            conn.commit()
        except Exception as e:
            if conn: conn.rollback()
            print(f"[Prices] Error tracking {symbol}: {e}")
        finally:
            if cur:  cur.close()
            if conn: db.release_conn(conn)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        lb = sys.argv[2] if len(sys.argv) > 2 else "2y"
        train_ranking_model(lookback=lb)
    else:
        print("Usage: python recommender.py --train [lookback, e.g. 2y]")