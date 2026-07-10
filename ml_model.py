"""
ml_model.py - TensorFlow-free stock forecast using yfinance + scikit-learn

Compatible with Python 3.13.

Model:
  - Uses sklearn HistGradientBoostingRegressor instead of TensorFlow/Keras LSTM.
  - Forecasts recursively using lag features and rolling statistics.
  - Works for Indian stocks such as RELIANCE.NS and US stocks.
  - Disk-cached models - retrain once a week.
  - In-memory caching for models, raw data, forecasts, and plots.
  - Background training for fast responses when a stale model exists.
  - Prices shown in local currency labels: INR for .NS/.BO, USD otherwise.
"""

import os
import io
import base64
import warnings
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "model_cache")
MODEL_EXPIRY_DAYS = int(os.getenv("MODEL_EXPIRY_DAYS", "7"))

LOOKBACK = 60
MIN_ROWS = LOOKBACK + 120

DATA_CACHE_TTL = timedelta(minutes=10)
FORECAST_CACHE_TTL = timedelta(minutes=10)
PLOT_CACHE_TTL = timedelta(minutes=10)

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

_training_lock = threading.Lock()
_currently_training: set[str] = set()

MODEL_MEMORY_CACHE = {}  # symbol -> bundle dict
DATA_CACHE = {}          # symbol -> (df, timestamp)
FORECAST_CACHE = {}      # (symbol, forecast_type) -> (df, timestamp)
PLOT_CACHE = {}          # (symbol, forecast_type) -> (b64_str, timestamp)

_cache_lock = threading.Lock()


# -----------------------------------------------------------------------------
# Disk cache helpers
# -----------------------------------------------------------------------------

def _safe_symbol(symbol: str) -> str:
    return symbol.replace(".", "_").replace("/", "_").upper()


def _paths(symbol: str) -> dict[str, str]:
    safe = _safe_symbol(symbol)
    base = os.path.join(MODEL_CACHE_DIR, safe)

    return {
        "bundle": base + "_sklearn_model.joblib",
        "meta": base + "_meta.txt",
    }


def _is_fresh(symbol: str) -> bool:
    p = _paths(symbol)

    if not os.path.exists(p["bundle"]) or not os.path.exists(p["meta"]):
        return False

    try:
        with open(p["meta"], "r", encoding="utf-8") as f:
            saved_at = datetime.fromisoformat(f.read().strip())

        age_days = (datetime.now() - saved_at).days
        return age_days < MODEL_EXPIRY_DAYS
    except Exception:
        return False


def _save(symbol: str, bundle: dict) -> None:
    p = _paths(symbol)

    joblib.dump(bundle, p["bundle"])

    with open(p["meta"], "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat())

    with _cache_lock:
        MODEL_MEMORY_CACHE[symbol] = bundle

    print(f"[Cache] Saved sklearn model for {symbol}")


def _load(symbol: str) -> dict:
    with _cache_lock:
        cached = MODEL_MEMORY_CACHE.get(symbol)

    if cached is not None:
        return cached

    p = _paths(symbol)
    bundle = joblib.load(p["bundle"])

    with _cache_lock:
        MODEL_MEMORY_CACHE[symbol] = bundle

    print(f"[Cache] Loaded sklearn model for {symbol} from disk")
    return bundle


# -----------------------------------------------------------------------------
# Data fetching
# -----------------------------------------------------------------------------

def fetch_stock_data(symbol: str) -> pd.DataFrame | None:
    """
    Fetch 5 years of daily close prices from yfinance.
    """
    symbol = symbol.strip().upper()

    with _cache_lock:
        cached = DATA_CACHE.get(symbol)

    if cached is not None:
        df, ts = cached
        if datetime.now() - ts < DATA_CACHE_TTL:
            return df.copy()

    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5y", auto_adjust=False)

            if df.empty:
                print(f"[yfinance] Empty response for {symbol}, attempt {attempt + 1}")

                if attempt < 2:
                    time.sleep(2)

                continue

            df = df.reset_index()

            if "Date" not in df.columns or "Close" not in df.columns:
                print(f"[yfinance] Missing Date/Close columns for {symbol}")
                return None

            df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
            df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
            df["y"] = pd.to_numeric(df["y"], errors="coerce")

            df = (
                df[["ds", "y"]]
                .dropna()
                .sort_values("ds")
                .reset_index(drop=True)
            )

            if len(df) < MIN_ROWS:
                print(f"[yfinance] Not enough rows for {symbol}: {len(df)}")
                return None

            with _cache_lock:
                DATA_CACHE[symbol] = (df.copy(), datetime.now())

            print(f"[yfinance] {symbol}: {len(df)} rows fetched")
            return df

        except Exception as e:
            print(f"[yfinance] Error {symbol} attempt {attempt + 1}: {e}")

            if attempt < 2:
                time.sleep(2)

    print(f"[yfinance] All attempts failed for {symbol}")
    return None


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def _make_features_from_window(window: np.ndarray) -> np.ndarray:
    """
    Build one feature row from the latest price window.

    IMPORTANT DESIGN NOTE: every feature here is scale-free (a ratio, a
    percentage, or a return) — none are raw price levels. This matters a
    lot for a stock that has drifted a long way over the 5y training
    window (e.g. up 3-4x): a feature like "price 60 days ago = 412" seen
    only during the low-price part of history becomes out-of-distribution
    once the stock triples, and tree models cannot extrapolate outside
    the ranges they were trained on. Ratios like "price / 60d MA" stay in
    a roughly stable range across the whole history, so the model keeps
    seeing familiar input even as the absolute price moves far beyond
    anything in training.

    Features include:
      - short-horizon returns
      - rolling volatility
      - distance from moving averages (ratios, not levels)
      - normalized trend slope (% per day, not raw slope)
      - normalized recent swings (% change, not raw price delta)
    """
    w = np.asarray(window, dtype="float64")

    if len(w) < LOOKBACK:
        raise ValueError(f"Window must have at least {LOOKBACK} values")

    recent = w[-LOOKBACK:]
    current = recent[-1]

    returns = pd.Series(recent).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ret_features = [
        returns.iloc[-1],
        returns.iloc[-2],
        returns.iloc[-3],
        returns.tail(5).mean(),
        returns.tail(10).mean(),
        returns.tail(20).mean(),
        returns.tail(20).std(),
        returns.tail(60).std(),
    ]

    ma5 = recent[-5:].mean()
    ma10 = recent[-10:].mean()
    ma20 = recent[-20:].mean()
    ma60 = recent[-60:].mean()

    ma_features = [
        current / ma5 - 1.0 if ma5 else 0.0,
        current / ma10 - 1.0 if ma10 else 0.0,
        current / ma20 - 1.0 if ma20 else 0.0,
        current / ma60 - 1.0 if ma60 else 0.0,
        ma5 / ma20 - 1.0 if ma20 else 0.0,   # short vs medium trend cross
        ma10 / ma60 - 1.0 if ma60 else 0.0,  # medium vs long trend cross
    ]

    x = np.arange(20, dtype="float64")
    y = recent[-20:]

    try:
        slope20 = np.polyfit(x, y, 1)[0] / current if current else 0.0
    except Exception:
        slope20 = 0.0

    x60 = np.arange(60, dtype="float64")
    y60 = recent[-60:]
    try:
        slope60 = np.polyfit(x60, y60, 1)[0] / current if current else 0.0
    except Exception:
        slope60 = 0.0

    trend_features = [
        slope20,
        slope60,
        (recent[-1] / recent[-5] - 1.0) if recent[-5] else 0.0,
        (recent[-1] / recent[-20] - 1.0) if recent[-20] else 0.0,
        (recent[-1] / recent[-60] - 1.0) if recent[-60] else 0.0,
    ]

    features = np.array(ret_features + ma_features + trend_features, dtype="float64")
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features.reshape(1, -1)


def _build_training_matrix(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a price series into supervised learning rows.

    X at time t uses prices strictly before t (t-LOOKBACK .. t-1).
    y at time t is the log-return from price[t-1] to price[t] — NOT the
    raw price. Predicting returns instead of levels is what fixes the
    extrapolation problem described above: returns stay in a roughly
    constant range over years of history, so the model is always
    predicting inside the distribution it was trained on, no matter how
    far the price itself has moved.
    """
    prices = np.asarray(prices, dtype="float64").flatten()

    X_rows = []
    y_rows = []

    for i in range(LOOKBACK, len(prices)):
        window = prices[i - LOOKBACK:i]
        prev_price = prices[i - 1]
        if prev_price <= 0 or prices[i] <= 0:
            continue
        log_ret = np.log(prices[i] / prev_price)
        X_rows.append(_make_features_from_window(window).flatten())
        y_rows.append(log_ret)

    X = np.asarray(X_rows, dtype="float64")
    y = np.asarray(y_rows, dtype="float64")

    return X, y


def _build_model() -> Pipeline:
    """
    Fast sklearn model suitable for tabular lag/rolling features.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.04,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    random_state=42,
                ),
            ),
        ]
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def _train_and_save(symbol: str, df: pd.DataFrame) -> None:
    try:
        prices = df["y"].values.astype("float64")

        X, y = _build_training_matrix(prices)

        if len(X) < 100:
            raise ValueError(f"Not enough training samples for {symbol}: {len(X)}")

        # Time-based holdout (last 15%, no shuffling) so residual_std — and
        # therefore the confidence interval width — reflects genuine
        # out-of-sample error rather than in-sample fit, which is always
        # optimistic and would make the CI misleadingly narrow.
        split = int(len(X) * 0.85)
        X_train, X_holdout = X[:split], X[split:]
        y_train, y_holdout = y[:split], y[split:]

        model = _build_model()
        model.fit(X_train, y_train)

        if len(X_holdout) >= 10:
            holdout_pred = model.predict(X_holdout)
            residual_std = float(np.std(y_holdout - holdout_pred))
        else:
            # Not enough data for a holdout split — fall back to in-sample,
            # but this case should be rare given MIN_ROWS.
            train_pred = model.predict(X_train)
            residual_std = float(np.std(y_train - train_pred))

        # Refit on the full dataset for the final deployed model, now that
        # residual_std has already been estimated honestly on unseen data.
        model = _build_model()
        model.fit(X, y)

        last_prices = prices[-LOOKBACK:].copy()

        bundle = {
            "model": model,
            "residual_std": residual_std,  # std of LOG-RETURN residuals
            "last_prices": last_prices,
            "trained_at": datetime.now().isoformat(),
            "lookback": LOOKBACK,
            "target": "log_return",
            # Feature-schema fingerprint — lets get_or_train_model() detect
            # a stale cached model whose scaler/column count no longer
            # matches the current _make_features_from_window(), instead of
            # crashing inside StandardScaler.transform() at predict time.
            "n_features": X.shape[1],
        }

        _save(symbol, bundle)

    except Exception as e:
        print(f"[Train] Error training {symbol}: {e}")

    finally:
        with _training_lock:
            _currently_training.discard(symbol)


def _spawn_training(symbol: str, df: pd.DataFrame) -> None:
    with _training_lock:
        if symbol in _currently_training:
            return

        _currently_training.add(symbol)

    t = threading.Thread(target=_train_and_save, args=(symbol, df.copy()), daemon=True)
    t.start()

    print(f"[BG] Training started for {symbol}")


# -----------------------------------------------------------------------------
# Get or train
# -----------------------------------------------------------------------------

def _bundle_is_compatible(bundle: dict) -> bool:
    """
    Returns False if this bundle was trained under a different version of
    _make_features_from_window() than the one currently running (e.g. the
    feature engineering was changed since this model was cached). Older
    bundles saved before this check existed won't have "n_features" and
    are treated as incompatible, forcing a one-time retrain.
    """
    try:
        expected = bundle.get("n_features")
        if expected is None:
            return False
        last_prices = np.asarray(bundle.get("last_prices"), dtype="float64")
        if len(last_prices) < LOOKBACK:
            return False
        actual = _make_features_from_window(last_prices[-LOOKBACK:]).shape[1]
        return expected == actual
    except Exception:
        return False


def get_or_train_model(symbol: str, df: pd.DataFrame) -> dict:
    """
    If a fresh, schema-compatible model exists, load it.

    If a stale model exists, serve it immediately (if schema-compatible)
    and retrain in background — otherwise retrain synchronously, since a
    schema-incompatible bundle would crash at predict() time rather than
    just being a bit out of date.

    If no model exists, train synchronously once.
    """
    symbol = symbol.strip().upper()
    p = _paths(symbol)

    if _is_fresh(symbol):
        bundle = _load(symbol)
        if _bundle_is_compatible(bundle):
            return bundle
        print(f"[Train] {symbol}: cached model schema is stale "
              f"(feature count changed) — retraining synchronously.")
        _train_and_save(symbol, df)
        if os.path.exists(p["bundle"]):
            return _load(symbol)
        raise RuntimeError(f"Could not retrain stale model for {symbol}")

    if os.path.exists(p["bundle"]):
        bundle = _load(symbol)
        if _bundle_is_compatible(bundle):
            _spawn_training(symbol, df)
            return bundle
        print(f"[Train] {symbol}: cached model schema is stale "
              f"(feature count changed) — retraining synchronously.")
        _train_and_save(symbol, df)
        if os.path.exists(p["bundle"]):
            return _load(symbol)
        raise RuntimeError(f"Could not retrain stale model for {symbol}")

    print(f"[Train] First-time training for {symbol} - cold start")
    _train_and_save(symbol, df)

    if os.path.exists(p["bundle"]):
        return _load(symbol)

    raise RuntimeError(f"Could not train model for {symbol}")


# -----------------------------------------------------------------------------
# Forecast helpers
# -----------------------------------------------------------------------------

def _forecast_days(bundle: dict, last_window: np.ndarray, n_days: int) -> np.ndarray:
    """
    Recursively forecast n_days ahead. The model predicts one log-return
    per step; each predicted return is compounded onto the last known
    price to get the next price, and that reconstructed price extends the
    window so the next step's features can be computed. Because the model
    only ever outputs a return (typically a few percent), it never has to
    extrapolate outside its training range no matter how many days out we
    forecast — that's what the earlier price-level version couldn't do.
    """
    model = bundle["model"]

    window = np.asarray(last_window, dtype="float64").flatten().copy()

    if len(window) < LOOKBACK:
        raise ValueError(f"last_window must contain at least {LOOKBACK} prices")

    preds = []

    for _ in range(n_days):
        X_next = _make_features_from_window(window[-LOOKBACK:])
        pred_ret = float(model.predict(X_next)[0])

        if not np.isfinite(pred_ret):
            pred_ret = 0.0

        # Clip per-step log-return to a sane daily range (~±15%) — guards
        # against a runaway compounding forecast if the model ever
        # extrapolates into an unstable region.
        pred_ret = max(-0.15, min(0.15, pred_ret))

        next_price = float(window[-1]) * np.exp(pred_ret)
        if not np.isfinite(next_price) or next_price <= 0:
            next_price = float(window[-1])

        preds.append(next_price)
        window = np.append(window, next_price)

    return np.asarray(preds, dtype="float64")


def _ci(preds: np.ndarray, residual_std: float | None = None, z: float = 1.65) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a confidence band in price space from the model's log-return
    residual_std. Log-return errors accumulate roughly like a random walk,
    so the cumulative log-return std at horizon h scales with sqrt(h);
    converting that back through exp() gives a price-space band that
    widens the further out the forecast goes, instead of the fixed-scale
    band the old level-based version produced.
    """
    preds = np.asarray(preds, dtype="float64")

    if residual_std is None or not np.isfinite(residual_std) or residual_std <= 0:
        residual_std = 0.02  # reasonable default daily log-return std if unavailable

    horizon = np.arange(1, len(preds) + 1)
    cum_log_std = residual_std * np.sqrt(horizon)

    lower = preds * np.exp(-z * cum_log_std)
    upper = preds * np.exp(z * cum_log_std)
    lower = np.maximum(lower, 0)

    return lower, upper


def _business_dates(last_date: pd.Timestamp, n_days: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=last_date + timedelta(days=1), periods=n_days)


# -----------------------------------------------------------------------------
# Currency
# -----------------------------------------------------------------------------

def _currency(symbol: str) -> str:
    symbol = symbol.upper()

    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return "₹"

    return "$"


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_aggregated_forecast(symbol: str, forecast_type: str = "6m") -> pd.DataFrame | None:
    symbol = symbol.strip().upper()
    forecast_type = forecast_type.strip().lower()

    cache_key = (symbol, forecast_type)

    with _cache_lock:
        cached = FORECAST_CACHE.get(cache_key)

    if cached is not None:
        agg, ts = cached

        if datetime.now() - ts < FORECAST_CACHE_TTL:
            return agg.copy()

    df = fetch_stock_data(symbol)

    if df is None or df.empty:
        return None

    try:
        bundle = get_or_train_model(symbol, df)
    except Exception as e:
        print(f"[Forecast] Model unavailable for {symbol}: {e}")
        return None

    last_date = df["ds"].max()
    last_window = df["y"].values[-LOOKBACK:]

    if forecast_type == "6m":
        n_days = 180
        preds = _forecast_days(bundle, last_window, n_days)
        dates = _business_dates(last_date, n_days)

        lo, hi = _ci(preds, bundle.get("residual_std"))

        tmp = pd.DataFrame(
            {
                "ds": dates,
                "yhat": preds,
                "yhat_lower": lo,
                "yhat_upper": hi,
            }
        )

        tmp["period"] = tmp["ds"].dt.to_period("M")

        agg = (
            tmp.groupby("period")[["yhat", "yhat_lower", "yhat_upper"]]
            .mean()
            .reset_index()
        )

        agg["ds"] = agg["period"].dt.to_timestamp()
        agg = agg[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(6)

    elif forecast_type == "5y":
        n_days = 252 * 5
        preds = _forecast_days(bundle, last_window, n_days)
        dates = _business_dates(last_date, n_days)

        lo, hi = _ci(preds, bundle.get("residual_std"))

        tmp = pd.DataFrame(
            {
                "ds": dates,
                "yhat": preds,
                "yhat_lower": lo,
                "yhat_upper": hi,
            }
        )

        tmp["year"] = tmp["ds"].dt.year

        agg = (
            tmp.groupby("year")[["yhat", "yhat_lower", "yhat_upper"]]
            .mean()
            .reset_index()
        )

        agg["ds"] = pd.to_datetime(agg["year"].astype(str) + "-01-01")
        agg = agg[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(5)

    else:
        print(f"[Forecast] Unsupported forecast_type: {forecast_type}")
        return None

    with _cache_lock:
        FORECAST_CACHE[cache_key] = (agg.copy(), datetime.now())

    return agg


def generate_stock_plot(symbol: str, forecast_type: str = "6m") -> str | None:
    symbol = symbol.strip().upper()
    forecast_type = forecast_type.strip().lower()

    plot_key = (symbol, forecast_type)

    with _cache_lock:
        cached = PLOT_CACHE.get(plot_key)

    if cached is not None:
        b64, ts = cached

        if datetime.now() - ts < PLOT_CACHE_TTL:
            return b64

    fdf = get_aggregated_forecast(symbol, forecast_type)
    b64 = generate_stock_plot_from_dataframe(symbol, forecast_type, fdf)

    if b64 is not None:
        with _cache_lock:
            PLOT_CACHE[plot_key] = (b64, datetime.now())

    return b64


def generate_stock_plot_from_dataframe(
    symbol: str,
    forecast_type: str,
    fdf: pd.DataFrame | None,
) -> str | None:
    if fdf is None or fdf.empty:
        return None

    symbol = symbol.strip().upper()
    forecast_type = forecast_type.strip().lower()

    cur = _currency(symbol)
    display = symbol.replace(".NS", "").replace(".BO", "")

    label = "Monthly Avg" if forecast_type == "6m" else "Yearly Avg"
    period_label = "6-Month" if forecast_type == "6m" else "5-Year"

    title = f"{display} - {period_label} ML Forecast ({label})"

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=110)

    ax.plot(
        fdf["ds"],
        fdf["yhat"],
        marker="o",
        color="#6c5ce7",
        linewidth=2,
        label="Predicted",
    )

    ax.fill_between(
        fdf["ds"],
        fdf["yhat_lower"],
        fdf["yhat_upper"],
        alpha=0.25,
        color="#a29bfe",
        label="90% CI",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Price ({cur})")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.xticks(rotation=35)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    buf.seek(0)

    return base64.b64encode(buf.read()).decode()


def get_forecast_and_plot(symbol: str, forecast_type: str = "6m"):
    symbol = symbol.strip().upper()
    forecast_type = forecast_type.strip().lower()

    fdf = get_aggregated_forecast(symbol, forecast_type)

    if fdf is None:
        return None, None

    plot_key = (symbol, forecast_type)

    with _cache_lock:
        cached = PLOT_CACHE.get(plot_key)

    if cached is not None:
        b64, ts = cached

        if datetime.now() - ts < PLOT_CACHE_TTL:
            return fdf, b64

    b64 = generate_stock_plot_from_dataframe(symbol, forecast_type, fdf)

    if b64 is not None:
        with _cache_lock:
            PLOT_CACHE[plot_key] = (b64, datetime.now())

    return fdf, b64