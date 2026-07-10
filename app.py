"""
app.py — StockWise Flask application
Indian Nifty 50 stocks via yfinance (no API limits)
"""
import os, csv
import pandas as pd
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv
# Always load .env from the same folder as app.py — works regardless of cwd
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

from msg      import fetch_current_price, send_alert_sms, send_stock_news_alert
from ml_model import generate_stock_plot, get_aggregated_forecast
import db
import scheduler
from ai_routes import ai_bp  # LangChain + Groq RAG/GenAI routes

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change_me_in_production!")
app.register_blueprint(ai_bp)

# ── Load Nifty 50 companies ─────────────────────────────────────────────────

def load_companies():
    out  = []
    path = os.path.join(os.path.dirname(__file__), "companies_india.csv")
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "Symbol" in row and "Company" in row:
                    out.append((row["Symbol"], row["Company"]))
    except Exception as e:
        print(f"[App] companies_india.csv error: {e}")
    return out

COMPANIES   = load_companies()
COMPANY_MAP = {sym: name for sym, name in COMPANIES}

scheduler.start()


# ── Auth helpers ─────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.is_json or request.method == "POST":
                return jsonify({"error": "Login required.", "redirect": "/login"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

def current_user_id():
    return session.get("user_id")


# ══════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════

@app.route("/login")
def login_page():
    if "user_id" in session:
        return redirect(url_for("index"))
    return render_template("auth.html", page="login")

@app.route("/register")
def register_page():
    if "user_id" in session:
        return redirect(url_for("index"))
    return render_template("auth.html", page="register")

@app.route("/api/register", methods=["POST"])
def api_register():
    data     = request.get_json(silent=True) or request.form
    username = (data.get("username") or "").strip()
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    phone    = (data.get("phone") or "").strip() or None

    if not username or not email or not password:
        return jsonify({"error": "Username, email and password are required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    ok, result = db.register_user(username, email, password, phone)
    if not ok:
        return jsonify({"error": result}), 400

    session["user_id"]  = result["id"]
    session["username"] = result["username"]
    session["email"]    = result["email"]
    return jsonify({"message": "Account created!", "username": result["username"]}), 201

@app.route("/api/login", methods=["POST"])
def api_login():
    data     = request.get_json(silent=True) or request.form
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    ok, result = db.login_user(email, password)
    if not ok:
        return jsonify({"error": result}), 401

    session["user_id"]  = result["id"]
    session["username"] = result["username"]
    session["email"]    = result["email"]
    return jsonify({"message": f"Welcome back, {result['username']}!", "username": result["username"]})

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"message": "Logged out."})

@app.route("/api/me")
def api_me():
    if "user_id" not in session:
        return jsonify({"logged_in": False})
    return jsonify({
        "logged_in": True,
        "user_id":   session["user_id"],
        "username":  session.get("username"),
        "email":     session.get("email"),
    })


# ══════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def index():
    return render_template("index.html",
                           companies=COMPANIES,
                           username=session.get("username"))


# ══════════════════════════════════════════════════════════════════
# STOCK INFO
# ══════════════════════════════════════════════════════════════════

@app.route("/get_current_stock_info")
@login_required
def get_current_stock_info():
    symbol = request.args.get("symbol", "").upper()
    if not symbol:
        return jsonify({"error": "Symbol required."}), 400
    try:
        price, _ = fetch_current_price(symbol)
        if price is None:
            return jsonify({"error": "Could not fetch price."}), 404
        currency = "₹" if symbol.endswith(".NS") or symbol.endswith(".BO") else "$"
        return jsonify({
            "symbol":        symbol,
            "company_name":  COMPANY_MAP.get(symbol, symbol),
            "current_price": f"{price:.2f}",
            "currency":      currency,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
# FORECAST
# ══════════════════════════════════════════════════════════════════

@app.route("/get_forecast")
@login_required
def get_forecast():
    symbol        = request.args.get("symbol", "").upper()
    forecast_type = request.args.get("forecast_type", "6m")

    if not symbol:
        return jsonify({"error": "Symbol required."}), 400
    if forecast_type not in ("6m", "5y"):
        return jsonify({"error": "forecast_type must be '6m' or '5y'."}), 400

    try:
        fdf = get_aggregated_forecast(symbol, forecast_type)
        if fdf is None or fdf.empty:
            return jsonify({"error": "Could not generate forecast."}), 500
        if not pd.api.types.is_datetime64_any_dtype(fdf["ds"]):
            fdf["ds"] = pd.to_datetime(fdf["ds"])
        plot     = generate_stock_plot(symbol, forecast_type)
        currency = "₹" if symbol.endswith(".NS") or symbol.endswith(".BO") else "$"
        return jsonify({
            "dates":      fdf["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "yhat":       fdf["yhat"].round(2).tolist(),
            "yhat_lower": fdf["yhat_lower"].round(2).tolist(),
            "yhat_upper": fdf["yhat_upper"].round(2).tolist(),
            "plot_img":   plot,
            "currency":   currency,
        })
    except Exception as e:
        print(f"[Forecast] {symbol}: {e}")
        return jsonify({"error": "Forecast failed. Please try again."}), 500


# ══════════════════════════════════════════════════════════════════
# ALERTS
# ══════════════════════════════════════════════════════════════════

@app.route("/set_alert", methods=["POST"])
@login_required
def set_alert():
    data   = request.get_json(silent=True) or request.form
    symbol = (data.get("stock") or "").upper()
    phone  = data.get("phone", "")
    uid    = current_user_id()

    if not symbol or not phone:
        return jsonify({"error": "Stock and phone are required."}), 400

    company = COMPANY_MAP.get(symbol, symbol)
    try:
        saved = db.save_alert(symbol, phone, uid)
        if not saved:
            return jsonify({"error": "Could not save alert."}), 500
        send_alert_sms(phone, f"StockWise: Alert set for {symbol} ({company}).")
        send_stock_news_alert(symbol, company, phone, threshold_percent=1)
        return jsonify({"message": "Alert set successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO — BUY
# ══════════════════════════════════════════════════════════════════

@app.route("/portfolio/buy", methods=["POST"])
@login_required
def portfolio_buy():
    data        = request.get_json(silent=True) or request.form
    symbol      = (data.get("symbol") or "").upper()
    quantity    = float(data.get("quantity", 1))
    stop_loss   = float(data["stop_loss"])   if data.get("stop_loss")   else None
    take_profit = float(data["take_profit"]) if data.get("take_profit") else None
    phone       = data.get("phone")
    uid         = current_user_id()

    if not symbol:
        return jsonify({"error": "Symbol required."}), 400
    if quantity <= 0:
        return jsonify({"error": "Quantity must be positive."}), 400

    price, _ = fetch_current_price(symbol)
    if price is None:
        return jsonify({"error": f"Could not fetch price for {symbol}."}), 400

    company  = COMPANY_MAP.get(symbol, symbol)
    currency = "₹" if symbol.endswith(".NS") or symbol.endswith(".BO") else "$"
    pid      = db.buy_stock(symbol, company, quantity, price,
                            stop_loss, take_profit, phone, uid)
    if pid is None:
        return jsonify({"error": "Database error saving position."}), 500

    total = round(quantity * price, 2)
    if phone:
        send_alert_sms(phone,
            f"StockWise Buy: {symbol} x{quantity} @ {currency}{price:.2f} | Total: {currency}{total}")

    return jsonify({
        "message":      f"Bought {quantity} x {symbol} @ {currency}{price:.2f}",
        "portfolio_id": pid,
        "symbol":       symbol,
        "quantity":     quantity,
        "buy_price":    price,
        "total":        total,
        "stop_loss":    stop_loss,
        "take_profit":  take_profit,
        "currency":     currency,
    })


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO — SELL
# ══════════════════════════════════════════════════════════════════

@app.route("/portfolio/sell", methods=["POST"])
@login_required
def portfolio_sell():
    data = request.get_json(silent=True) or request.form
    pid  = data.get("portfolio_id")
    uid  = current_user_id()

    if not pid:
        return jsonify({"error": "portfolio_id required."}), 400

    positions = {p["id"]: p for p in db.get_open_positions(uid)}
    pos = positions.get(int(pid))
    if not pos:
        return jsonify({"error": "Position not found or already closed."}), 404

    symbol   = pos["stock_symbol"]
    currency = "₹" if symbol.endswith(".NS") or symbol.endswith(".BO") else "$"
    price, _ = fetch_current_price(symbol)
    if price is None:
        return jsonify({"error": f"Could not fetch price for {symbol}."}), 400

    ok = db.sell_stock(int(pid), price, action="sell", user_id=uid)
    if not ok:
        return jsonify({"error": "Could not close position."}), 500

    pnl  = round((price - float(pos["buy_price"])) * float(pos["quantity"]), 2)
    sign = "+" if pnl >= 0 else ""

    if pos.get("phone_number"):
        send_alert_sms(pos["phone_number"],
            f"StockWise Sell: {symbol} @ {currency}{price:.2f} | P&L: {sign}{currency}{pnl}")

    return jsonify({
        "message":    f"Sold {pos['quantity']} x {symbol} @ {currency}{price:.2f}",
        "sell_price": price,
        "pnl":        pnl,
        "symbol":     symbol,
        "currency":   currency,
    })


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO — VIEW
# ══════════════════════════════════════════════════════════════════

@app.route("/portfolio")
@login_required
def portfolio_view():
    uid       = current_user_id()
    positions = db.get_all_positions(uid)
    summary   = db.get_portfolio_summary(uid) or {}

    enriched = []
    for p in positions:
        row = dict(p)
        sym = row.get("stock_symbol", "")
        row["currency"] = "₹" if sym.endswith(".NS") or sym.endswith(".BO") else "$"
        if row["status"] == "open" and row.get("current_price") and row.get("buy_price"):
            row["unrealised_pnl"] = round(
                (float(row["current_price"]) - float(row["buy_price"])) * float(row["quantity"]), 2
            )
        else:
            row["unrealised_pnl"] = None
        for k in ("bought_at", "sold_at"):
            if row.get(k):
                row[k] = str(row[k])
        enriched.append(row)

    return jsonify({
        "positions": enriched,
        "summary": {
            "open_count":   int(summary.get("open_count") or 0),
            "closed_count": int(summary.get("closed_count") or 0),
            "invested":     float(summary.get("invested") or 0),
            "total_pnl":    float(summary.get("total_pnl") or 0),
        }
    })


# ══════════════════════════════════════════════════════════════════
# AI RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════

import threading as _threading
import datetime as _dt

_reco_generating  = False
_reco_last_error  = None
_reco_started_at  = None


def _generate_and_save_bg(broadcast_sms: bool = False):
    global _reco_generating, _reco_last_error, _reco_started_at
    _reco_last_error = None
    try:
        from recommender import generate_recommendations, save_recommendations
        recs = generate_recommendations()
        if recs:
            save_recommendations(recs)
            print(f"[App] Background recommendations done: {len(recs)} saved.")
            if broadcast_sms:
                try:
                    from scheduler import _broadcast_recommendations
                    _broadcast_recommendations(recs)
                except Exception as sms_err:
                    print(f"[App] SMS broadcast error: {sms_err}")
        else:
            _reco_last_error = "No data returned from market APIs. Try again in a few minutes."
            print(f"[App] No recommendations generated.")
    except Exception as e:
        _reco_last_error = str(e)
        print(f"[App] Background recommendation error: {e}")
    finally:
        _reco_generating = False
        _reco_started_at = None


@app.route("/recommendations")
@login_required
def get_recommendations():
    global _reco_generating, _reco_last_error
    try:
        from recommender import get_todays_recommendations
        recs = get_todays_recommendations()

        if recs:
            return jsonify({
                "recommendations": recs,
                "date":            str(_dt.date.today()),
                "generating":      False,
                "count":           len(recs),
            })

        if not _reco_generating:
            _reco_generating  = True
            _reco_started_at  = _dt.datetime.utcnow()
            t = _threading.Thread(target=_generate_and_save_bg, daemon=True)
            t.start()
            print("[App] Auto-started background recommendation generation.")

        return jsonify({
            "recommendations": [],
            "date":            str(_dt.date.today()),
            "generating":      True,
            "error":           _reco_last_error,
            "message":         "Generating recommendations… this takes ~20-30 seconds.",
        })
    except Exception as e:
        print(f"[Recommendations] Error: {e}")
        return jsonify({"recommendations": [], "error": str(e), "generating": False}), 500


@app.route("/recommendations/refresh", methods=["POST"])
@login_required
def refresh_recommendations():
    global _reco_generating, _reco_last_error, _reco_started_at
    if _reco_generating:
        return jsonify({
            "message":    "Already generating — please wait ~30 seconds.",
            "generating": True,
        })

    data      = request.get_json(silent=True) or {}
    broadcast = bool(data.get("broadcast", False))

    _reco_generating  = True
    _reco_last_error  = None
    _reco_started_at  = _dt.datetime.utcnow()
    t = _threading.Thread(
        target=_generate_and_save_bg,
        kwargs={"broadcast_sms": broadcast},
        daemon=True,
    )
    t.start()

    return jsonify({
        "message":    "Refresh started. Poll GET /recommendations every 5s.",
        "generating": True,
        "broadcast":  broadcast,
    })


@app.route("/recommendations/broadcast", methods=["POST"])
@login_required
def broadcast_recommendations():
    try:
        from recommender import get_todays_recommendations
        from scheduler   import _broadcast_recommendations
        recs = get_todays_recommendations()
        if not recs:
            return jsonify({"error": "No recommendations for today yet. Generate them first."}), 400
        _broadcast_recommendations(recs)
        return jsonify({"message": f"SMS broadcast triggered for {len(recs)} picks."})
    except Exception as e:
        print(f"[Broadcast] Error: {e}")
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    # use_reloader=False is CRITICAL — Flask's reloader detects TensorFlow
    # file changes and restarts mid-training, killing the LSTM model save.
    app.run(
        debug=True,
        use_reloader=False,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080))
    )