"""
scheduler.py — Background scheduler
  • Every 5 min: auto-sell check (stop-loss / take-profit)
  • Every day 9:15 AM IST: generate AI recommendations + SMS ALL users
  • Every day 3:30 PM IST: track closing prices
"""
import os
import threading
import time
from datetime import datetime
import pytz
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

from msg import fetch_current_price, send_alert_sms
import db

CHECK_INTERVAL = int(os.getenv("AUTO_SELL_INTERVAL", "300"))
IST = pytz.timezone("Asia/Kolkata")


# ── Auto-sell check ──────────────────────────────────────────────────────────

def _check_auto_sell():
    positions = db.get_open_positions()
    for pos in positions:
        symbol = pos["stock_symbol"]
        pid    = pos["id"]
        if pos["stop_loss"] is None and pos["take_profit"] is None:
            continue
        price, _ = fetch_current_price(symbol)
        if price is None:
            continue
        db.update_current_price(pid, price)
        stop = float(pos["stop_loss"])   if pos["stop_loss"]   else None
        tp   = float(pos["take_profit"]) if pos["take_profit"] else None
        buy  = float(pos["buy_price"])
        triggered = False
        reason    = ""
        if stop and price <= stop:
            triggered = True
            reason    = f"Stop-loss hit at ₹{price:.2f} (limit ₹{stop:.2f})"
        elif tp and price >= tp:
            triggered = True
            reason    = f"Take-profit hit at ₹{price:.2f} (target ₹{tp:.2f})"
        if triggered:
            ok  = db.sell_stock(pid, price, action="auto_sell")
            pnl = round((price - buy) * float(pos["quantity"]), 2)
            sign = "+" if pnl >= 0 else ""
            print(f"[AutoSell] {symbol} pos#{pid} — {reason} | PnL ₹{pnl}")
            if ok and pos.get("phone_number"):
                msg = (f"📊 StockWise Auto-Sell\n"
                       f"{symbol} — {reason}\n"
                       f"Qty: {pos['quantity']} @ ₹{price:.2f}\n"
                       f"P&L: {sign}₹{pnl}")
                send_alert_sms(pos["phone_number"], msg)


# ── Get all phone numbers to notify ──────────────────────────────────────────

def _get_all_phones() -> list:
    """
    Collect phone numbers from ALL registered users who provided one.
    Falls back to alert subscribers if users table has no phones.
    Returns a deduped list.
    """
    phones = set()

    # Primary: every registered user with a phone number
    try:
        conn = db.get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT phone_number FROM users WHERE phone_number IS NOT NULL AND phone_number != ''")
        for row in cur.fetchall():
            phones.add(row[0])
        cur.close()
        db.release_conn(conn)
    except Exception as e:
        print(f"[Scheduler] Error fetching user phones: {e}")

    # Also include phones from user_alerts
    try:
        alerts = db.get_all_alerts()
        for a in alerts:
            if a.get("phone_number"):
                phones.add(a["phone_number"])
    except Exception as e:
        print(f"[Scheduler] Error fetching alert phones: {e}")

    result = list(phones)
    print(f"[Scheduler] Will notify {len(result)} phone numbers.")
    return result


# ── Market schedule ──────────────────────────────────────────────────────────

_last_recommendation_date = None
_last_close_track_date    = None

def _run_market_jobs():
    global _last_recommendation_date, _last_close_track_date

    now     = datetime.now(IST)
    today   = now.date()
    weekday = now.weekday()  # 0=Mon, 4=Fri

    # Only run on weekdays (Mon-Fri)
    if weekday >= 5:
        return

    hour   = now.hour
    minute = now.minute

    # 9:15 AM IST — generate AI recommendations + SMS broadcast
    if hour == 9 and 15 <= minute <= 20:
        if _last_recommendation_date != today:
            _last_recommendation_date = today
            print(f"[Scheduler] Running morning AI recommendations for {today}...")
            try:
                from recommender import generate_recommendations, save_recommendations, track_daily_prices
                track_daily_prices()
                recs = generate_recommendations()
                if recs:
                    save_recommendations(recs)
                    _broadcast_recommendations(recs)
                else:
                    print("[Scheduler] No recommendations generated.")
            except Exception as e:
                print(f"[Scheduler] Recommendation error: {e}")

    # 3:30 PM IST — track closing prices
    if hour == 15 and 30 <= minute <= 35:
        if _last_close_track_date != today:
            _last_close_track_date = today
            print(f"[Scheduler] Tracking closing prices for {today}...")
            try:
                from recommender import track_daily_prices
                track_daily_prices()
            except Exception as e:
                print(f"[Scheduler] Close price tracking error: {e}")


def _broadcast_recommendations(recs: list):
    """
    Send today's top 5 picks via SMS to ALL registered users with a phone number.
    """
    phones = _get_all_phones()
    if not phones:
        print("[Scheduler] No phone numbers found — skipping broadcast.")
        return

    today_str = datetime.now(IST).strftime("%d %b %Y")

    msg  = f"📈 StockWise AI Picks — {today_str}\n"
    msg += "Top 5 stocks for today:\n\n"
    for r in recs:
        sym  = r["symbol"].replace(".NS", "").replace(".BO", "")
        sign = "+" if r["predicted_gain"] >= 0 else ""
        msg += f"{r['rank']}. {sym} ({r.get('company','')[:15]})\n"
        msg += f"   ₹{r['current_price']:.2f} → ₹{r['target_price']:.2f}  {sign}{r['predicted_gain']:.1f}%\n"
        msg += f"   {r.get('reason','')[:60]}\n\n"

    msg += "⚠️ Informational only. Not financial advice."

    sent = 0
    failed = 0
    for phone in phones[:100]:   # hard cap at 100 to protect Twilio credits
        try:
            ok = send_alert_sms(phone, msg)
            if ok:
                sent += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[Scheduler] SMS error for {phone}: {e}")
            failed += 1

    print(f"[Scheduler] Broadcast done — {sent} sent, {failed} failed out of {len(phones)} numbers.")


# ── Main loop ────────────────────────────────────────────────────────────────

def _run():
    print("[Scheduler] Started — auto-sell + AI recommendations + SMS broadcast active.")
    while True:
        try:
            _check_auto_sell()
        except Exception as e:
            print(f"[Scheduler] Auto-sell error: {e}")
        try:
            _run_market_jobs()
        except Exception as e:
            print(f"[Scheduler] Market job error: {e}")
        time.sleep(CHECK_INTERVAL)


def start():
    t = threading.Thread(target=_run, daemon=True)
    t.start()