# msg.py — yfinance for prices (no API limit), Twilio for alerts
import os
import requests
import yfinance as yf
from twilio.rest import Client

# ── Twilio credentials ──────────────────────────────────────────────────────
account_sid            = os.getenv("TWILIO_ACCOUNT_SID")
auth_token             = os.getenv("TWILIO_AUTH_TOKEN")
twilio_sms_number      = os.getenv("TWILIO_SMS_NUMBER")
twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
NEWS_API_KEY           = os.getenv("NEWS_API_KEY", "")

client = Client(account_sid, auth_token)

# ── TwelveData (optional — only used if key is set and symbol is supported) ──

TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "")

def _to_td_symbol(symbol: str) -> str:
    if symbol.endswith(".NS"): return symbol.replace(".NS", "") + ":NSE"
    if symbol.endswith(".BO"): return symbol.replace(".BO", "") + ":BSE"
    return symbol

def _fetch_twelvedata(symbol: str):
    """Fetch current price from Twelve Data API. Only called if key is set."""
    if not TWELVE_DATA_KEY:
        return None
    try:
        td_sym = _to_td_symbol(symbol)
        url    = f"https://api.twelvedata.com/price?symbol={td_sym}&apikey={TWELVE_DATA_KEY}"
        resp   = requests.get(url, timeout=8)
        data   = resp.json()
        if "price" in data:
            price = float(data["price"])
            print(f"[TwelveData] {symbol} = {price}")
            return price
        # Silently skip on plan/credit errors — yfinance will handle it
        msg = data.get("message", "unknown error")
        print(f"[TwelveData] Skipping {symbol}: {msg}")
    except Exception as e:
        print(f"[TwelveData] Error for {symbol}: {e}")
    return None


def fetch_current_price(symbol: str):
    """
    Fetch latest price.
    Primary: yfinance (no rate limits, no plan restrictions, works for all .NS symbols).
    Fallback: TwelveData (only if TWELVE_DATA_KEY env var is set).
    Returns (price, symbol) or (None, None) on failure.
    """
    import time

    # Primary: yfinance — reliable, no API key needed
    for attempt in range(2):
        try:
            ticker = yf.Ticker(symbol)
            try:
                p = ticker.fast_info.last_price
                if p and p > 0:
                    return float(p), symbol
            except Exception:
                pass
            hist = ticker.history(period="2d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1]), symbol
        except Exception as e:
            print(f"[yfinance] Attempt {attempt+1} failed for {symbol}: {e}")
            if attempt < 1:
                time.sleep(2)

    # Fallback: TwelveData (if key configured)
    price = _fetch_twelvedata(symbol)
    if price and price > 0:
        return price, symbol

    print(f"[Price] All sources failed for {symbol}")
    return None, None


# ── SMS ─────────────────────────────────────────────────────────────────────

def send_alert_sms(to_phone_number: str, message: str) -> bool:
    if not all([account_sid, auth_token, twilio_sms_number]):
        print("Twilio SMS credentials not set.")
        return False
    try:
        resp = client.messages.create(
            body=message, from_=twilio_sms_number, to=to_phone_number
        )
        print(f"SMS sent: {resp.sid}")
        return True
    except Exception as e:
        print(f"SMS error to {to_phone_number}: {e}")
        return False


# ── WhatsApp ─────────────────────────────────────────────────────────────────

def send_alert_whatsapp(to_number: str, message: str) -> bool:
    if not all([account_sid, auth_token, twilio_whatsapp_number]):
        print("Twilio WhatsApp credentials not set.")
        return False
    if not to_number.startswith("whatsapp:"):
        to_number = "whatsapp:" + to_number
    try:
        resp = client.messages.create(
            body=message, from_=twilio_whatsapp_number, to=to_number
        )
        print(f"WhatsApp sent: {resp.sid}")
        return True
    except Exception as e:
        print(f"WhatsApp error to {to_number}: {e}")
        return False


# ── News + price change alert ─────────────────────────────────────────────────

def send_stock_news_alert(stock_symbol: str, company_name: str,
                          phone_number: str, threshold_percent: int = 1) -> bool:
    """
    Checks 2-day price change using yfinance.
    Sends SMS + WhatsApp news alert if change >= threshold.
    Works for both Indian (.NS) and US stocks.
    """
    try:
        ticker = yf.Ticker(stock_symbol)
        hist   = ticker.history(period="5d")

        if len(hist) < 2:
            print(f"Not enough data for {stock_symbol}")
            return False

        yesterday  = float(hist["Close"].iloc[-1])
        day_before = float(hist["Close"].iloc[-2])
        diff       = yesterday - day_before
        diff_pct   = round((diff / day_before) * 100, 2)
        up_down    = "🔺" if diff > 0 else "🔻"

        display_sym = stock_symbol.replace(".NS", "").replace(".BO", "")

        print(f"[Alert] {display_sym}: {up_down}{diff_pct}% change")

        if abs(diff_pct) < threshold_percent:
            print(f"Below threshold ({threshold_percent}%). No alert sent.")
            return False

        # Fetch news (only if NEWS_API_KEY is configured)
        articles = []
        if NEWS_API_KEY:
            try:
                news_resp = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={"apiKey": NEWS_API_KEY, "qInTitle": company_name, "pageSize": 3},
                    timeout=10
                )
                articles = news_resp.json().get("articles", [])
            except Exception as e:
                print(f"[News] Failed to fetch news for {company_name}: {e}")

        if not articles:
            msg = (f"📊 StockWise Alert\n"
                   f"{display_sym} ({company_name})\n"
                   f"{up_down} {diff_pct}% price change\n"
                   f"Current: ₹{yesterday:.2f}")
            send_alert_sms(phone_number, msg)
            send_alert_whatsapp(phone_number, msg)
            return True

        sent = False
        for article in articles:
            msg = (f"📊 {display_sym}: {up_down}{diff_pct}%\n"
                   f"📰 {article.get('title', '')}\n"
                   f"💬 {article.get('description', '')[:100]}")
            if send_alert_sms(phone_number, msg):
                sent = True
            send_alert_whatsapp(phone_number, msg)

        return sent

    except Exception as e:
        print(f"[send_stock_news_alert] Error for {stock_symbol}: {e}")
        return False