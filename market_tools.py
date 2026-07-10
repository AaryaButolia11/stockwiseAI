"""
market_tools.py — Live market data & news tools for the StockWise chat agent.

These are LangChain @tool-decorated functions so the LLM (via LangChain +
Groq, see rag_chat_agent.py) can call them on demand instead of only
answering from the static DB/RAG context. This is what lets the assistant
answer "best stock to buy today" with real numbers instead of
"no relevant data found".

Design notes:
  • get_top_movers() uses a single batched yf.download() call across the
    whole tracked symbol list instead of one request per symbol — much
    faster and less likely to get rate-limited.
  • Results are cached in memory for a few minutes (market data doesn't
    need to be recomputed on every message in a conversation).
  • get_market_news() uses Google News RSS (no API key required). Swap in
    a paid news API (NewsAPI, Finnhub, etc.) later if you want more
    structured/reliable results — the tool's interface stays the same.
"""

COMPANY_MAP = {
    "reliance industries": "RELIANCE.NS",
    "tata consultancy services": "TCS.NS",
    "tcs": "TCS.NS",
    "hdfc bank": "HDFCBANK.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "icici bank": "ICICIBANK.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    "hul": "HINDUNILVR.NS",
    "state bank of india": "SBIN.NS",
    "sbi": "SBIN.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS",
    "wipro": "WIPRO.NS",
    "zomato": "ETERNAL.NS",
    "eternal": "ETERNAL.NS",
    "adani ports": "ADANIPORTS.NS",
    "tata motors": "TATAMOTORS.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",
    "sun pharma": "SUNPHARMA.NS",
    "sun pharmaceutical": "SUNPHARMA.NS",
    "titan": "TITAN.NS",
    "ultratech cement": "ULTRACEMCO.NS",
    "ongc": "ONGC.NS",
    "power grid": "POWERGRID.NS",
    "ntpc": "NTPC.NS",
    "itc": "ITC.NS",
    "kotak bank": "KOTAKBANK.NS",
    "kotak mahindra bank": "KOTAKBANK.NS",
    "l&t": "LT.NS",
    "larsen and toubro": "LT.NS",
    "axis bank": "AXISBANK.NS",
    "nestle": "NESTLEIND.NS",
    "nestle india": "NESTLEIND.NS",
    "tech mahindra": "TECHM.NS",
    "hcl": "HCLTECH.NS",
    "hcl technologies": "HCLTECH.NS",
    "asian paints": "ASIANPAINT.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "britannia": "BRITANNIA.NS",
    "cipla": "CIPLA.NS",
    "coal india": "COALINDIA.NS",
    "divis laboratories": "DIVISLAB.NS",
    "dr reddys": "DRREDDY.NS",
    "dr reddy's": "DRREDDY.NS",
    "eicher motors": "EICHERMOT.NS",
    "grasim": "GRASIM.NS",
    "hero motocorp": "HEROMOTOCO.NS",
    "hindalco": "HINDALCO.NS",
    "indusind bank": "INDUSINDBK.NS",
    "jsw steel": "JSWSTEEL.NS",
    "mahindra and mahindra": "M&M.NS",
    "mahindra": "M&M.NS",
    "pidilite": "PIDILITIND.NS",
    "sbi life": "SBILIFE.NS",
    "shree cement": "SHREECEM.NS",
    "tata consumer": "TATACONSUM.NS",
    "tata steel": "TATASTEEL.NS",
    "trent": "TRENT.NS",
    "upl": "UPL.NS",
    "vedanta": "VEDL.NS",
    "bpcl": "BPCL.NS",
    "bharat petroleum": "BPCL.NS",
}

import os
import csv
from datetime import datetime, timedelta

import yfinance as yf
import requests
import xml.etree.ElementTree as ET

from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader

_COMPANIES_CSV = os.path.join(os.path.dirname(__file__), "companies_india.csv")

_MOVERS_CACHE = {"ts": None, "data": None}
_MOVERS_TTL = timedelta(minutes=5)


def _load_nse_symbols(limit: int = 50) -> list[str]:
    """Reads the same companies_india.csv rag_engine.py already uses, so
    the 'today's movers' universe matches what StockWise tracks."""
    symbols = []
    try:
        with open(_COMPANIES_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sym = row.get("Symbol")
                if not sym:
                    continue
                symbols.append(sym if sym.endswith((".NS", ".BO")) else sym + ".NS")
                if len(symbols) >= limit:
                    break
    except Exception as e:
        print(f"[market_tools] couldn't read companies_india.csv: {e}")
    return symbols


@tool
def get_top_movers(direction: str = "gainers", n: int = 5) -> str:
    """Get today's top N gaining or losing stocks from StockWise's tracked
    NSE universe. direction must be 'gainers' or 'losers'. Use this for
    questions like 'best stock to buy today' or 'what's moving right now'."""
    now = datetime.now()
    if _MOVERS_CACHE["data"] is not None and now - _MOVERS_CACHE["ts"] < _MOVERS_TTL:
        rows = _MOVERS_CACHE["data"]
    else:
        symbols = _load_nse_symbols()
        if not symbols:
            return "No tracked symbols available."
        try:
            data = yf.download(symbols, period="2d", group_by="ticker",
                                progress=False, threads=True)
        except Exception as e:
            return f"Live market data unavailable right now ({e})."

        rows = []
        for sym in symbols:
            try:
                closes = data[sym]["Close"].dropna()
                if len(closes) < 2:
                    continue
                prev, curr = closes.iloc[-2], closes.iloc[-1]
                pct = (curr - prev) / prev * 100
                rows.append({
                    "symbol": sym.replace(".NS", "").replace(".BO", ""),
                    "price": round(float(curr), 2),
                    "pct_change": round(float(pct), 2),
                })
            except Exception:
                continue
        _MOVERS_CACHE["data"], _MOVERS_CACHE["ts"] = rows, now

    if not rows:
        return "No live price data available right now."

    reverse = direction.lower() != "losers"
    top = sorted(rows, key=lambda r: r["pct_change"], reverse=reverse)[:n]
    lines = [f"{r['symbol']}: Rs.{r['price']} ({r['pct_change']:+.2f}%)" for r in top]
    return f"Top {n} {direction} today — " + "; ".join(lines)


@tool
def get_stock_quote(symbol: str) -> str:
    """
    ALWAYS use this whenever the user mentions a stock/company.

    Examples:
    - Infosys
    - How is TCS today?
    - How's Infosys market right now?
    - Reliance price
    - Is SBI going up?
    """
    sym = COMPANY_MAP.get(symbol.strip().lower(), symbol)
    # sym = symbol.upper().strip()
    if not sym.endswith((".NS", ".BO")):
        sym += ".NS"
    try:
        t = yf.Ticker(sym)
        hist = t.history(period="2d")
        if hist.empty:
            return f"No price data found for {symbol}."
        closes = hist["Close"]
        prev = closes.iloc[-2] if len(closes) > 1 else closes.iloc[-1]
        curr = closes.iloc[-1]
        pct = (curr - prev) / prev * 100 if prev else 0
        info = t.fast_info
        return (f"{sym}: current price Rs.{curr:.2f} ({pct:+.2f}% today), "
                f"day range Rs.{info.get('dayLow', 'NA')}-Rs.{info.get('dayHigh', 'NA')}, "
                f"52wk range Rs.{info.get('yearLow', 'NA')}-Rs.{info.get('yearHigh', 'NA')}.")
    except Exception as e:
        return f"Couldn't fetch live quote for {symbol}: {e}"


@tool
def get_stock_news(symbol: str, n: int = 3) -> str:
    """Get the latest N news headlines for a specific stock symbol, e.g.
    'TCS' or 'INFY'. Use this when the user asks why a stock is moving or
    wants recent company news."""
    sym = COMPANY_MAP.get(symbol.strip().lower(), symbol)
    # sym = symbol.upper().strip()
    if not sym.endswith((".NS", ".BO")):
        sym += ".NS"
    try:
        items = yf.Ticker(sym).news or []
        lines = []
        for it in items[:n]:
            content = it.get("content") or {}
            title = it.get("title") or content.get("title")
            link = (
                it.get("link")
                or content.get("canonicalUrl", {}).get("url")
                or content.get("clickThroughUrl", {}).get("url")
            )
            if title:
                lines.append(f"{title}" + (f" ({link})" if link else ""))
        return (f"Recent headlines for {symbol}: " + " | ".join(lines)) if lines \
            else f"No recent news found for {symbol}."
    except Exception as e:
        return f"Couldn't fetch news for {symbol}: {e}"


@tool
def get_market_news(query: str = "Indian stock market", n: int = 5) -> str:
    """Get general financial/market news headlines, not tied to one stock.
    Use for questions like 'what's happening in the market today' or
    'any big news today'. query can narrow the topic, e.g. 'RBI repo rate'."""
    try:
        url = ("https://news.google.com/rss/search?q="
               f"{requests.utils.quote(query)}+when:1d&hl=en-IN&gl=IN&ceid=IN:en")
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        lines = []
        for item in root.findall(".//item")[:n]:
            title_el = item.find("title")
            link_el = item.find("link")
            if title_el is None:
                continue
            title = title_el.text
            link = link_el.text if link_el is not None else None
            lines.append(f"{title}" + (f" ({link})" if link else ""))
        return ("Top market headlines: " + " | ".join(lines)) if lines \
            else "No current market news found."
    except Exception as e:
        return f"Couldn't fetch market news right now ({e})."


@tool
def read_full_article(url: str) -> str:
    """Fetch and read the full text of a specific news article URL (a link
    you got back from get_stock_news or get_market_news). Use this when a
    headline alone doesn't answer the user's question — e.g. 'why did TCS
    drop today' needs the article body, not just the title."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs or not docs[0].page_content.strip():
            return f"Couldn't extract readable content from {url}."
        text = docs[0].page_content.strip()
        return text[:2500] + ("..." if len(text) > 2500 else "")
    except Exception as e:
        return f"Couldn't fetch article at {url}: {e}"