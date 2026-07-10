"""
ai_routes.py — GenAI + RAG feature routes for StockWise, powered by Groq.

Register in app.py:
    from ai_routes import ai_bp
    app.register_blueprint(ai_bp)

Endpoints:
    POST /api/ai/chat                 — RAG chat over live data + portfolio
    GET  /api/ai/explain/<symbol>     — natural-language recommendation rationale
    GET  /api/ai/portfolio-insights   — structured JSON risk/health summary
    GET  /api/ai/news-digest/<symbol> — retrieval-filtered news -> sentiment digest
"""
import json
import os

import requests
from flask import Blueprint, request, jsonify, session

import db
from groq_client import chat
from rag_engine import get_context
import rag_chat_agent

ai_bp = Blueprint("ai", __name__, url_prefix="/api/ai")


def _uid():
    return session.get("user_id")


# ══════════════════════════════════════════════════════════════════
# 1. RAG CHAT
# ══════════════════════════════════════════════════════════════════

@ai_bp.route("/chat", methods=["POST"])
def ai_chat():
    if "user_id" not in session:
        return jsonify({"error": "Login required."}), 401

    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "message is required."}), 400

    try:
        reply = rag_chat_agent.answer(question, user_id=_uid())
        return jsonify({"answer": reply})
    except Exception as e:
        print(f"[AI] chat error: {e}")
        return jsonify({"error": "AI service unavailable. Try again shortly."}), 500


# ══════════════════════════════════════════════════════════════════
# 2. NATURAL-LANGUAGE RECOMMENDATION RATIONALE
# ══════════════════════════════════════════════════════════════════

@ai_bp.route("/explain/<symbol>")
def ai_explain(symbol):
    symbol = symbol.upper()
    from recommender import get_todays_recommendations
    recs = {r["stock_symbol"]: r for r in get_todays_recommendations()}
    r = recs.get(symbol)
    if not r:
        return jsonify({"error": "No recommendation found for this symbol today."}), 404

    prompt = (
        f"Stock: {symbol} ({r.get('company_name')})\n"
        f"Current price: Rs.{r['current_price']}, Target: Rs.{r['target_price']}, "
        f"Predicted gain: {r['predicted_gain']}%\n"
        f"Technical reason: {r['reason']}\n\n"
        "Write a 3-4 sentence, plain-English explanation of why this stock was "
        "flagged today, suitable for a retail user with no trading background. "
        "End with: 'Not financial advice.'"
    )
    messages = [
        {"role": "system", "content": (
            "You are a financial explainer for a retail investing app. Be clear, "
            "honest about uncertainty, and never guarantee returns."
        )},
        {"role": "user", "content": prompt},
    ]
    try:
        explanation = chat(messages, temperature=0.5, max_tokens=200)
        return jsonify({"symbol": symbol, "explanation": explanation})
    except Exception as e:
        print(f"[AI] explain error: {e}")
        return jsonify({"error": "AI service unavailable."}), 500


# ══════════════════════════════════════════════════════════════════
# 3. PORTFOLIO RISK / HEALTH ANALYZER (structured JSON output)
# ══════════════════════════════════════════════════════════════════

@ai_bp.route("/portfolio-insights")
def portfolio_insights():
    if "user_id" not in session:
        return jsonify({"error": "Login required."}), 401

    uid = _uid()
    positions = db.get_all_positions(uid)
    summary   = db.get_portfolio_summary(uid)

    if not positions:
        return jsonify({
            "insight": {
                "risk_level": "N/A",
                "diversification_comment": "No positions yet.",
                "top_concern": "None.",
                "suggestion": "Buy a stock to get AI portfolio insights.",
            }
        })

    lines = [
        f"{p['stock_symbol']} qty={p['quantity']} buy=Rs.{p['buy_price']} "
        f"status={p['status']} pnl={p.get('pnl')}"
        for p in positions
    ]
    prompt = (
        "Portfolio positions:\n" + "\n".join(lines) +
        f"\n\nSummary: {summary}\n\n"
        "Respond ONLY as JSON with keys: risk_level (Low/Medium/High), "
        "diversification_comment (1 sentence), top_concern (1 sentence), "
        "suggestion (1 sentence, general framing only, not financial advice)."
    )
    messages = [
        {"role": "system", "content": (
            "You are a risk-analysis assistant for a paper-trading app. "
            "Output valid JSON only, no markdown fences."
        )},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = chat(messages, temperature=0.3, max_tokens=300, json_mode=True)
        return jsonify({"insight": json.loads(raw)})
    except Exception as e:
        print(f"[AI] portfolio insight error: {e}")
        return jsonify({"error": "AI service unavailable."}), 500


# ══════════════════════════════════════════════════════════════════
# 4. NEWS SENTIMENT DIGEST (retrieval-filtered headlines -> summary)
# ══════════════════════════════════════════════════════════════════

@ai_bp.route("/news-digest/<symbol>")
def news_digest(symbol):
    symbol  = symbol.upper()
    company = request.args.get("company", symbol)
    news_key = os.getenv("NEWS_API_KEY", "")

    if not news_key:
        return jsonify({"error": "NEWS_API_KEY not configured."}), 400

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "apiKey": news_key, "qInTitle": company,
                "pageSize": 5, "sortBy": "publishedAt",
            },
            timeout=10,
        )
        articles = resp.json().get("articles", [])
    except Exception as e:
        return jsonify({"error": f"News fetch failed: {e}"}), 500

    if not articles:
        return jsonify({"symbol": symbol, "sentiment": "neutral",
                         "summary": "No recent news found.", "headlines": []})

    # Retrieval step — keep only articles that actually mention the company,
    # rather than blindly summarizing whatever NewsAPI returned.
    relevant = [
        a for a in articles
        if company.lower() in (a.get("title", "") + a.get("description", "")).lower()
    ] or articles

    headlines = "\n".join(f"- {a.get('title', '')}" for a in relevant[:5])
    prompt = (
        f"Company: {company} ({symbol})\nRecent headlines:\n{headlines}\n\n"
        "Summarize the overall news sentiment in 2-3 sentences and classify it "
        "as one of: bullish, bearish, neutral, mixed. Respond ONLY as JSON with "
        "keys 'sentiment' and 'summary'."
    )
    messages = [
        {"role": "system", "content": (
            "You are a financial news summarizer. Output valid JSON only. Be "
            "factual — do not speculate beyond the given headlines."
        )},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = chat(messages, temperature=0.2, max_tokens=250, json_mode=True)
        result = json.loads(raw)
        result["symbol"] = symbol
        result["headlines"] = [a.get("title", "") for a in relevant[:5]]
        return jsonify(result)
    except Exception as e:
        print(f"[AI] news digest error: {e}")
        return jsonify({"error": "AI service unavailable."}), 500