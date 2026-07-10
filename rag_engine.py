"""
rag_engine.py — Lightweight Retrieval-Augmented Generation layer for StockWise.

WHY THIS DESIGN (worth saying out loud in an interview):
  • Corpus = short text "documents" built on the fly from data StockWise
    already has: today's AI recommendations, recent OHLC price history,
    the logged-in user's own portfolio, and Nifty-50 company reference facts.
  • Retrieval = TF-IDF + cosine similarity (scikit-learn is already a
    dependency for the LSTM model — no new heavy libs needed).
  • At this scale (~50 stocks, a few hundred short chunks) TF-IDF is fast
    enough to rebuild per-request, which keeps the index always fresh
    (no stale prices). The retrieve() interface is the same shape you'd
    use with FAISS/pgvector + sentence embeddings — swapping the backend
    later doesn't touch the rest of the app.
  • The retrieved chunks are stitched into the LLM prompt as CONTEXT, so
    Groq answers from real numbers instead of guessing — this is what
    actually makes it "RAG" rather than "an LLM with a system prompt".

FIX (2024): TF-IDF similarity against generic phrasing like "best stock for
short term purchase" often scores too low to surface templated recommendation
text. retrieve() now guarantees recommendation docs are included whenever
they exist, regardless of raw cosine score, because they're cheap and are
almost always what a "which stock should I buy" question actually needs.
"""
import os
import csv
import threading
from datetime import date, timedelta

import psycopg2.extras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import db

_lock = threading.Lock()


# ── Document builders ────────────────────────────────────────────────────

def _company_docs():
    docs = []
    path = os.path.join(os.path.dirname(__file__), "companies_india.csv")
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sym, name = row.get("Symbol"), row.get("Company")
                if sym and name:
                    docs.append({
                        "id": f"company:{sym}",
                        "text": f"{name} trades on the NSE under the symbol {sym}.",
                        "meta": {"type": "company", "symbol": sym},
                    })
    except Exception as e:
        print(f"[RAG] company docs error: {e}")
    return docs


def _recommendation_docs():
    docs = []
    try:
        from recommender import get_todays_recommendations
        for r in get_todays_recommendations():
            text = (
                f"AI recommendation rank {r['rank']} for {r['stock_symbol']} "
                f"({r.get('company_name', '')}): current price Rs.{r['current_price']}, "
                f"target price Rs.{r['target_price']}, predicted gain "
                f"{r['predicted_gain']}%. Reason: {r['reason']}."
            )
            docs.append({
                "id": f"reco:{r['stock_symbol']}",
                "text": text,
                "meta": {"type": "recommendation", "symbol": r["stock_symbol"]},
            })
    except Exception as e:
        print(f"[RAG] recommendation docs error: {e}")
    return docs


def _price_history_docs(days: int = 5):
    docs = []
    conn = cur = None
    try:
        conn = db.get_conn()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT stock_symbol, date, open_price, close_price, pct_change
            FROM daily_prices
            WHERE date >= %s
            ORDER BY stock_symbol, date DESC
        """, (date.today() - timedelta(days=days),))
        by_symbol = {}
        for row in cur.fetchall():
            by_symbol.setdefault(row["stock_symbol"], []).append(row)
        for sym, rows in by_symbol.items():
            parts = [
                f"{r['date']}: open Rs.{r['open_price']}, close Rs.{r['close_price']} "
                f"({r['pct_change']}%)" for r in rows
            ]
            docs.append({
                "id": f"price:{sym}",
                "text": f"Recent price history for {sym} — " + "; ".join(parts) + ".",
                "meta": {"type": "price_history", "symbol": sym},
            })
    except Exception as e:
        print(f"[RAG] price history docs error: {e}")
    finally:
        if cur:  cur.close()
        if conn: db.release_conn(conn)
    return docs


def _portfolio_docs(user_id: int):
    docs = []
    try:
        positions = db.get_all_positions(user_id)
        for p in positions:
            status = p["status"]
            text = (
                f"User position: {p['quantity']} shares of {p['stock_symbol']} "
                f"({p['company_name']}) bought at Rs.{p['buy_price']}, status={status}."
            )
            if status == "open" and p.get("current_price"):
                text += f" Current price Rs.{p['current_price']}."
            if p.get("stop_loss"):
                text += f" Stop-loss Rs.{p['stop_loss']}."
            if p.get("take_profit"):
                text += f" Take-profit Rs.{p['take_profit']}."
            if p.get("pnl") is not None:
                text += f" Realised PnL Rs.{p['pnl']}."
            docs.append({
                "id": f"portfolio:{p['id']}",
                "text": text,
                "meta": {"type": "portfolio", "symbol": p["stock_symbol"]},
            })

        summary = db.get_portfolio_summary(user_id)
        if summary:
            docs.append({
                "id": "portfolio:summary",
                "text": (
                    f"Portfolio summary: {summary.get('open_count', 0)} open positions, "
                    f"{summary.get('closed_count', 0)} closed, total invested "
                    f"Rs.{summary.get('invested', 0)}, total realised PnL "
                    f"Rs.{summary.get('total_pnl', 0)}."
                ),
                "meta": {"type": "portfolio_summary"},
            })
    except Exception as e:
        print(f"[RAG] portfolio docs error: {e}")
    return docs


# ── TF-IDF index ──────────────────────────────────────────────────────────

class _RagIndex:
    def __init__(self):
        self.docs = []
        self.vectorizer = None
        self.matrix = None

    def build(self, user_id: int = None):
        docs = _company_docs() + _recommendation_docs() + _price_history_docs()
        if user_id:
            docs += _portfolio_docs(user_id)

        if not docs:
            self.docs, self.vectorizer, self.matrix = [], None, None
            return

        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in docs])

    def retrieve(self, query: str, k: int = 6, min_reco: int = 3):
        """
        Returns up to k relevant chunks.

        FIX: previously only returned docs above a raw similarity threshold
        (sims[i] > 0.03), which meant generic questions like "best stock for
        short term purchase" often surfaced ZERO recommendation docs even
        though that's exactly the data the user wants. Now we always
        guarantee up to `min_reco` recommendation docs are present in the
        result, topped up with whatever else scores well.
        """
        if not self.docs or self.vectorizer is None:
            return []

        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        hits = [self.docs[i] for i in top_idx if sims[i] > 0.03]

        has_reco = any(h["meta"]["type"] == "recommendation" for h in hits)
        if not has_reco:
            reco_docs = [d for d in self.docs if d["meta"]["type"] == "recommendation"]
            if reco_docs:
                n_reco = min(min_reco, len(reco_docs))
                remaining_slots = max(k - n_reco, 0)
                hits = reco_docs[:n_reco] + hits[:remaining_slots]

        return hits


def get_context(query: str, user_id: int = None, k: int = 6) -> str:
    """
    Build a fresh index and return the top-k relevant chunks as a single
    context block, ready to drop into an LLM prompt.

    Rebuilding per request is deliberate — it keeps prices/recommendations
    always current. For higher traffic you'd cache the index and refresh it
    on a timer (e.g. whenever recommender.py / scheduler.py run), which is
    a natural "next step" talking point in an interview.
    """
    with _lock:
        idx = _RagIndex()
        idx.build(user_id=user_id)
        hits = idx.retrieve(query, k=k)

    if not hits:
        return "No relevant data found in the knowledge base."
    return "\n".join(f"- {h['text']}" for h in hits)


def get_recommendation_context(limit: int = 5) -> str:
    """
    Explicit helper (used by rag_chat_agent.py as a hard fallback) that
    returns today's AI recommendations directly, bypassing TF-IDF entirely.
    Cheap to call and guarantees the chat agent always has something
    concrete to answer "best stock" style questions with.
    """
    try:
        from recommender import get_todays_recommendations
        recos = get_todays_recommendations()
        if not recos:
            return ""
        lines = []
        for r in recos[:limit]:
            lines.append(
                f"- Rank {r['rank']}: {r['stock_symbol']} "
                f"({r.get('company_name', '')}) — current Rs.{r['current_price']}, "
                f"target Rs.{r['target_price']}, predicted gain {r['predicted_gain']}%. "
                f"Reason: {r['reason']}."
            )
        return "\n".join(lines)
    except Exception as e:
        print(f"[RAG] get_recommendation_context error: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# NEW: Portfolio context — hard fallback for "my portfolio" questions
# ══════════════════════════════════════════════════════════════════

def get_portfolio_context(user_id: int = None, max_closed: int = 10) -> str:
    """
    Explicit helper (used by rag_chat_agent.py as a hard fallback, same
    pattern as get_recommendation_context) that returns the logged-in
    user's actual holdings directly, bypassing TF-IDF entirely.

    Guarantees questions like "is my portfolio too concentrated in
    banking?" or "how am I doing overall?" are answered against real
    positions instead of the model improvising generic advice — TF-IDF
    similarity against short, abstract questions like that often scores
    too low to reliably surface the user's own portfolio docs.
    """
    if not user_id:
        return ""
    try:
        positions = db.get_all_positions(user_id)
        if not positions:
            return "User currently holds no positions (portfolio is empty)."

        open_positions   = [p for p in positions if p["status"] == "open"]
        closed_positions = [p for p in positions if p["status"] != "open"]

        lines = []

        if open_positions:
            lines.append("Open positions:")
            for p in open_positions:
                line = (
                    f"- {p['stock_symbol']} ({p.get('company_name', '')}): "
                    f"{p['quantity']} shares @ buy Rs.{p['buy_price']}"
                )
                if p.get("current_price"):
                    line += f", current Rs.{p['current_price']}"
                if p.get("stop_loss"):
                    line += f", stop-loss Rs.{p['stop_loss']}"
                if p.get("take_profit"):
                    line += f", take-profit Rs.{p['take_profit']}"
                lines.append(line)
        else:
            lines.append("No open positions.")

        if closed_positions:
            lines.append("\nClosed positions (most recent):")
            for p in closed_positions[:max_closed]:
                pnl  = p.get("pnl") or 0
                sign = "+" if pnl >= 0 else ""
                lines.append(
                    f"- {p['stock_symbol']}: {p['quantity']} shares, "
                    f"realised PnL {sign}Rs.{pnl}"
                )

        summary = db.get_portfolio_summary(user_id)
        if summary:
            lines.append(
                f"\nSummary: {summary.get('open_count', 0)} open, "
                f"{summary.get('closed_count', 0)} closed, total invested "
                f"Rs.{summary.get('invested', 0)}, total realised PnL "
                f"Rs.{summary.get('total_pnl', 0)}."
            )

        return "\n".join(lines)
    except Exception as e:
        print(f"[RAG] get_portfolio_context error: {e}")
        return ""