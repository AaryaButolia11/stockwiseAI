"""
rag_chat_agent.py — LangChain + Groq agent for the StockWise chat assistant.

Replaces a plain "stuff context into groq_client.chat()" call with a real
tool-calling agent, so the model can go get live data instead of saying
"no relevant data found":

  1. rag_engine.get_context()  -> StockWise's own DB knowledge (today's AI
     recommendations, the user's portfolio, recent DB-cached prices,
     Nifty-50 company facts). Injected up front, no tool call needed.
  2. market_tools               -> LIVE yfinance prices, today's top
     movers, and live news headlines, fetched on demand via LangChain
     tool calling when the question needs "right now" data.

Groq's low-latency inference is what makes the extra tool-call round trip
still feel instant in a chat UI.

FIX (2024): previously, when a live tool (e.g. get_top_movers) failed, the
model would just say "data unavailable, consult a financial advisor" even
though rag_engine's DB_CONTEXT had real recommendation data sitting right
there unused. Two changes fix this:

  1. SYSTEM_PROMPT now explicitly forbids "consult a financial advisor" as
     a non-answer and requires falling back to DB_CONTEXT recommendations.
  2. answer() adds a hard safety net: for "which stock should I buy" style
     questions, today's AI recommendations are always appended to
     DB_CONTEXT directly via rag_engine.get_recommendation_context(),
     bypassing TF-IDF retrieval entirely, so the model always has concrete
     stock symbols/targets to work with even if retrieval scored low or a
     live tool errored out.

FIX (2026): same problem, different question shape — "is my portfolio too
concentrated in banking?" / "how is my portfolio doing?" often scored too
low under TF-IDF for the user's own portfolio docs to surface, so the model
answered with generic diversification advice instead of looking at the
user's actual holdings. _build_db_context now applies the same hard
fallback pattern used for recommendations: portfolio-style questions always
get the user's real positions appended via
rag_engine.get_portfolio_context(), bypassing retrieval entirely.

Install:
    pip install langchain-core langchain-groq

Env var (same as groq_client.py):
    GROQ_API_KEY=gsk_...
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

import rag_engine
from market_tools import (
    get_top_movers, get_stock_quote, get_stock_news, get_market_news,
    read_full_article,
)

TOOLS = [get_top_movers, get_stock_quote, get_stock_news, get_market_news, read_full_article]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# llama-3.3-70b-versatile is a strong general model on Groq's free tier.
# Swap to "llama-3.1-8b-instant" if you want lower latency for the chat route.
MODEL = "llama-3.1-8b-instant"

# Queries that imply "give me an actual stock pick" — used to force-include
# today's AI recommendations in DB_CONTEXT regardless of TF-IDF score.
PICK_KEYWORDS = (
    "best stock", "top pick", "top stock", "which stock", "what stock",
    "short term", "long term", "should i buy", "should i invest",
    "recommend", "good stock", "safe stock", "stock to buy",
)

# NEW: Queries that imply "look at what I actually hold" — used to force-
# include the user's real portfolio in DB_CONTEXT regardless of TF-IDF
# score. Same rationale as PICK_KEYWORDS above.
PORTFOLIO_KEYWORDS = (
    "my portfolio", "my holdings", "my positions", "my stocks",
    "my investments", "i own", "i hold", "i'm holding", "am i diversified",
    "diversification", "too concentrated", "overexposed", "over-exposed",
    "how am i doing", "how is my portfolio", "my pnl", "my p&l",
    "my profit", "my loss", "rebalance",
)

SYSTEM_PROMPT = """
You are StockWise AI, an expert Indian stock market assistant.

You have TWO knowledge sources:

1. DB_CONTEXT
- Portfolio
- AI recommendations
- Cached stock data
- Company information
- Previous analyses

2. LIVE DATA
Accessible through tools.

------------------------------------------------
TOOL USAGE POLICY (MANDATORY)
------------------------------------------------

You MUST use tools whenever the user asks about ANYTHING that can change over time.

ALWAYS use tools for:

• stock price
• company price
• share price
• current price
• latest price
• today's price
• today's performance
• market today
• market sentiment
• today's news
• latest news
• why stock moved
• buy today
• sell today
• top gainers
• top losers
• best stocks today
• worst stocks today
• strongest stocks today
• weakest stocks today
• market movers
• trending stocks
• Nifty today
• Sensex today

Whenever a company is mentioned,
ALWAYS call get_stock_quote() FIRST.

Examples:

User: Infosys
→ get_stock_quote("Infosys")

User: How is Reliance today?
→ get_stock_quote("Reliance")

User: Should I buy TCS?
→ get_stock_quote("TCS")
→ get_stock_news("TCS")

User: Why is SBI falling?
→ get_stock_quote("SBI")
→ get_stock_news("SBI")

User: Best stocks today?
→ get_top_movers("gainers")

User: Worst stocks today?
→ get_top_movers("losers")

User: Market news
→ get_market_news()

If a news headline contains the answer only partially,
call read_full_article() before answering.

------------------------------------------------
PORTFOLIO QUESTIONS (MANDATORY)
------------------------------------------------

Whenever the user asks about THEIR OWN portfolio, holdings, positions,
diversification, concentration, or P&L (e.g. "is my portfolio too
concentrated in banking?", "how am I doing?", "should I rebalance?"):

1. NEVER answer generically ("diversification is important...") without
   looking at DB_CONTEXT first.
2. DB_CONTEXT contains the user's actual open/closed positions under
   "USER'S CURRENT PORTFOLIO" when relevant — use those real symbols,
   quantities, and sectors to ground your answer.
3. If DB_CONTEXT shows no open positions, say so plainly and suggest
   buying a stock first — do not invent holdings.
4. You may still call get_stock_quote() for any symbol the user actually
   holds, to comment on its current performance.

------------------------------------------------
WHEN NOT TO USE TOOLS
------------------------------------------------

Do NOT use tools for general educational finance questions that do not require live data, e.g.:

• What is PE ratio?
• What is RSI?
• Explain CAGR
• Explain mutual funds
• What is diversification?
• Educational finance questions.

Answer these directly.

------------------------------------------------
IF A LIVE TOOL FAILS OR RETURNS NO DATA
------------------------------------------------

Do NOT just say "data unavailable" and do NOT tell the user to
"consult a financial advisor" or "do your own research" as your whole answer.
That is not an acceptable response and must never be the entire reply.

Instead:
1. Immediately fall back to DB_CONTEXT. Use the AI recommendation entries
   (rank, current price, target price, predicted gain %, reason) to give a
   real, specific answer naming actual stock symbols.
2. You must name at least one stock symbol if DB_CONTEXT contains any
   recommendation data, even if every live tool failed.
3. Mention the live-data gap in ONE short sentence, placed at the END of
   your answer as a caveat — never as the headline or the only content.

------------------------------------------------
IMPORTANT
------------------------------------------------

Never guess.

Never invent prices.

Never invent portfolio holdings — only use what DB_CONTEXT actually lists.

Never answer a live-data question without first calling the appropriate tool.

Never expose tool names.

Never output JSON.

Use DB_CONTEXT whenever useful.

If both DB_CONTEXT and live tools are available,
combine both into one answer.

Investment recommendations should include:

• Current Trend
• Technical View
• Risks
• Recent News
• Short-term Outlook
• Long-term Outlook

Finish recommendations with:

"This is AI-generated analysis, not financial advice."
"""


def _llm():
    return ChatGroq(model=MODEL, temperature=0.3, max_tokens=800).bind_tools(TOOLS)


def _build_db_context(query: str, user_id: int = None) -> str:
    """
    Builds DB_CONTEXT for the prompt. Always runs the normal TF-IDF
    retrieval, then adds hard safety nets for specific question shapes —
    this guarantees the model has concrete data to fall back on even if
    retrieval scored low or a live tool later fails:

      • "which stock should I buy" style questions -> today's AI
        recommendations (existing).
      • "is my portfolio too concentrated" style questions -> the user's
        actual current holdings (new).
    """
    db_context = rag_engine.get_context(query, user_id=user_id)

    query_lower = query.lower()

    if any(kw in query_lower for kw in PICK_KEYWORDS):
        reco_block = rag_engine.get_recommendation_context(limit=5)
        if reco_block:
            db_context += "\n\nTODAY'S AI RECOMMENDATIONS (use these if live data is unavailable):\n" + reco_block

    if any(kw in query_lower for kw in PORTFOLIO_KEYWORDS):
        portfolio_block = rag_engine.get_portfolio_context(user_id=user_id)
        if portfolio_block:
            db_context += "\n\nUSER'S CURRENT PORTFOLIO (use this for portfolio-specific questions — do not invent holdings):\n" + portfolio_block

    return db_context


def answer(query: str, user_id: int = None, max_tool_rounds: int = 6) -> str:
    """Main entrypoint for the chat route. Returns the assistant's final
    text reply, having called whatever tools it needed along the way."""
    db_context = _build_db_context(query, user_id=user_id)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT + f"\n\nDB_CONTEXT:\n{db_context}"),
        HumanMessage(content=query),
    ]

    llm = _llm()
    for _ in range(max_tool_rounds):
        try:
            ai_msg = llm.invoke(messages)
        except Exception as e:
            print(f"[rag_chat_agent] LLM invoke failed: {e}")
            try:
                fallback = ChatGroq(model=MODEL, temperature=0.3, max_tokens=800)
                return fallback.invoke(messages).content
            except Exception:
                return (
                    "I couldn't reach the AI service just now. Please try again "
                    "in a moment. This is AI-generated analysis, not financial advice."
                )

        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            return ai_msg.content

        for call in ai_msg.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(call["name"])
            if tool_fn is None:
                result = f"Unknown tool {call['name']}"
            else:
                try:
                    result = tool_fn.invoke(call["args"])
                except Exception as e:
                    # Tool errors are surfaced to the model as text, not raised —
                    # the system prompt tells it to fall back to DB_CONTEXT
                    # recommendations rather than dead-ending the answer.
                    result = f"Tool error: {e}. Fall back to DB_CONTEXT recommendations."
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Ran out of tool rounds (unusual) — force a final answer with what we have.
    final = llm.invoke(
        messages
        + [HumanMessage(
            content=(
                "Answer now with what you have so far. If live data is "
                "missing, use DB_CONTEXT recommendations and name a real "
                "stock — do not just say 'consult a financial advisor'."
            )
        )]
    )
    return final.content