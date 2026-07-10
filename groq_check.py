"""
benchmark_groq_latency.py — Latency benchmark for StockWise's Groq calls.

Run locally with GROQ_API_KEY set (same as your app). Measures wall-clock
latency of groq_client.chat() directly — this isolates model/network
latency from your Flask/DB overhead, which is the honest number to quote
("Groq API latency") vs. a conflated one ("my endpoint latency").

Tests 3 realistic prompt shapes pulled from your actual routes:
  - short   : ai_routes.py /explain/<symbol>   (~150 input tokens)
  - medium  : ai_routes.py /portfolio-insights (~300 input tokens)
  - agentic : rag_chat_agent.py-style system prompt + question (~900 input tokens)

For each, runs N calls and reports p50/p95/p99 latency + tokens/sec.

Usage:
    pip install groq --break-system-packages
    export GROQ_API_KEY=gsk_...
    python benchmark_groq_latency.py --n 20
"""
import argparse
import os
import statistics
import time

from dotenv import load_dotenv
# groq_client.py itself never calls load_dotenv() -- in the running app,
# db.py does that first and the vars stick around in the process. Running
# this script standalone skips that, so load .env explicitly here.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

from groq_client import chat, DEFAULT_MODEL

SHORT_PROMPT = [
    {"role": "system", "content": (
        "You are a financial explainer for a retail investing app. Be clear, "
        "honest about uncertainty, and never guarantee returns."
    )},
    {"role": "user", "content": (
        "Stock: TCS (Tata Consultancy Services)\n"
        "Current price: Rs.3850, Target: Rs.3990, Predicted gain: 3.6%\n"
        "Technical reason: strong upward momentum, RSI bullish (62)\n\n"
        "Write a 3-4 sentence, plain-English explanation of why this stock was "
        "flagged today, suitable for a retail user with no trading background. "
        "End with: 'Not financial advice.'"
    )},
]

MEDIUM_PROMPT = [
    {"role": "system", "content": (
        "You are a risk-analysis assistant for a paper-trading app. "
        "Output valid JSON only, no markdown fences."
    )},
    {"role": "user", "content": (
        "Portfolio positions:\n"
        "TCS qty=10 buy=Rs.3800 status=open pnl=None\n"
        "INFY qty=25 buy=Rs.1500 status=open pnl=None\n"
        "HDFCBANK qty=15 buy=Rs.1650 status=closed pnl=1200\n\n"
        "Summary: {'open_count': 2, 'closed_count': 1, 'invested': 75000, 'total_pnl': 1200}\n\n"
        "Respond ONLY as JSON with keys: risk_level (Low/Medium/High), "
        "diversification_comment (1 sentence), top_concern (1 sentence), "
        "suggestion (1 sentence, general framing only, not financial advice)."
    )},
]

AGENTIC_PROMPT = [
    {"role": "system", "content": (
        "You are StockWise AI, an expert Indian stock market assistant.\n"
        "You have TWO knowledge sources: DB_CONTEXT and LIVE DATA via tools.\n"
        "Never guess. Never invent prices. Never invent portfolio holdings.\n"
        "DB_CONTEXT:\n"
        "- AI recommendation rank 1 for TCS: current Rs.3850, target Rs.3990, gain 3.6%.\n"
        "- AI recommendation rank 2 for INFY: current Rs.1520, target Rs.1560, gain 2.6%.\n"
        "- User position: 10 shares of TCS bought at Rs.3800, status=open, current Rs.3850.\n"
        "- Portfolio summary: 1 open position, 0 closed, invested Rs.38000, total PnL Rs.0.\n"
    )},
    {"role": "user", "content": "Should I buy more TCS or diversify into something else?"},
]

PROMPT_SETS = {"short": SHORT_PROMPT, "medium": MEDIUM_PROMPT, "agentic": AGENTIC_PROMPT}


def percentile(data, pct):
    data = sorted(data)
    k = (len(data) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    return data[f] + (data[c] - data[f]) * (k - f)


def run_benchmark(name, messages, n, max_tokens=250):
    latencies = []
    tokens_per_sec = []
    errors = 0

    for i in range(n):
        start = time.perf_counter()
        try:
            reply = chat(messages, max_tokens=max_tokens)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            # rough token estimate: ~4 chars/token, good enough for tok/s ballpark
            approx_tokens = max(1, len(reply) // 4)
            tokens_per_sec.append(approx_tokens / elapsed)
            print(f"  [{name}] call {i+1}/{n}: {elapsed*1000:.0f}ms")
        except Exception as e:
            errors += 1
            print(f"  [{name}] call {i+1}/{n}: ERROR ({e})")

    if not latencies:
        print(f"  [{name}] all calls failed.")
        return None

    return {
        "name": name,
        "n": len(latencies),
        "errors": errors,
        "mean_ms": statistics.mean(latencies) * 1000,
        "p50_ms": percentile(latencies, 50) * 1000,
        "p95_ms": percentile(latencies, 95) * 1000,
        "p99_ms": percentile(latencies, 99) * 1000,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "avg_tokens_per_sec": statistics.mean(tokens_per_sec),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="calls per prompt shape")
    args = parser.parse_args()

    print(f"Model: {DEFAULT_MODEL}\nCalls per shape: {args.n}\n")

    results = []
    for name, messages in PROMPT_SETS.items():
        print(f"Running '{name}' prompt ({args.n} calls)...")
        r = run_benchmark(name, messages, args.n)
        if r:
            results.append(r)
        print()

    print("=" * 78)
    print(f"{'Prompt':<10}{'n':<5}{'err':<5}{'mean':<9}{'p50':<9}{'p95':<9}{'p99':<9}{'tok/s':<8}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<10}{r['n']:<5}{r['errors']:<5}"
              f"{r['mean_ms']:<9.0f}{r['p50_ms']:<9.0f}{r['p95_ms']:<9.0f}"
              f"{r['p99_ms']:<9.0f}{r['avg_tokens_per_sec']:<8.0f}")
    print("=" * 78)

    if results:
        overall_p95 = max(r["p95_ms"] for r in results)
        print("\nResume-ready line, e.g.:")
        print(f'  "Benchmarked {sum(r["n"] for r in results)} live Groq API calls '
              f'across 3 prompt shapes on Llama-3.3-70B: p50 latency of '
              f'{min(r["p50_ms"] for r in results):.0f}-{max(r["p50_ms"] for r in results):.0f}ms, '
              f'p95 under {overall_p95:.0f}ms, ~{statistics.mean([r["avg_tokens_per_sec"] for r in results]):.0f} tok/s throughput."')


if __name__ == "__main__":
    main()