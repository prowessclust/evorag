"""
validate_phase3.py — EvoRAG Phase 3 sanity check.

Starts the API server as a subprocess, runs tests against it, then shuts it down.

Run:
    python validate_phase3.py

Prerequisites:
    - FAISS index built (python embedder.py --build)
    - Ollama running with phi3:latest (ollama serve)

Exit code 0 = PASSED, 1 = FAILED.
"""

import subprocess
import sys
import time
import httpx

API_URL = "http://127.0.0.1:8000"
STARTUP_TIMEOUT = 60   # max seconds to wait for server to become ready
POLL_INTERVAL   = 2    # seconds between health-check retries


def fail(msg: str):
    print(f"\n[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str):
    print(f"[OK]   {msg}")


print("\n" + "=" * 60)
print("  EvoRAG Phase 3 — Validation")
print("=" * 60)

# ── Step 1: Start the API server + wait for it to be ready ───────────────────
print("\n[...] Starting API server (python api.py)…")
server = subprocess.Popen(
    [sys.executable, "api.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

# Poll /health until the server responds or we hit STARTUP_TIMEOUT
print(f"[...] Waiting for server to be ready (up to {STARTUP_TIMEOUT}s)…")
deadline = time.time() + STARTUP_TIMEOUT
server_ready = False
while time.time() < deadline:
    # Check if the subprocess died unexpectedly
    if server.poll() is not None:
        out, _ = server.communicate()
        fail(f"API server exited early:\n{out}")
    try:
        with httpx.Client(timeout=3) as client:
            r = client.get(f"{API_URL}/health")
        if r.status_code == 200:
            server_ready = True
            break
    except httpx.ConnectError:
        pass   # server not up yet — keep retrying
    time.sleep(POLL_INTERVAL)

if not server_ready:
    server.terminate()
    fail(f"Server did not become ready within {STARTUP_TIMEOUT}s.")

ok(f"Server is ready (PID {server.pid})")

try:
    # ── Step 2: GET /health ───────────────────────────────────────────────────
    print("\n[...] GET /health")
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_URL}/health")

    if resp.status_code != 200:
        fail(f"/health returned HTTP {resp.status_code}: {resp.text}")

    health = resp.json()
    if health.get("status") != "ok":
        fail(f"/health status is not 'ok': {health}")

    n_vectors = health.get("index_vectors", 0)
    model_name = health.get("model", "?")
    ok(f"Health check passed — {n_vectors} vectors in index, model: {model_name}")

    # ── Step 3: POST /query ───────────────────────────────────────────────────
    print("\n[...] POST /query — 'What are the latest developments in artificial intelligence?'")
    print("      (This may take 15–60 seconds while phi3 generates the answer…)\n")

    with httpx.Client(timeout=300) as client:
        resp = client.post(
            f"{API_URL}/query",
            json={"query": "What are the latest developments in artificial intelligence?", "top_k": 5},
        )

    if resp.status_code != 200:
        fail(f"/query returned HTTP {resp.status_code}: {resp.text}")

    data = resp.json()

    # Validate schema
    if "answer" not in data:
        fail(f"Response missing 'answer' key: {data}")
    if "sources" not in data:
        fail(f"Response missing 'sources' key: {data}")
    if not data["answer"].strip():
        fail("'answer' field is empty.")
    if not isinstance(data["sources"], list):
        fail("'sources' is not a list.")

    ok(f"POST /query returned answer ({len(data['answer'])} chars)")
    ok(f"Sources returned: {len(data['sources'])}")

    print("\n  --- Answer (first 400 chars) ---")
    print(f"  {data['answer'][:400].strip()}")
    print()

    print("  --- Sources ---")
    for i, s in enumerate(data["sources"]):
        score = s.get("score", "?")
        title = (s.get("title") or "(no title)")[:60]
        source = s.get("source", "?")
        print(f"  [{i}] score={score} | {source} | {title}")

    # ── Step 4: Validate source schema ────────────────────────────────────────
    required_source_keys = {"title", "url", "source", "score"}
    if data["sources"]:
        missing = required_source_keys - set(data["sources"][0].keys())
        if missing:
            fail(f"Source dict missing keys: {missing}")
    ok("Source schema is correct.")

    # ── Step 5: Second query for generalization ───────────────────────────────
    print("\n[...] POST /query — 'What is happening with global markets?'")
    with httpx.Client(timeout=300) as client:
        resp2 = client.post(
            f"{API_URL}/query",
            json={"query": "What is happening with global markets?", "top_k": 3},
        )
    if resp2.status_code != 200:
        fail(f"Second /query returned HTTP {resp2.status_code}: {resp2.text}")
    data2 = resp2.json()
    if not data2.get("answer", "").strip():
        fail("Second query returned empty answer.")
    ok(f"Second query answered ({len(data2['answer'])} chars).")

finally:
    # ── Shut down the server ─────────────────────────────────────────────────
    print("\n[...] Stopping API server…")
    server.terminate()
    server.wait(timeout=5)
    ok("Server stopped cleanly.")

print("\n" + "=" * 60)
print("  === Phase 3 PASSED ===")
print("=" * 60 + "\n")
