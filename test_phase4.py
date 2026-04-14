import httpx
import json
import time

API_URL = "http://127.0.0.1:8000"

print("=" * 60)
print("  EvoRAG Phase 4 — Multi-Persona Testing")
print("=" * 60)
print("Sending query to your 8 parallel personas... (this might take a few moments)\n")

# Disable timeouts completely to give Ollama time to execute 8 threads on local hardware
start_time = time.time()
try:
    with httpx.Client(timeout=None) as client:
        response = client.post(
            f"{API_URL}/query",
            json={
                "query": "What are the latest developments in artificial intelligence?",
                "top_k": 2
            }
        )
except Exception as e:
    print(f"Error reaching the API server: {e}")
    exit(1)

elapsed = round(time.time() - start_time, 2)
if response.status_code != 200:
    print(f"Server returned an error: {response.text}")
    exit(1)

data = response.json()
print(f"[OK] Response received in {elapsed} seconds!\n")
print(f"Query: {data['query']}\n")

# Print out the results for each persona formatting
print("--- 8 PERSONA RESPONSES ---")
for r in data["responses"]:
    print(f"\n>> Persona: {r['persona'].upper()}")
    print(f"{r['answer']}")

print("\n--- SOURCES USED ---")
for i, s in enumerate(data["sources"], 1):
    print(f"[{i}] {s['source']} - {s['title'][:70]}...")
