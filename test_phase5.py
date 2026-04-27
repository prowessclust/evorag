import httpx
import json
import time

API_URL = "http://127.0.0.1:8000"

def run_test(query):
    print("=" * 70)
    print(f"  EvoRAG Phase 5 — Query: {query}")
    print("=" * 70)
    print("Sending query (Generating 8 responses simultaneously + 1 Judge evaluation)...")
    print("Note: This involves only 2 LLM calls total (Optimized for fast response).\n")

    start_time = time.time()
    try:
        with httpx.Client(timeout=None) as client:
            response = client.post(
                f"{API_URL}/query",
                json={
                    "query": query,
                    "top_k": 1
                }
            )
    except Exception as e:
        print(f"Error reaching the API server: {e}")
        return

    elapsed = round(time.time() - start_time, 2)
    if response.status_code != 200:
        print(f"Server returned an error ({response.status_code}): {response.text}")
        return

    data = response.json()
    print(f"[OK] Response received in {elapsed} seconds!\n")

    print("-" * 30)
    print(f"[*] BEST RESPONSE: {data['best_response']['persona'].upper()}")
    print("-" * 30)
    print(f"{data['best_response']['answer'][:500]}...")
    print("\n" + "-" * 30)

    print("\n[#] VOTE BREAKDOWN (Aggregate Scores):")
    # Calculate totals for display
    totals = {}
    for entry in data['vote_breakdown']:
        for target, score in entry['scores'].items():
            totals[target] = totals.get(target, 0) + score
    
    # Sort by score
    sorted_totals = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    for persona, score in sorted_totals:
        star = ">>" if persona == data['best_response']['persona'] else "  "
        print(f"{star} {persona:25s} : {score:5.1f} pts")

    print("\nSources used:")
    for i, s in enumerate(data['sources'], 1):
        print(f"[{i}] {s['source']} - {s['title'][:70]}")

if __name__ == "__main__":
    print("\n\n[>>> STARTING QUERY <<<]\n")
    run_test("What are the pros and cons of online education?")

