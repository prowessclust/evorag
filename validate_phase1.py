"""validate_phase1.py — quick sanity check for Phase 1 output."""
import json, glob, sys

files = sorted(glob.glob("data/chunks_*.json"))
if not files:
    print("[FAIL] No output file found in data/ — did fetcher.py run?")
    sys.exit(1)

chunks = json.load(open(files[-1], encoding="utf-8"))
if not chunks:
    print("[FAIL] Output file exists but chunk list is empty.")
    sys.exit(1)

print(f"[OK] Latest file : {files[-1]}")
print(f"[OK] Total chunks: {len(chunks)}")
print(f"[OK] Word range  : {min(c['word_count'] for c in chunks)}–{max(c['word_count'] for c in chunks)} words")

# Source breakdown
sources = {}
for c in chunks:
    sources[c["source"]] = sources.get(c["source"], 0) + 1
print("[OK] By source:")
for src, cnt in sorted(sources.items()):
    print(f"       {src:30s} {cnt:4d} chunks")

# Show first 5
print("\nFirst 5 chunks:")
for c in chunks[:5]:
    print(f"  [{c['chunk_index']}/{c['total_chunks']}] {c['source']:25s} | {c['word_count']} words | {c['title'][:55]}")

# Check for out-of-range word counts
bad = [c for c in chunks if not 250 <= c["word_count"] <= 500]
if bad:
    print(f"\n[WARN] {len(bad)} chunks outside 250-500 word range (may be edge cases).")
else:
    print(f"\n[OK] All chunks within acceptable word-count range.")

# Check required keys
required = {"chunk_id","source","url","url_hash","title","published_at","chunk_index","total_chunks","word_count","text"}
missing_keys = [c for c in chunks if required - set(c.keys())]
if missing_keys:
    print(f"[FAIL] {len(missing_keys)} chunks missing required keys!")
    sys.exit(1)
else:
    print("[OK] All required metadata keys present.")

print("\n=== Phase 1 PASSED ===")
