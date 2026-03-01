import os
import requests
import msgpack
import json
import sys

# Use cached model only - don't try to download/check HuggingFace Hub
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "motor-laws"

# Accept query from command line or prompt
if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])
else:
    query = input("Enter your question: ")

print(f"\nQuery: {query}")
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(query).tolist()

print("Searching...")
response = requests.post(
    f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search",
    json={
        "vector": embedding,
        "k": 5
    }
)

print("Status code:", response.status_code)

if response.status_code == 200:
    # Endee returns msgpack array of arrays: [similarity, id, meta_bytes, filter, norm, vector]
    results = msgpack.unpackb(response.content, raw=False)
    print(f"\nFound {len(results)} results:\n")

    for i, r in enumerate(results):
        # Each result is a list: [similarity, id, meta, filter, norm, vector]
        similarity = r[0]
        vec_id = r[1]
        meta_raw = r[2]

        # Decode meta - C++ vector<uint8_t> can arrive as bytes, list of ints, or string
        if isinstance(meta_raw, bytes):
            meta_text = meta_raw.decode("utf-8", errors="replace")
        elif isinstance(meta_raw, list):
            meta_text = bytes(meta_raw).decode("utf-8", errors="replace")
        else:
            meta_text = str(meta_raw)

        # Parse the JSON meta to extract text
        try:
            meta = json.loads(meta_text)
            text = meta.get("text", meta_text)
        except (json.JSONDecodeError, TypeError):
            text = meta_text

        print(f"--- Result {i+1}  |  Score: {similarity:.4f}  |  ID: {vec_id} ---")
        print(text[:500])
        print()
else:
    print("Error:", response.text)