import os
import json
import argparse
import requests
import msgpack
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -------- Config --------
ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "motor-laws"
DIMENSION = 384
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data")

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# -------- PDF Loader --------
def load_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text


# -------- Text Chunking --------
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks


# -------- Endee API --------
def create_index():
    response = requests.post(
        f"{ENDEE_URL}/api/v1/index/create",
        json={
            "index_name": INDEX_NAME,
            "dim": DIMENSION,
            "space_type": "cosine"
        }
    )
    print("Create index status:", response.status_code)
    print("Create index response:", response.text)

def upload_batch(vectors_batch):
    """Upload a batch of vectors via msgpack (meta is preserved correctly)."""
    # HybridVectorObject MSGPACK_DEFINE order:
    #   id, meta, filter, norm, vector, sparse_ids, sparse_values
    packed_vectors = []
    for v in vectors_batch:
        meta_bytes = v["meta"].encode("utf-8") if isinstance(v["meta"], str) else v["meta"]
        packed_vectors.append([
            v["id"],        # id (string)
            meta_bytes,     # meta (bytes -> vector<uint8_t>)
            "",             # filter (string)
            0.0,            # norm (float)
            v["vector"],    # vector (list of floats)
            [],             # sparse_ids
            [],             # sparse_values
        ])

    payload = msgpack.packb(packed_vectors, use_bin_type=True)
    response = requests.post(
        f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert",
        data=payload,
        headers={"Content-Type": "application/msgpack"}
    )
    if response.status_code != 200:
        print(f"  Batch upload failed: {response.status_code} {response.text}")
        return False
    return True


# -------- Main Ingestion --------
def ingest(chunk_size=500, delete_first=False):
    if delete_first:
        print(f"Deleting existing index '{INDEX_NAME}'...")
        r = requests.delete(f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/delete")
        print(f"  Delete status: {r.status_code} {r.text}")

    create_index()

    print(f"Looking for PDFs in: {os.path.abspath(DATA_PATH)}")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data directory not found: {os.path.abspath(DATA_PATH)}")
        return

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        print("ERROR: No PDF files found in data directory.")
        return

    all_chunks = []
    all_ids = []
    vector_count = 0

    for file in pdf_files:
        path = os.path.join(DATA_PATH, file)
        print(f"Processing: {file}")

        text = load_pdf(path)
        chunks = chunk_text(text, chunk_size=chunk_size)
        print(f"  -> {len(chunks)} chunks (chunk_size={chunk_size})")

        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"{file}_{vector_count}")
            vector_count += 1

    if not all_chunks:
        print("ERROR: No text extracted from PDFs.")
        return

    # Batch encode all chunks at once (much faster than one-by-one)
    print(f"\nEncoding {len(all_chunks)} chunks in batch...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=32)
    print("Encoding complete.")

    # Upload in batches of 50 vectors
    BATCH_SIZE = 50
    success_count = 0

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(all_chunks))
        batch = []
        for j in range(i, batch_end):
            batch.append({
                "id": all_ids[j],
                "vector": embeddings[j].tolist(),
                "meta": json.dumps({"text": all_chunks[j]})
            })

        if upload_batch(batch):
            success_count += len(batch)
        print(f"  Uploaded {min(batch_end, len(all_chunks))}/{len(all_chunks)} vectors...")

    print(f"\nIngestion complete. {success_count}/{vector_count} vectors stored successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into Endee vector database")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Characters per chunk (default: 500). Try 200, 500, 1000 to experiment.")
    parser.add_argument("--delete-index", action="store_true",
                        help="Delete existing index before re-ingesting")
    args = parser.parse_args()

    print(f"\n=== Ingesting with chunk_size={args.chunk_size} ===")
    ingest(chunk_size=args.chunk_size, delete_first=args.delete_index)