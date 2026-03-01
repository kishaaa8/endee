import json
import os
import re
import requests
import msgpack



from dotenv import load_dotenv
load_dotenv()

# Use cached model - avoid network timeouts
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "motor-laws"
RELEVANCE_THRESHOLD = 0.30  # below this score, query is off-topic

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_meta(meta_raw):
    """Decode Endee meta bytes → dict or plain string."""
    if isinstance(meta_raw, bytes):
        meta_text = meta_raw.decode("utf-8", errors="replace")
    elif isinstance(meta_raw, list):
        meta_text = bytes(meta_raw).decode("utf-8", errors="replace")
    else:
        meta_text = str(meta_raw)
    try:
        meta = json.loads(meta_text)
        return meta.get("text", meta_text)
    except (json.JSONDecodeError, TypeError):
        return meta_text


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(query, top_k=2):
    """Search Endee and return concatenated context text.

    Returns empty string if best score is below RELEVANCE_THRESHOLD.
    """
    query_vector = model.encode(query).tolist()

    response = requests.post(
        f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search",
        json={"vector": query_vector, "k": top_k},
    )

    if response.status_code != 200:
        print(f"Search failed: {response.status_code} {response.text}")
        return ""

    results = msgpack.unpackb(response.content, raw=False)

    # Check if best result is relevant enough
    if not results or results[0][0] < RELEVANCE_THRESHOLD:
        return ""

    contexts = []
    for r in results:
        # [similarity, id, meta, filter, norm, vector]
        if r[0] < RELEVANCE_THRESHOLD:
            continue
        text = _decode_meta(r[2])
        if text:
            contexts.append(text)
    return "\n\n".join(contexts)


def retrieve_results(query, top_k=2):
    """Search Endee and return list of dicts with score + text."""
    query_vector = model.encode(query).tolist()

    response = requests.post(
        f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search",
        json={"vector": query_vector, "k": top_k},
    )

    if response.status_code != 200:
        print(f"Search failed: {response.status_code} {response.text}")
        return []

    results = msgpack.unpackb(response.content, raw=False)
    out = []
    for r in results:
        out.append({
            "score": r[0],
            "id": r[1],
            "text": _decode_meta(r[2]),
        })
    return out


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def _clean_legal_text(text: str) -> str:
    """Strip legal formatting noise so the text is easier to read."""
    # Remove repeated dashes/underscores used as separators
    text = re.sub(r'[-_]{3,}', '', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove standalone section/clause markers like "(a)" at start of line
    text = re.sub(r'(?m)^\s*\([a-z]\)\s*', '  - ', text)
    # Simplify "sub-section (1) of section" → "Section"
    text = re.sub(r'sub-section\s*\(\d+\)\s*of\s*section', 'Section', text, flags=re.I)
    # Replace "shall be punishable with" → "can be punished with"
    text = re.sub(r'shall be punishable with', 'can be punished with', text, flags=re.I)
    # Replace "notwithstanding anything contained" → "regardless of other rules"
    text = re.sub(r'notwithstanding anything contained\s*(in\s*)?', 'Regardless of other rules in ', text, flags=re.I)
    # Replace "thereof" "therein" "thereto"
    text = re.sub(r'\bthereof\b', 'of it', text, flags=re.I)
    text = re.sub(r'\btherein\b', 'in it', text, flags=re.I)
    text = re.sub(r'\bthereto\b', 'to it', text, flags=re.I)
    return text.strip()


def _extract_sections(text: str) -> list:
    """Pull out Section numbers mentioned in the text."""
    return re.findall(r'Section\s+\d+[A-Za-z]*', text, re.I)


OFF_TOPIC_MSG = ("This question doesn't seem to be related to Indian Motor Vehicle law. "
                 "I can help with topics like driving licences, traffic penalties, "
                 "vehicle registration, insurance, road safety rules, etc.\n\n"
                 "Try asking something like:\n"
                 "  • What is the penalty for drunk driving?\n"
                 "  • What are the rules for getting a driving licence?\n"
                 "  • What happens if I drive without insurance?")


def _build_local_summary(context: str, question: str = "") -> str:
    """Offline fallback: simplify the law text into a readable answer."""
    if not context or not context.strip():
        return OFF_TOPIC_MSG

    cleaned = _clean_legal_text(context)

    # Split into logical paragraphs
    paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]

    # Extract which sections are referenced
    sections = list(dict.fromkeys(_extract_sections(cleaned)))  # unique, ordered

    # Build the answer
    answer = ""

    if question:
        answer += f"**Your question:** {question}\n\n"

    answer += "**Here's what the law says (in simple terms):**\n\n"

    for i, para in enumerate(paragraphs):
        # Clean up each paragraph — remove excess whitespace
        para = re.sub(r'\s+', ' ', para).strip()
        if len(para) < 20:
            continue
        # Truncate very long paragraphs
        if len(para) > 400:
            # Try to cut at a sentence boundary
            cut = para[:400].rfind('.')
            if cut > 200:
                para = para[:cut + 1]
            else:
                para = para[:400] + "..."
        answer += f"{para}\n\n"

    if sections:
        answer += "**Sections referenced:** " + ", ".join(sections[:6]) + "\n\n"

    answer += ("_Note: For a more detailed plain-English explanation, "
               "set your OpenAI API key (OPENAI_API_KEY environment variable)._")

    return answer


def generate_answer(question, challan_text=""):
    """Generate an answer using OpenAI (or offline fallback).

    Returns (answer_text, source_text) where source_text is the raw law
    excerpt so users can verify the summary against the original.
    """
    context = retrieve_context(question)

    # ---- Off-topic check ----
    if not context or not context.strip():
        return OFF_TOPIC_MSG, ""

    # ---- Build prompt ----
    challan_block = ""
    if challan_text and challan_text.strip():
        challan_block = f"\nUser's Traffic Challan:\n{challan_text}\n"

    prompt = f"""You are a legal assistant specialising in Indian Motor Vehicle law.
{challan_block}
Relevant Motor Vehicle Laws:
{context}

Question:
{question}

Explain in simple and practical language. Cite specific sections where possible."""


    # ---- Try OpenRouter ----
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key:
        try:
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Motor Vehicle Legal Assistant"
            }

            data = {
                "model": "qwen/qwen3-vl-235b-a22b-thinking",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                return answer, context
            else:
                print("OpenRouter error:", response.text)

        except Exception as e:
            print(f"OpenRouter exception: {e}")

    # ---- Offline fallback ----
    if not context or not context.strip():
        return OFF_TOPIC_MSG, ""

    return _build_local_summary(context, question), context
    
    # gemini_key = os.getenv("GEMINI_API_KEY", "")
    # if gemini_key:
    #     try:
    #         genai.configure(api_key=gemini_key)
    #     except Exception:
    #         print("Failed to configure Gemini API key.")
    #         return OFF_TOPIC_MSG, context

    #     gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    #     response = gemini_model.generate_content(prompt)
    #     return response.text, context

    #     except Exception as e:
    #         print(f"Gemini error: {e}")

    # # ---- Offline fallback ----
    # if not context or not context.strip():
    #     return OFF_TOPIC_MSG, ""
    # return _build_local_summary(context, question), context


# ---------------------------------------------------------------------------
# Interactive mode  (python -m scripts.rag_pipeline)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Motor Vehicle Legal Assistant")
    print("Type 'quit' to exit.\n")
    while True:
        q = input("Your question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break

        results = retrieve_results(q)
        if not results:
            print("No results found.\n")
            continue

        print(f"\n{'='*60}")
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i}  |  Score: {r['score']:.4f}  |  ID: {r['id']} ---")
            print(r["text"][:500])
        print(f"\n{'='*60}")

        answer, sources = generate_answer(q)
        print(f"\nAnswer:\n{answer}\n")
        print(f"--- Source (original law text) ---\n{sources[:600]}\n")