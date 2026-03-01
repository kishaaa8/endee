# Motor Vehicle Legal Assistant (RAG using Endee)

## Project Overview

Understanding traffic laws and legal notices can be difficult for common users because legal documents are long, complex, and written in formal language. People receiving traffic challans or looking for rule clarifications often need simple, practical explanations rather than raw legal text.

This project builds an **AI-powered Motor Vehicle Legal Assistant** that:

- Stores Motor Vehicles Act content in a vector database
- Understands user questions using semantic search
- Retrieves the most relevant law sections
- Explains them in simple language using an LLM

The system uses **Endee** as the core vector database and follows a **Retrieval Augmented Generation (RAG)** architecture.

---

## Problem Statement

Legal documents are:

- Lengthy and difficult to navigate
- Written in complex legal terminology
- Hard to search using keyword-based methods

Users need a system that can:

- Understand natural language questions
- Retrieve relevant legal sections intelligently
- Provide clear, easy-to-understand explanations

This project solves the problem using **vector search + AI-based explanation**.

---

## Key Features

- Semantic search over Motor Vehicles Act
- RAG-based question answering
- Context-aware legal explanations
- Off-topic query detection using a relevance threshold
- Offline fallback if LLM is unavailable
- Fast similarity search using Endee

---

## System Architecture

```
Motor Vehicles PDF
        ↓
Text Extraction
        ↓
Chunking
        ↓
Embeddings (MiniLM)
        ↓
Stored in Endee (Vector DB)
        ↓
User Question
        ↓
Query Embedding
        ↓
Endee Semantic Search
        ↓
Relevant Law Context
        ↓
Qwen LLM (Open Router)
        ↓
Simple Explanation
```

---

## Technical Approach

### 1. Data Ingestion

- PDF processed using `pypdf`
- Text split into manageable chunks
- Each chunk converted into embeddings using:

`sentence-transformers/all-MiniLM-L6-v2`
Dimension: **384**

---

### 2. Vector Storage (Endee)

- Index Name: `motor-laws`
- Dimension: 384
- Similarity Metric: Cosine
- Metadata stores original law text
- Vectors inserted in batches using **MsgPack** for performance

---

### 3. Retrieval

- User query converted into embedding
- Endee returns top-k most similar chunks
- Low-similarity results filtered using a relevance threshold

---

### 4. Generation (RAG)

- Retrieved context + user question sent to **QWEN**
- Qwen generates a plain-English explanation
- If API is unavailable → local summarization fallback

---

## How Endee is Used

Endee acts as the **knowledge engine** of the system.

### Index Creation

**Endpoint**

```
POST /api/v1/index/create
```

Parameters:

- name: motor-laws
- dim: 384
- space_type: cosine

---

### Vector Insertion

**Endpoint**

```
POST /api/v1/index/motor-laws/vector/insert
```

Each stored vector contains:

- id
- embedding vector
- meta (original law text)

---

### Semantic Search

**Endpoint**

```
POST /api/v1/index/motor-laws/search
```

Returns:

- similarity score
- matched text chunks
- metadata

Endee enables fast and scalable semantic retrieval for the RAG pipeline.

---

## Project Structure

```
vehicle_legal_assistant/
│
├── data/
│   └── motor_vehicles_act.pdf
│
├── scripts/
│   ├── ingest.py          # Loads PDF and stores vectors in Endee
│   ├── rag_pipeline.py    # Main RAG pipeline (retrieval + Qwen)
│   └── test_search.py     # Retrieval testing script
│
├── .env                   # API keys
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```
git clone <your-repo-link>
cd vehicle_legal_assistant
```

---

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate   (Windows)
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Start Endee Server

From project root:

```
docker compose up -d
```

Check server status:

```
http://localhost:8080/api/v1/health
```

---

### 5. Add Open Router API Key

Create a `.env` file:

```
OPENROUTER_API_KEY=your_api_key_here
```

---

### 6. Ingest Data into Endee

```
python scripts/ingest.py
```

This step:

- Creates index
- Processes PDF
- Stores embeddings

---

### 7. Run the Assistant

```
python scripts/rag_pipeline.py
```

Example query:

```
What is the penalty for drunk driving?
```

---

## Example Use Cases

- Fine for no helmet
- Drunk driving penalty
- Driving without insurance
- Licence suspension rules

---

## Technologies Used

- Python
- Endee (Vector Database)
- Sentence Transformers
- QWEN: Qwen3 VL 235B A22B Thinking(OPEN ROUTER)
- Docker
- Requests & MsgPack
- pypdf

---

## Future Improvements

- Web interface (Streamlit / React)
- OCR support for challan images
- Support for multiple legal datasets
- Multilingual support
- Case-specific legal guidance

---

## License

This project is created for educational and evaluation purposes as part of the Endee Internship project submission.
