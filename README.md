# RAG Document Q&A Bot

A Retrieval-Augmented Generation (RAG) based Q&A bot that allows users to ask natural language questions against a collection of documents and receive accurate, grounded answers with clear source citations. The bot ingests PDF documents, stores them in a vector database, and uses Google Gemini to generate answers strictly based on the retrieved content.


## Tech Stack

| Library / Tool | Version |
|---|---|
| Python | 3.11 |
| langchain | 0.3.x |
| langchain-community | 0.3.x |
| langchain-google-genai | latest |
| langchain-chroma | latest |
| langchain-text-splitters | 0.3.x |
| langchain-core | latest |
| chromadb | latest |
| google-generativeai | latest |
| streamlit | latest |
| pypdf | 5.9.0 |
| python-dotenv | 1.2.2 |
| sentence-transformers | latest |


## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        INGESTION                            │
│                                                             │
│  PDF Files  →  PyPDFLoader  →  RecursiveCharacterSplitter   │
│                                        ↓                    │
│                              Gemini Embeddings              │
│                                        ↓                    │
│                            ChromaDB (persisted to disk)     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        RETRIEVAL                            │
│                                                             │
│  User Question  →  Gemini Embeddings  →  Similarity Search  │
│                                                ↓            │
│                                     Top-K Relevant Chunks   │
│                                                ↓            │
│                                      Gemini LLM (Flash)     │
│                                                ↓            │
│                              Grounded Answer + Citations     │
└─────────────────────────────────────────────────────────────┘
```

**Pipeline Steps:**
1. **Ingestion** — PDFs are loaded using `PyPDFLoader`, split into chunks, embedded using Google Gemini, and stored in ChromaDB
2. **Retrieval** — User query is embedded and compared against stored vectors using cosine similarity
3. **Generation** — Top-K relevant chunks are passed as context to Gemini LLM which generates a grounded answer


## Chunking Strategy

**Strategy used:** `RecursiveCharacterTextSplitter`

- **Chunk size:** 500 characters
- **Chunk overlap:** 100 characters

**Why this strategy?**
`RecursiveCharacterTextSplitter` splits text by trying multiple separators in order — paragraphs, sentences, words — ensuring chunks break at natural language boundaries rather than mid-sentence. The 100-character overlap ensures that context is not lost at chunk boundaries, which is critical for questions that span across paragraphs.


## Embedding Model and Vector Database

**Embedding Model:** Google Gemini `models/gemini-embedding-001`

- Chosen for its high semantic accuracy and native integration with the Gemini ecosystem
- Produces dense vector representations that capture meaning, not just keywords

**Vector Database:** ChromaDB

- Chosen for its simplicity — no external server required
- Persists embeddings to disk so documents do not need to be re-indexed on every run
- Provides fast similarity search out of the box
- Ideal for local development and small-to-medium document collections


## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/gayathri0153-code/RAG-qa-bot.git
cd RAG-qa-bot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root of the project:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get your API key from: https://aistudio.google.com

### 5. Add documents
Place your PDF files in the `/data` folder. The bot comes with 4 pre-loaded documents.

### 6. Run ingestion (index documents into vector database)
```bash
python src/ingest.py
```
> Note: Due to Gemini free tier rate limits (100 requests/min), ingestion runs in batches with 65-second pauses between batches. For ~575 chunks this takes approximately 8 minutes.

### 7. Run the Streamlit web UI
```bash
streamlit run src/app.py
```
Opens at: http://localhost:8501

### 8. Or run via command line
```bash
python src/retriever.py
```

---

## Environment Variables

| Variable | Description | How to get |
|---|---|---|
| `GOOGLE_API_KEY` | Google Gemini API key for embeddings and LLM | https://aistudio.google.com → Get API Key |

**Important:** Never commit your `.env` file. It is listed in `.gitignore`. Always use environment variables for API keys.

---

## Example Queries

| # | Question | Expected Answer Theme |
|---|---|---|
| 1 | What are the applications of AI in healthcare? | Diagnosis, treatment planning, virtual health assistants, workflow optimization |
| 2 | What are common cybersecurity threats? | Malware, phishing, ransomware, data breaches, social engineering |
| 3 | What are the principles of generative AI? | Core GenAI guidelines, responsible use, capabilities and limitations |
| 4 | What does Scientific Culturals discuss? | Overview of scientific and cultural topics covered in the document |
| 5 | How does AI relate to cybersecurity? | Cross-document answer linking AI-powered threat detection and security automation |
| 6 | Who won the 2024 FIFA World Cup? | "I don't have enough information in my documents to answer this." |

---

## Project Structure

```
RAG-qa-bot/
├── data/                          # Document knowledge base
│   ├── AI in healthcare.pdf
│   ├── cybersecurity.pdf
│   ├── genai-principles.pdf
│   └── Scientific Culturals.pdf
├── src/
│   ├── ingest.py                  # Document ingestion pipeline
│   ├── retriever.py               # CLI-based Q&A interface
│   └── app.py                     # Streamlit web UI
├── chroma_store/                  # Persisted vector database (auto-generated)
├── .env                           # API keys (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Known Limitations

- **API Rate Limits** — Gemini free tier allows only 100 embedding requests/minute and 1000/day. Large document sets require batched ingestion with delays
- **Context Window** — Only top-4 chunks are passed to the LLM. Questions requiring information spread across many pages may get incomplete answers
- **PDF Quality** — Scanned PDFs or image-based PDFs without OCR will not be extracted correctly
- **No Memory** — The bot does not retain conversation history. Each question is answered independently without context from previous questions
- **Hallucination Risk** — If retrieved chunks are partially relevant, the LLM may occasionally generate answers that go slightly beyond the source material
- **Single Language** — Optimized for English documents only

---
