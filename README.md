# DocuMind — RAG Document Q&A

Upload any PDF and ask questions using Groq LLaMA 3.1 + FastAPI.

---

## Features

- 📄 Upload any PDF document
- 🔍 Custom semantic retrieval using trigram similarity search
- 🧠 LLaMA 3.1 powered answers via Groq API
- ⚡ Fast responses grounded in your document
- 🎨 Modern responsive AI interface
- 📊 Structured markdown answers with tables & formatting
- 🌐 REST API built with FastAPI

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python · FastAPI · Uvicorn |
| Retrieval | Custom Trigram Similarity Search |
| LLM | LLaMA 3.1 via Groq API |
| PDF Parsing | pypdf |
| Frontend | HTML · CSS · Vanilla JavaScript |
| Hosting | Render |

---

## Architecture

```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunk Generation
    ↓
Trigram Embedding
    ↓
Cosine Similarity Retrieval
    ↓
Groq LLaMA 3.1 Response
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/upload` | Upload and process a PDF |
| POST | `/ask` | Ask a question about the document |
| GET | `/health` | Health check |

---

## Setup

```bash
git clone https://github.com/TechnoDream956/rag-doc-qa.git
cd rag-doc-qa

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

export GROQ_API_KEY="your_key_here"

uvicorn app.main:app --reload --port 8000
```

---

## Live Demo

https://rag-doc-qa-4udm.onrender.com

> Note: Initial load may take a few seconds because the backend is hosted on Render free tier.

---

## Author

Arpit Sehrawat — @TechnoDream956

Bennett University | Cloud , AI & Backend Developer
