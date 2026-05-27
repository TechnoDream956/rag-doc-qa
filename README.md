# DocuMind — RAG Document Q&A

Upload any PDF and ask questions using Google Gemini + ChromaDB + FastAPI.

## Features

- 📄 Upload any PDF document
- 🔍 Semantic search using ChromaDB vector database
- 🤖 LLaMA 3.1 powered answers via Groq API
- ⚡ Fast responses — no hallucinations, grounded in your document
- 🌐 REST API with FastAPI

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python · FastAPI · Uvicorn |
| Vector DB | ChromaDB |
| LLM | LLaMA 3.1 via Groq API |
| PDF Parsing | pypdf |
| Hosting | Render |

## API Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/upload` | Upload and process a PDF |
| POST | `/ask` | Ask a question about the document |
| GET | `/health` | Health check |

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

## Author

**Arpit Sehrawat** — [@TechnoDream956](https://github.com/TechnoDream956)
Bennett University | Cloud & Backend Developer
EOF
