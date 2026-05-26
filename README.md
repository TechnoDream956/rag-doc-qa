# DocuMind — RAG Document Q&A

Upload any PDF and ask questions using Google Gemini + ChromaDB + FastAPI.

## Stack
- FastAPI + Python
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- Google Gemini 1.5 Flash (LLM)

## How it works
1. Upload PDF → split into 500-char chunks
2. Generate embeddings with sentence-transformers
3. Store in ChromaDB vector database
4. On question → find similar chunks → send to Gemini → return answer

## Author
Arpit Sehrawat | @TechnoDream956
