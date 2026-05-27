from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq
import pypdf
import numpy as np
import io
import os
import re

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Simple in-memory store
chunks = []

def simple_embed(text):
    """Lightweight keyword-based embedding using character trigrams"""
    text = text.lower()
    trigrams = [text[i:i+3] for i in range(len(text)-2)]
    vocab = {}
    for t in trigrams:
        vocab[t] = vocab.get(t, 0) + 1
    return vocab

def cosine_sim(a, b):
    """Cosine similarity between two dicts"""
    keys = set(a.keys()) & set(b.keys())
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    mag_a = sum(v*v for v in a.values()) ** 0.5
    mag_b = sum(v*v for v in b.values()) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global chunks
    content = await file.read()
    reader = pypdf.PdfReader(io.BytesIO(content))
    text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    # Split into chunks
    raw = [text[i:i+500] for i in range(0, len(text), 400)]
    chunks = [{"text": c, "embed": simple_embed(c)} for c in raw if len(c) > 50]
    return {"message": f"Uploaded {file.filename}", "chunks": len(chunks)}

@app.post("/ask")
async def ask(data: dict):
    global chunks
    question = data.get("question", "")
    if not chunks:
        return {"answer": "Please upload a document first.", "context": ""}
    
    q_embed = simple_embed(question.lower())
    scored = [(cosine_sim(q_embed, c["embed"]), c["text"]) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [t for _, t in scored[:3]]
    context = "\n\n".join(top)
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer questions based only on the provided context. Be clear and concise. If the answer is not in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return {"answer": response.choices[0].message.content, "context": context[:300]}

@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(chunks)}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
