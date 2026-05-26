from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import pypdf
import io
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

genai.configure(api_key=os.environ.get("AIzaSyDOktkNaDxVKcPdyhSUMTShV6P5FZcRrL8"))
model = genai.GenerativeModel("gemini-1.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("docs")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    reader = pypdf.PdfReader(io.BytesIO(content))
    text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = embedder.encode(chunks).tolist()
    collection.add(documents=chunks, embeddings=embeddings, ids=[f"chunk_{i}" for i in range(len(chunks))])
    return {"message": f"Uploaded {file.filename}", "chunks": len(chunks)}

@app.post("/ask")
async def ask(data: dict):
    question = data.get("question", "")
    q_embedding = embedder.encode([question]).tolist()[0]
    results = collection.query(query_embeddings=[q_embedding], n_results=3)
    context = "\n".join(results["documents"][0])
    prompt = f"Based on this context:\n{context}\n\nAnswer this question: {question}"
    response = model.generate_content(prompt)
    return {"answer": response.text, "context": context[:200]}

@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
