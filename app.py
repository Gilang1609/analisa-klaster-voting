from fastapi import FastAPI, Request
from analyzer import run_clustering  # fungsi ini harus ada di analyzer.py
import uvicorn

app = FastAPI()

@app.post("/analisis")
async def analisis(request: Request):
    data = await request.json()
    texts = data.get("texts", [])
    embeddings = data.get("embeddings", [])
    result = run_clustering(texts, embeddings)
    return result
