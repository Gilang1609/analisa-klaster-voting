from flask import Flask, request, jsonify
from analyzer import analisa   # ⬅️  ganti impor ke 'analisa', bukan analisa_klaster

app = Flask(__name__)

@app.route("/analisis", methods=["POST"])
def analisis():
    data = request.get_json() or {}
    texts      = data.get("texts", [])
    embeddings = data.get("embeddings", [])

    hasil = analisa(texts, embeddings)  
    return jsonify(hasil)
