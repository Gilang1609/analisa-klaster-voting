from flask import Flask, request, jsonify
from analyzer import analisa_klaster

app = Flask(__name__)

@app.route("/analisis", methods=["POST"])
def analisis():
    data = request.get_json()
    result = analisa_klaster(data)
    return jsonify(result)
