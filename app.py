from flask import Flask, request, jsonify
from analyzer import analisa

app = Flask(__name__)

@app.route("/analisa", methods=["POST"])
def route_analisa():
    data = request.get_json() or {}
    result = analisa(data.get("texts", []), data.get("embeddings", []))
    return jsonify(result)

# if __name__ == "__main__":
#     app.run()
