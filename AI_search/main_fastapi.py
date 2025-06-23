from flask import Flask, request, jsonify
from notice_search import search_similar_notices  # 기존 코드 import

app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()

    if not data or "user_input" not in data:
        return jsonify({"error": "user_input가 필요합니다."}), 400

    user_input = data["user_input"]
    results = search_similar_notices(user_input)

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
