from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random

app = Flask(__name__)
CORS(app)

pairs = []

def load_pairs():
    global pairs
    data_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "data.txt")
    data_path = os.path.normpath(data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    for i in range(len(lines) - 1):
        pairs.append((lines[i].lower(), lines[i + 1]))
    print(f"Loaded {len(pairs)} conversation pairs")

def find_best_response(message):
    message = message.lower().strip()
    msg_words = set(message.split())

    best_score = 0
    best_responses = []

    for prompt, response in pairs:
        prompt_words = set(prompt.split())
        if not prompt_words:
            continue
        overlap = len(msg_words & prompt_words)
        score = overlap / max(len(msg_words), len(prompt_words))
        if score > best_score:
            best_score = score
            best_responses = [response]
        elif score == best_score and best_score > 0:
            best_responses.append(response)

    if best_responses and best_score > 0.1:
        return random.choice(best_responses)

    fallbacks = [
        "i am not sure about that one tell me more",
        "that is interesting what else is on your mind",
        "yeah i get what you mean",
        "i hear you what do you think about it",
        "honestly i am still learning but i am here to chat",
    ]
    return random.choice(fallbacks)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400
    reply = find_best_response(message)
    return jsonify({"response": reply})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "pairs_loaded": len(pairs)
    })

if __name__ == "__main__":
    load_pairs()
    app.run(host="0.0.0.0", port=5000, debug=True)