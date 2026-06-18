from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model import GPTLike

app = Flask(__name__)
CORS(app)

model = None
stoi = {}
itos = {}
block_size = 128
vocab_size = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model, stoi, itos, block_size, vocab_size
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "mark.pth")
    model_path = os.path.normpath(model_path)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = checkpoint["vocab_size"]
    block_size = checkpoint["block_size"]
    model = GPTLike(vocab_size=vocab_size, block_size=block_size, d_model=256, n_heads=8, n_layers=4).to(device)
    model.load_state_dict(checkpoint["mark"])
    model.eval()
    print(f"Mark loaded — vocab size: {vocab_size}, block size: {block_size}")

def generate(prompt, max_new_chars=150, temperature=0.8, top_k=40):
    tokens = [stoi.get(c, 0) for c in prompt]
    tokens = tokens[-block_size:]

    for _ in range(max_new_chars):
        context = tokens[-block_size:]
        input_tensor = torch.tensor([context], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor)
        logits = logits[0, -1, :] / temperature
        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[-1]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)

    output_tokens = tokens[len(tokens) - max_new_chars:]
    return "".join([itos.get(t, "") for t in output_tokens])

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message provided"}), 400
    reply = generate(message)
    return jsonify({"response": reply})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "vocab_size": vocab_size
    })

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
