import torch
import torch.nn.functional as F
from model import GPTLike

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(
    "models/mark.pth",
    map_location=device,
    weights_only=False
)

stoi = ckpt["stoi"]
itos = ckpt["itos"]
vocab_size = ckpt["vocab_size"]
block_size = ckpt["block_size"]

model = GPTLike(vocab_size=vocab_size, block_size=block_size, d_model=256, n_heads=8, n_layers=4).to(device)
model.load_state_dict(ckpt["mark"])
model.eval()

print("Mark is ready. Type 'quit' to exit.\n")

while True:
    prompt = input("You: ")
    if prompt.lower() == "quit":
        break

    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)

    print("Mark: ", end="", flush=True)

    for _ in range(200):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / 0.8
        top_k = 40
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
        print(itos[nxt.item()], end="", flush=True)

    print()