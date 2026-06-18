import torch
import torch.nn.functional as F
import os
import time
from model import GPTLike
from tqdm import tqdm

epochs = 50
batch_size = 32
block_size = 128
learning_rate = 3e-4
eval_iters = 40
min_loss = 0.4
eval_every = 2
warmup_epochs = 2

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("datasets/data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Dataset loaded — {len(raw_text):,} characters | {len(raw_text.split()):,} words")

chars = sorted(list(set(raw_text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

data = torch.tensor([stoi[c] for c in raw_text], dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

def get_batch(split="train"):
    src = train_data if split == "train" else val_data
    if len(src) <= block_size:
        ix = torch.zeros(batch_size, dtype=torch.long)
    else:
        ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits = model(X)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), Y.reshape(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def sample(prompt, max_new_chars=100, temperature=0.8, top_k=40):
    model.eval()
    tokens = [stoi.get(c, 0) for c in prompt]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    for _ in range(max_new_chars):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_token], dim=1)
    model.train()
    output_tokens = idx[0].tolist()[len(tokens):]
    return "".join([itos.get(t, "") for t in output_tokens])

model = GPTLike(vocab_size, block_size=block_size, d_model=256, n_heads=8, n_layers=4).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Training on {device} | vocab: {vocab_size} | params: {total_params:,}")

opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

batches_per_epoch = max(20, len(train_data) // (batch_size * block_size))
total_steps = epochs * batches_per_epoch

def lr_lambda(step):
    warmup_steps = warmup_epochs * batches_per_epoch
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.05, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

print(f"Batches per epoch: {batches_per_epoch} | Total steps: {total_steps:,}")

os.makedirs("models", exist_ok=True)
best_val_loss = float("inf")
global_step = 0
start_time = time.time()

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f}")
    epoch_losses = []
    epoch_start = time.time()

    for batch_idx in tqdm(range(batches_per_epoch), desc=f"Epoch {epoch + 1}"):
        x, y = get_batch("train")
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
        epoch_losses.append(loss.item())
        global_step += 1

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    epoch_time = time.time() - epoch_start

    if (epoch + 1) % eval_every == 0 or epoch == 0 or epoch == epochs - 1:
        losses = estimate_loss()
        print(f"Epoch {epoch + 1} | train: {avg_loss:.4f} | val: {losses['val']:.4f} | time: {epoch_time:.1f}s")

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save({
                "mark": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "vocab_size": vocab_size,
                "block_size": block_size
            }, "models/mark.pth")
            print(f"  Saved best model (val loss: {best_val_loss:.4f})")

        print(f"  Sample: {sample('hey', max_new_chars=80)}")

        if best_val_loss <= min_loss:
            print(f"Reached target loss {min_loss} — stopping early.")
            break
    else:
        print(f"Epoch {epoch + 1} | train: {avg_loss:.4f} | time: {epoch_time:.1f}s")

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time/60:.1f} minutes | Best val loss: {best_val_loss:.4f}")
input("Press Enter to exit...")