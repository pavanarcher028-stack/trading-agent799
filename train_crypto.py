"""
Crypto price prediction with autoresearch-style agent loop.
Modified for Colab T4 (Flash Attention 2) and price movement tokens.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_crypto import MAX_SEQ_LEN, TIME_BUDGET, PriceTokenizer

# Try Flash Attention 2 (works on T4), fallback to manual attention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False
    print("WARNING: flash-attn not installed, using manual attention")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOCAB_SIZE = 65  # 64 price bins + 1

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 4       # smaller model for less data
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 256
    window_pattern: str = "L"  # full attention (no sliding window)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if HAS_FLASH:
            y = flash_attn_func(q, k, v, causal=True)
            y = y.contiguous().view(B, T, -1)
        else:
            # Manual attention
            qk = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)
            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            qk = qk + mask
            attn = F.softmax(qk, dim=-1)
            y = attn @ v
            y = y.contiguous().view(B, T, -1)

        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.transformer.wte.to(dtype=torch.bfloat16)

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x)
        x = norm(x)
        logits = self.lm_head(x)
        logits = logits.float()

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (simple AdamW)
# ---------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


class SimpleAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.005, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                self._step_t.fill_(state['step'])
                self._lr_t.fill_(lr)
                self._beta1_t.fill_(beta1)
                self._beta2_t.fill_(beta2)
                self._eps_t.fill_(eps)
                self._wd_t.fill_(wd)
                adamw_step_fused(p, p.grad, state['exp_avg'], state['exp_avg_sq'],
                                self._step_t, self._lr_t, self._beta1_t,
                                self._beta2_t, self._eps_t, self._wd_t)

# ---------------------------------------------------------------------------
# Simplified dataloader for crypto data
# ---------------------------------------------------------------------------

class CryptoDataset:
    """Reads parquet files and yields batches of token sequences."""

    def __init__(self, split, data_dir, batch_size, seq_len):
        import pyarrow.parquet as pq
        files = sorted(f for f in os.listdir(data_dir)
                       if f.endswith(".parquet") and f.startswith(split))
        assert files, f"No {split} shards found in {data_dir}"
        self.paths = [os.path.join(data_dir, f) for f in files]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = PriceTokenizer()

    def __iter__(self):
        import pyarrow.parquet as pq
        epoch = 1
        while True:
            for path in self.paths:
                pf = pq.ParquetFile(path)
                for rg_idx in range(pf.num_row_groups):
                    rg = pf.read_row_group(rg_idx)
                    texts = rg.column('text').to_pylist()
                    for i in range(0, len(texts), self.batch_size):
                        batch_texts = texts[i:i + self.batch_size]
                        token_ids = self.tokenizer.encode(batch_texts, prepend=0)
                        # Pad to batch_size
                        if len(token_ids) < self.batch_size:
                            pad_len = self.batch_size - len(token_ids)
                            token_ids.extend([[0] * self.seq_len] * pad_len)
                        x = torch.tensor([t[:-1] for t in token_ids], dtype=torch.long)
                        y = torch.tensor([t[1:] for t in token_ids], dtype=torch.long)
                        yield x, y, epoch
            epoch += 1


def evaluate(model, data_dir, batch_size, seq_len):
    """Simple validation loss."""
    model.eval()
    val_dataset = CryptoDataset("val", data_dir, batch_size, seq_len)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for _ in range(20):
            x, y, _ = next(iter(val_dataset))
            x, y = x.cuda(), y.cuda()
            loss = model(x, y)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / count if count > 0 else 0

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEPTH = 4
N_HEAD = 4
N_EMBD = 256
TOTAL_BATCH_SIZE = 2**16  # tokens per step
DEVICE_BATCH_SIZE = 32
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.1

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

config = GPTConfig(
    sequence_len=MAX_SEQ_LEN, vocab_size=VOCAB_SIZE,
    n_layer=DEPTH, n_head=N_HEAD, n_kv_head=N_HEAD, n_embd=N_EMBD,
    window_pattern="L",
)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

tokens_per_batch = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_batch == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_batch

optimizer = SimpleAdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
model = torch.compile(model, dynamic=False)

from prepare_crypto import DATA_DIR
train_loader = CryptoDataset("train", DATA_DIR, DEVICE_BATCH_SIZE, MAX_SEQ_LEN)

print(f"Training on crypto data")
print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        x, y, epoch = next(iter(train_loader))
        x, y = x.cuda(), y.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()

    # LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    if progress < WARMUP_RATIO:
        lrm = progress / WARMUP_RATIO
    elif progress < 1.0 - WARMDOWN_RATIO:
        lrm = 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        lrm = cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
    for g in optimizer.param_groups:
        g['lr'] = LEARNING_RATE * lrm

    optimizer.step()
    model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 5:
        total_training_time += dt

    train_loss_f = train_loss.item()
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_loss:.6f} | lr: {lrm:.4f} | dt: {dt*1000:.0f}ms | epoch: {epoch}", end="", flush=True)

    if step <= 5:
        gc.collect()

    step += 1
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()
total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
val_loss = evaluate(model, DATA_DIR, DEVICE_BATCH_SIZE, MAX_SEQ_LEN)
t_end = time.time()

print("---")
print(f"val_loss:         {val_loss:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")
print(f"total_tokens:     {total_tokens:,}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params:,}")
