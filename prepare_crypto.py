"""
Prepare crypto market data for autoresearch.
Downloads BTC, ETH, SOL, XRP hourly data, quantizes price changes into tokens.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
MAX_SEQ_LEN = 256
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 65536

NUM_BINS = 64
VOCAB_SIZE = NUM_BINS + 1

COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
COIN_NAMES = {s: s.split("-")[0] for s in COINS}

# ---------------------------------------------------------------------------
# Step 1: Download data for all coins
# ---------------------------------------------------------------------------

def download_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    for symbol in COINS:
        path = os.path.join(DATA_DIR, f"{COIN_NAMES[symbol]}_hourly.parquet")
        if os.path.exists(path):
            print(f"{COIN_NAMES[symbol]}: already exists")
            continue
        print(f"Downloading {symbol} hourly...")
        df = yf.download(symbol, interval="1h", period="max", progress=False)
        if hasattr(df.columns, "get_level_values"):
            df.columns = [c[0].lower() for c in df.columns]
        df.to_parquet(path)
        print(f"  -> {len(df)} bars")

# ---------------------------------------------------------------------------
# Step 2: Quantize price changes
# ---------------------------------------------------------------------------

def price_to_token(pct_change, num_bins=NUM_BINS):
    clipped = np.clip(pct_change, -15, 15)
    normalized = (clipped + 15) / 30 * (num_bins - 1)
    return int(round(normalized))

def load_coin_tokens(symbol):
    path = os.path.join(DATA_DIR, f"{COIN_NAMES[symbol]}_hourly.parquet")
    df = pd.read_parquet(path)
    prices = df["close"].values
    log_returns = np.diff(np.log(prices)) * 100
    tokens = np.array([price_to_token(r) for r in log_returns], dtype=np.int32)
    return tokens

# ---------------------------------------------------------------------------
# Step 3: Build shards (interleaved across coins)
# ---------------------------------------------------------------------------

def make_shards(all_tokens, seq_len, shard_size_seqs=5000):
    num_seqs = len(all_tokens) // seq_len
    all_seqs = []
    for j in range(num_seqs):
        s = all_tokens[j * seq_len:(j + 1) * seq_len]
        all_seqs.append(" ".join(str(t) for t in s))
    num_shards = max(1, len(all_seqs) // shard_size_seqs)
    shards = []
    for i in range(num_shards):
        start = i * shard_size_seqs
        end = start + shard_size_seqs if i < num_shards - 1 else len(all_seqs)
        shards.append(all_seqs[start:end])
    return shards

def prepare_sequences(seq_len=MAX_SEQ_LEN):
    all_train = []
    all_val = []

    for symbol in COINS:
        tokens = load_coin_tokens(symbol)
        split = int(len(tokens) * 0.9)
        train_t, val_t = tokens[:split], tokens[split:]
        all_train.append(train_t)
        all_val.append(val_t)
        print(f"{COIN_NAMES[symbol]}: {len(tokens):,} tokens (train: {len(train_t):,}, val: {len(val_t):,})")

    # Interleave training tokens from all coins
    max_train = max(len(t) for t in all_train)
    interleaved_train = []
    for i in range(max_train):
        for ct in all_train:
            if i < len(ct):
                interleaved_train.append(ct[i])
    interleaved_train = np.array(interleaved_train, dtype=np.int32)
    print(f"Interleaved train tokens: {len(interleaved_train):,}")

    # Interleave val tokens
    max_val = max(len(t) for t in all_val)
    interleaved_val = []
    for i in range(max_val):
        for ct in all_val:
            if i < len(ct):
                interleaved_val.append(ct[i])
    interleaved_val = np.array(interleaved_val, dtype=np.int32)
    print(f"Interleaved val tokens: {len(interleaved_val):,}")

    train_shards = make_shards(interleaved_train, seq_len)
    val_shards = make_shards(interleaved_val, seq_len)

    # Save train shards
    for i, shard in enumerate(train_shards):
        tbl = pa.table({"text": pa.array(shard)})
        pq.write_table(tbl, os.path.join(DATA_DIR, f"train_shard_{i:05d}.parquet"))

    # Save val shards
    for i, shard in enumerate(val_shards):
        tbl = pa.table({"text": pa.array(shard)})
        pq.write_table(tbl, os.path.join(DATA_DIR, f"val_shard_{i:05d}.parquet"))

    print(f"Saved {len(train_shards)} train shards, {len(val_shards)} val shards")
    return interleaved_train, interleaved_val

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class PriceTokenizer:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.bos_token_id = 0

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        return cls()

    def get_vocab_size(self):
        return self.vocab_size

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if isinstance(text, str):
            ids = [int(x) for x in text.split()]
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else 0
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = []
            for t in text:
                row = [int(x) for x in t.split()]
                if prepend is not None:
                    prepend_id = prepend if isinstance(prepend, int) else 0
                    row.insert(0, prepend_id)
                ids.append(row)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def save_tokenizer():
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    token_bytes = torch.ones(VOCAB_SIZE, dtype=torch.int32)
    torch.save(token_bytes, os.path.join(TOKENIZER_DIR, "token_bytes.pt"))
    with open(os.path.join(TOKENIZER_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(None, f)
    print(f"Tokenizer saved to {TOKENIZER_DIR}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Coins: {', '.join(COIN_NAMES[s] for s in COINS)}")
    print()

    download_all()
    print()

    prepare_sequences()
    print()

    save_tokenizer()
    print(f"Vocab size: {VOCAB_SIZE} ({NUM_BINS} price change bins)")
    print(f"Sequence length: {MAX_SEQ_LEN}")
    print(f"Done! Ready to train crypto price model.")
