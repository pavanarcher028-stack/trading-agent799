"""
Prepare crypto market data for autoresearch.
Downloads BTC, ETH, SOL, XRP at 6 timeframes: 5m, 10m, 30m, 1h, 4h, 1d.
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

# Timeframes: (label, yfinance_interval, period, resample_minutes)
TIMEFRAMES = [
    ("5m",  "5m",  "2mo",  5),
    ("10m", "5m",  "2mo",  10),   # resampled from 5m
    ("30m", "30m", "2mo",  30),
    ("1h",  "1h",  "max",  60),
    ("4h",  "1h",  "max",  240),  # resampled from 1h
    ("1d",  "1d",  "max",  1440),
]

# Clip range scales with timeframe (wider for longer TFs)
TF_CLIP = {
    "5m": 5, "10m": 5, "30m": 8,
    "1h": 10, "4h": 12, "1d": 20,
}

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_symbol(symbol):
    for label, interval, period, _ in TIMEFRAMES:
        dirname = f"{COIN_NAMES[symbol]}_{label}"
        path = os.path.join(DATA_DIR, dirname, "data.parquet")
        if os.path.exists(path):
            print(f"  {label}: already exists")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Only download raw intervals; 10m & 4h are resampled later
        if label in ("10m", "4h"):
            continue
        print(f"  Downloading {symbol} {label} ({interval}, {period})...")
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df.empty:
            print(f"    no data")
            continue
        if hasattr(df.columns, "get_level_values"):
            df.columns = [c[0].lower() for c in df.columns]
        df.to_parquet(path)
        print(f"    -> {len(df)} bars")


def resample_if_needed(symbol):
    for label, _, _, resample_min in TIMEFRAMES:
        if label not in ("10m", "4h"):
            continue
        dirname = f"{COIN_NAMES[symbol]}_{label}"
        path = os.path.join(DATA_DIR, dirname, "data.parquet")
        if os.path.exists(path):
            continue
        # Find source interval
        src_label = "5m" if label == "10m" else "1h"
        src_dir = f"{COIN_NAMES[symbol]}_{src_label}"
        src_path = os.path.join(DATA_DIR, src_dir, "data.parquet")
        if not os.path.exists(src_path):
            print(f"  {label}: source {src_label} not found, skipping")
            continue
        df = pd.read_parquet(src_path)
        df = df.resample(f"{resample_min}min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)
        print(f"  Resampled {symbol} {src_label} -> {label}: {len(df)} bars")


def download_all():
    print("=== Downloading data ===")
    for symbol in COINS:
        print(f"\n{COIN_NAMES[symbol]}:")
        download_symbol(symbol)
        resample_if_needed(symbol)

# ---------------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------------

def price_to_token(pct_change, clip_max):
    clipped = np.clip(pct_change, -clip_max, clip_max)
    normalized = (clipped + clip_max) / (2 * clip_max) * (NUM_BINS - 1)
    return np.round(normalized).astype(np.int32)


def load_symbol_tokens(symbol, label, clip_max):
    dirname = f"{COIN_NAMES[symbol]}_{label}"
    path = os.path.join(DATA_DIR, dirname, "data.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    prices = df["close"].values
    log_returns = np.diff(np.log(prices)) * 100
    return price_to_token(log_returns, clip_max)


# ---------------------------------------------------------------------------
# Build shards (interleaved across coins AND timeframes)
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
    print("\n=== Tokenizing ===")
    sequences = []  # (train_tokens, val_tokens)
    total_train, total_val = 0, 0

    for symbol in COINS:
        for label, _, _, _ in TIMEFRAMES:
            clip_max = TF_CLIP[label]
            tokens = load_symbol_tokens(symbol, label, clip_max)
            if tokens is None or len(tokens) < seq_len * 2:
                continue
            split = int(len(tokens) * 0.9)
            train_t, val_t = tokens[:split], tokens[split:]
            sequences.append((train_t, val_t))
            total_train += len(train_t)
            total_val += len(val_t)
            print(f"  {COIN_NAMES[symbol]} {label}: {len(tokens):,} tokens")

    if not sequences:
        print("ERROR: no data downloaded!")
        return

    # Interleave across all (coin, timeframe) pairs
    max_len = max(len(t) for t, _ in sequences) if sequences else 0
    interleaved_train = []
    for i in range(max_len):
        for train_t, _ in sequences:
            if i < len(train_t):
                interleaved_train.append(train_t[i])
    interleaved_train = np.array(interleaved_train, dtype=np.int32)

    max_len = max(len(v) for _, v in sequences) if sequences else 0
    interleaved_val = []
    for i in range(max_len):
        for _, val_t in sequences:
            if i < len(val_t):
                interleaved_val.append(val_t[i])
    interleaved_val = np.array(interleaved_val, dtype=np.int32)

    print(f"\nInterleaved train: {len(interleaved_train):,} tokens")
    print(f"Interleaved val:   {len(interleaved_val):,} tokens")

    train_shards = make_shards(interleaved_train, seq_len)
    val_shards = make_shards(interleaved_val, seq_len)

    for i, shard in enumerate(train_shards):
        tbl = pa.table({"text": pa.array(shard)})
        pq.write_table(tbl, os.path.join(DATA_DIR, f"train_shard_{i:05d}.parquet"))
    for i, shard in enumerate(val_shards):
        tbl = pa.table({"text": pa.array(shard)})
        pq.write_table(tbl, os.path.join(DATA_DIR, f"val_shard_{i:05d}.parquet"))

    print(f"Saved {len(train_shards)} train shards, {len(val_shards)} val shards")

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
    print(f"\nTokenizer saved to {TOKENIZER_DIR}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Cache: {CACHE_DIR}")
    print(f"Coins: {', '.join(COIN_NAMES[s] for s in COINS)}")
    print(f"Timeframes: {', '.join(l for l, _, _, _ in TIMEFRAMES)}")
    print(f"Vocab: {VOCAB_SIZE} bins")

    download_all()
    prepare_sequences()
    save_tokenizer()

    print(f"\nDone! Ready to train.")
