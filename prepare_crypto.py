"""
Prepare crypto market data for autoresearch.
Downloads BTC/USDT hourly OHLCV, quantizes price changes into tokens.
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
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
MAX_SEQ_LEN = 256       # shorter context for price sequences
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 65536

# Number of price change bins (vocab size)
NUM_BINS = 64  # 32 up + 32 down
VOCAB_SIZE = NUM_BINS + 1  # +1 for padding/separator

# ---------------------------------------------------------------------------
# Step 1: Download BTC data
# ---------------------------------------------------------------------------

def download_btc_data():
    """Download hourly BTC/USDT data from Binance via yfinance."""
    os.makedirs(DATA_DIR, exist_ok=True)
    parquet_path = os.path.join(DATA_DIR, "btc_hourly.parquet")

    if os.path.exists(parquet_path):
        print(f"BTC data already exists at {parquet_path}")
        return

    print("Downloading BTC/USDT hourly data...")
    btc = yf.download("BTC-USD", interval="1h", period="max", progress=False)
    btc.columns = [c[0].lower() for c in btc.columns]
    btc.to_parquet(parquet_path)
    print(f"Downloaded {len(btc)} hourly bars -> {parquet_path}")

# ---------------------------------------------------------------------------
# Step 2: Quantize price changes into tokens
# ---------------------------------------------------------------------------

def price_to_token(pct_change, num_bins=NUM_BINS):
    """Map a percentage price change to a token ID."""
    half = num_bins // 2
    # Clip to reasonable range (±15% for hourly)
    clipped = np.clip(pct_change, -15, 15)
    # Normalize to [0, num_bins-1]
    normalized = (clipped + 15) / 30 * (num_bins - 1)
    return int(round(normalized))

def token_to_description(token_id, num_bins=NUM_BINS):
    """Convert token ID back to a human-readable price change."""
    half = num_bins // 2
    normalized = token_id / (num_bins - 1)
    pct = normalized * 30 - 15
    return f"{pct:+.2f}%"

def prepare_sequences(seq_len=MAX_SEQ_LEN):
    """Load BTC data, quantize, create sequences."""
    parquet_path = os.path.join(DATA_DIR, "btc_hourly.parquet")
    df = pd.read_parquet(parquet_path)

    # Compute log returns
    prices = df["close"].values
    log_returns = np.diff(np.log(prices)) * 100  # percentage
    tokens = np.array([price_to_token(r) for r in log_returns], dtype=np.int32)

    # Split into train/val (90/10)
    split = int(len(tokens) * 0.9)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    print(f"Total tokens: {len(tokens):,} | Train: {len(train_tokens):,} | Val: {len(val_tokens):,}")

    # Create sequences as space-separated token strings (to match autoresearch parquet format)
    def create_shard(token_array, num_shards=5):
        shard_size = len(token_array) // num_shards
        shards = []
        for i in range(num_shards):
            start = i * shard_size
            end = start + shard_size if i < num_shards - 1 else len(token_array)
            # Create sequences of seq_len tokens as strings
            seq_tokens = token_array[start:end]
            num_seqs = len(seq_tokens) // seq_len
            seqs = []
            for j in range(num_seqs):
                s = seq_tokens[j * seq_len:(j + 1) * seq_len]
                seqs.append(" ".join(str(t) for t in s))
            shards.append(seqs)
        return shards

    train_shards = create_shard(train_tokens)
    val_shards = create_shard(val_tokens)

    # Save train shards
    for i, shard in enumerate(train_shards):
        tbl = pa.table({"text": pa.array(shard)})
        pq.write_table(tbl, os.path.join(DATA_DIR, f"train_shard_{i:05d}.parquet"))

    # Save val shard
    for i, shard in enumerate(val_shards):
        tbl = pa.table({"text": pa.array(shard)})
        pq.write_table(tbl, os.path.join(DATA_DIR, f"val_shard_{i:05d}.parquet"))

    print(f"Saved {len(train_shards)} train shards, {len(val_shards)} val shards")
    return train_tokens, val_tokens

# ---------------------------------------------------------------------------
# Step 3: Simple tokenizer (maps token IDs to themselves)
# ---------------------------------------------------------------------------

class PriceTokenizer:
    """Simple tokenizer for price movement tokens."""

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
    """Save a dummy tokenizer file to keep prepare.py imports happy."""
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    # Save a pickle that matches the interface
    token_bytes = torch.ones(VOCAB_SIZE, dtype=torch.int32)
    torch.save(token_bytes, os.path.join(TOKENIZER_DIR, "token_bytes.pt"))
    # Save tokenizer pickle
    with open(os.path.join(TOKENIZER_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(None, f)
    print(f"Tokenizer saved to {TOKENIZER_DIR}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Cache directory: {CACHE_DIR}")
    print()

    download_btc_data()
    print()

    train_tokens, val_tokens = prepare_sequences()
    print()

    save_tokenizer()
    print(f"Vocab size: {VOCAB_SIZE} ({NUM_BINS} price change bins)")
    print(f"Sequence length: {MAX_SEQ_LEN}")
    print(f"Done! Ready to train crypto price model.")
