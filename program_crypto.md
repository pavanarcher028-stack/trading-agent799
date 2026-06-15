# Crypto Trading Agent - autoresearch

This is an experiment to have an AI agent autonomously improve a crypto price prediction model.

## Setup

1. Read `prepare_crypto.py` (read-only — data pipeline)
2. Read `train_crypto.py` (editable — model, training loop)
3. Run baseline: `python train_crypto.py`
4. Log results in `results.tsv`

## Experimentation loop

Same loop as autoresearch: edit `train_crypto.py`, run, keep/discard based on val_loss improvement.
