"""Training code (depends on models/, never imported by it).

    config.py      TrainConfig: optimizer / schedule / epochs / clipping / early stopping
    optim.py       AdamW + warmup-cosine schedule
    checkpoint.py  save / load (+ legacy checkpoint compatibility)
    im/            imitation learning (behaviour cloning of teacher tokens; DAgger etc. later)
    rl/            reinforcement learning (placeholder)
"""
