#!/usr/bin/env python3
"""Encode every command text (all templates x all distances, incl. the held-out 30 cm for the final test) with frozen
Qwen3-0.6B and store H_L in results/student/text_cache.pt.  The cache holds hidden states only; which texts the
train loader may use is decided in dataset.py / texts.py, not here.

  .venv/bin/python experiments/student_text_cache.py
"""
import argparse
import os

from token_policy.student.text_encoder import QwenTextEncoder, TextCache
from token_policy.student.texts import ALL_TEMPLATES, all_texts

DISTANCES = (0.20, 0.25, 0.30, 0.35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/student/text_cache.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    texts = all_texts(DISTANCES, ALL_TEMPLATES) + [""]        # "" = blank-text ablation
    enc = QwenTextEncoder(device=args.device)
    cache = TextCache(args.out)
    cache.build(texts, enc)
    lens = {t: int(cache.mask[t].sum()) for t in texts}
    print(f"cached {len(cache.H)} texts, hidden {cache.hidden}, tokens per text min {min(lens.values())} max {max(lens.values())}")
    for t in texts[:3] + texts[-4:]:
        print(f"  {lens[t]:2d} tokens | {t!r}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
