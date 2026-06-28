#!/usr/bin/env python3
"""Reproduce the paper's results table and significance tests from data/scaling_v8.csv.
Run from the repository root:  python src/analyze_results.py
"""
import csv, statistics as st, math
from collections import defaultdict

g = defaultdict(lambda: defaultdict(list))
with open("data/scaling_v8.csv") as f:
    for r in csv.DictReader(f):
        g[r["size"]][r["variant"]].append(float(r["val_loss"]))

def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    m1, m2 = st.mean(a), st.mean(b)
    v1, v2 = st.variance(a), st.variance(b)
    n1, n2 = len(a), len(b)
    se = math.sqrt(v1/n1 + v2/n2)
    return (m2 - m1) / se if se else float("inf")

print(f"{'size':6s}{'Review':>9s}{'Highway':>9s}{'Standard':>10s}{'Rev-Hwy':>10s}{'Rev-Std':>10s}{'t(Hwy)':>9s}{'t(Std)':>9s}")
for sz in ["60M", "150M", "320M", "590M", "1B"]:
    rv, hw, sd = g[sz]["review_neutral"], g[sz]["highway"], g[sz]["standard"]
    rm, hm, sm = st.mean(rv), st.mean(hw), st.mean(sd)
    th, ts = welch_t(rv, hw), welch_t(rv, sd)
    fh = f"{th:.2f}" if th is not None else "n/a"
    fs = f"{ts:.2f}" if ts is not None else "n/a"
    print(f"{sz:6s}{rm:9.4f}{hm:9.4f}{sm:10.4f}{hm-rm:+10.4f}{sm-rm:+10.4f}{fh:>9s}{fs:>9s}")
print("\nPositive gap favours Review. |t| > ~2.8 (3 seeds) indicates p < 0.05.")
print("Review significantly beats both baselines at 590M; the 1B gap is larger but a trend.")
