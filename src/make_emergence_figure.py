#!/usr/bin/env python3
"""Regenerate the emergence-at-scale figure (paper Figure 2) from data/scaling_v8.csv.
Run from the repository root:  python src/make_emergence_figure.py
"""
import csv, statistics as st, math, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
g = defaultdict(lambda: defaultdict(list)); pg = defaultdict(lambda: defaultdict(list))
with open("data/scaling_v8.csv") as f:
    for r in csv.DictReader(f):
        g[r["size"]][r["variant"]].append(float(r["val_loss"]))
        pg[r["size"]][r["variant"]].append(float(r["params_M"]))
sizes = ["60M","150M","320M","590M","1B"]
px = [st.mean(pg[s]["review_neutral"]) for s in sizes]
def gap_se(base, sz):
    rv, bl = g[sz]["review_neutral"], g[sz][base]
    gap = st.mean(bl) - st.mean(rv)
    se = math.sqrt(st.variance(rv)/len(rv) + st.variance(bl)/len(bl))
    return gap, se, abs(gap/se) > 2.6
fig, ax = plt.subplots(figsize=(8,5.2))
for base,col,mk,lab in [("highway","#8e44ad","s","vs Highway gate (param-matched)"),
                        ("standard","#444","o","vs Standard residual (param-matched)")]:
    gaps=[];ses=[];sig=[]
    for s in sizes:
        gp,se,si=gap_se(base,s); gaps.append(gp);ses.append(se);sig.append(si)
    ax.errorbar(px,gaps,yerr=ses,marker=mk,color=col,lw=2,ms=8,capsize=4,label=lab)
    for x,y,si in zip(px,gaps,sig):
        if si: ax.annotate("*",(x,y),textcoords="offset points",xytext=(0,8),fontsize=18,color=col,ha="center")
ax.axhline(0,color="#aaa",lw=1.2,ls="--")
ax.set_xscale("log"); ax.set_xlabel("parameters (millions, log scale)")
ax.set_ylabel("Review's advantage  (baseline loss - Review loss)")
ax.set_title("Review Residuals: the benefit emerges with scale")
ax.set_xticks(px); ax.set_xticklabels(sizes); ax.grid(alpha=.25); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("paper/emergence_curve.png",dpi=150)
print("wrote paper/emergence_curve.png")
