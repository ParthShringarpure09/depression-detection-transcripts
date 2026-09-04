#!/usr/bin/env python
# coding: utf-8

# # Results: Punctual or Continuous Depression Traces in Language?
# 
# This notebook collects all experimental results for the text branch of the
# depression detection pipeline. **No models are trained here** — every result is
# loaded from the JSON files written by the experiment notebooks:
# 
# | File | Contents |
# |---|---|
# | `FFNN_R5/results_sil_folds.json` | FFNN + majority vote (SIL) |
# | `Linear_baseline/results_linear_folds.json` | LogReg & LinearSVC + majority vote (SIL) |
# | `MIL/results_minn_pooled.json` | MINN, max- and mean-pooling |
# | `MIL_Linear/results_summary.json` | Raw-feature max/mean pooling + LogReg |
# 
# All experiments use the same protocol: 116 speakers from the Androids Corpus
# (64 with depression, 52 controls), person-independent 5-fold cross-validation
# using the folds distributed with the corpus, and `dbmdz/bert-base-italian-xxl-cased`
# embeddings (768-dim, mean-pooled per segment). Neural models are averaged over
# R = 5 random initialisations within each fold.
# 
# Reported values are **recording-level F1**, averaged across the 5 folds, with
# standard deviation computed **across folds**.

# In[1]:


from pathlib import Path
import json

import numpy as np
from scipy import stats

MODELS_DIR = Path("../Models")

def load_json(relpath):
    with open(MODELS_DIR / relpath) as f:
        return json.load(f)

def to_int_keys(d):
    """JSON keys are strings; convert N-level keys back to ints."""
    return {int(k): v for k, v in d.items()}

# --- load every result file ---
sil_ffnn   = to_int_keys(load_json("FFNN_R5/results_sil_folds.json"))
_linear    = load_json("Linear_baseline/results_linear_folds.json")
sil_lr     = to_int_keys(_linear["logistic_regression"])
#sil_svm    = to_int_keys(_linear["linear_svm"])
_minn      = load_json("MIL/results_minn_pooled.json")
minn_max   = to_int_keys(_minn["max"])
minn_mean  = to_int_keys(_minn["mean"])
_pooled_lr = load_json("MIL_Linear/results_summary.json")
pool_max   = to_int_keys(_pooled_lr["max"])
pool_mean  = to_int_keys(_pooled_lr["mean"])

N_LEVELS = [1, 2, 4, 8, 16, 32, 64]

# --- register every method with a label and its aggregation type ---
METHODS = {
    "FFNN-SIL":      {"data": sil_ffnn,  "agg": "vote",  "family": "neural"},
    "LogReg-SIL":    {"data": sil_lr,    "agg": "vote",  "family": "linear"},
    #"LinSVC-SIL":    {"data": sil_svm,   "agg": "vote",  "family": "linear"},
    "MINN-max":      {"data": minn_max,  "agg": "pool",  "family": "neural"},
    "MINN-mean":     {"data": minn_mean, "agg": "pool",  "family": "neural"},
    "PoolLR-max":    {"data": pool_max,  "agg": "pool",  "family": "linear"},
    "PoolLR-mean":   {"data": pool_mean, "agg": "pool",  "family": "linear"},
}

# --- integrity check: every method must have all N-levels and 5 per-fold F1s ---
problems = []
for name, m in METHODS.items():
    for n in N_LEVELS:
        if n not in m["data"]:
            problems.append(f"{name}: missing N={n}")
        elif len(m["data"][n].get("fold_f1s", [])) != 5:
            problems.append(f"{name} N={n}: expected 5 fold F1s")

print(f"Loaded {len(METHODS)} methods x {len(N_LEVELS)} N-levels")
print("Integrity check:", "PASS" if not problems else "FAIL")
for p in problems:
    print("  !", p)


# ## 1. Main results table
# 
# Recording-level F1 (%) for every method at every segmentation granularity.
# Methods are grouped by how segment-level evidence is combined into a
# speaker-level decision:
# 
# - **Vote** — each segment is classified independently and the speaker label is
#   assigned by majority vote over segments (Single Instance Learning).
# - **Pool** — segment representations are combined into a single speaker-level
#   representation before any decision is made (Multiple Instance Learning and
#   its raw-feature equivalents).
# 
# Within the pooling methods, `max` implements the selection assumption
# (a bag is positive if at least one instance is positive) and `mean` implements
# the aggregation assumption (evidence is combined across all instances).

# 

# In[2]:


def f1_row(method_name):
    """Return the 7 mean F1s (as %) for a method, across N-levels."""
    d = METHODS[method_name]["data"]
    return [d[n]["f1_mean"] * 100 for n in N_LEVELS]

def std_row(method_name):
    d = METHODS[method_name]["data"]
    return [d[n]["f1_std"] * 100 for n in N_LEVELS]


# --- header ---
header = f"{'Method':<13} {'Agg':<6}" + "".join(f"{'N='+str(n):>14}" for n in N_LEVELS)
print(header)
print("-" * len(header))

# --- print vote methods first, then pool methods ---
for agg_type in ["vote", "pool"]:
    for name, m in METHODS.items():
        if m["agg"] != agg_type:
            continue
        means, stds = f1_row(name), std_row(name)
        cells = "".join(f"{mu:8.2f}±{sd:4.2f}" for mu, sd in zip(means, stds))
        print(f"{name:<13} {agg_type:<6}{cells}")
    print()

# --- group averages at the finest granularities ---
print("Mean F1 across methods, by aggregation type:")
for agg_type in ["vote", "pool"]:
    names = [k for k, v in METHODS.items() if v["agg"] == agg_type]
    for n in [1, 8, 32, 64]:
        vals = [METHODS[k]["data"][n]["f1_mean"] * 100 for k in names]
        print(f"  {agg_type:>4}  N={n:<3}: {np.mean(vals):6.2f}%")
    print()


# ## 2. Significance testing
# 
# All tests are **paired two-tailed t-tests** over the five cross-validation folds.
# Pairing is appropriate because every method is evaluated on the identical folds
# and identical data, so fold-to-fold variation is shared and can be removed.
# 
# Two comparisons are treated as **primary**:
# 
# 1. **max- vs mean-pooling** — this distinguishes the punctual account (selection
#    should help) from the continuous account (selection and aggregation should be
#    equivalent). Tested in both the neural and linear settings.
# 2. **vote- vs pool-based aggregation** — this tests whether the mechanism used to
#    combine segment-level evidence affects performance at fine granularity.
# 
# Remaining comparisons are reported as exploratory. With only five folds these
# tests have low statistical power: a non-significant result indicates that a
# difference could not be detected at this sample size, **not** that no difference
# exists.

# In[3]:


def paired_test(name_a, name_b, n):
    """Paired two-tailed t-test between two methods at one N-level.
    Returns (mean_a, mean_b, difference, p-value), all F1s as %."""
    a = np.array(METHODS[name_a]["data"][n]["fold_f1s"])
    b = np.array(METHODS[name_b]["data"][n]["fold_f1s"])
    if np.allclose(a, b):                       # identical -> undefined test
        return a.mean()*100, b.mean()*100, 0.0, np.nan
    _, p = stats.ttest_rel(a, b)
    return a.mean()*100, b.mean()*100, (a.mean()-b.mean())*100, p


def comparison_table(name_a, name_b, title, alpha=0.05, n_tests=7):
    """Print a paired comparison across all N-levels with Bonferroni flagging."""
    bonf = alpha / n_tests
    print(f"{title}")
    print(f"{'N':>4} | {name_a:>12} | {name_b:>12} | {'diff':>7} | {'p':>8} | sig")
    print("-" * 66)
    for n in N_LEVELS:
        ma, mb, diff, p = paired_test(name_a, name_b, n)
        if np.isnan(p):
            sig = "identical"
        else:
            sig = "**" if p < bonf else ("*" if p < alpha else "")
        p_str = "   n/a  " if np.isnan(p) else f"{p:8.4f}"
        print(f"{n:>4} | {ma:11.2f}% | {mb:11.2f}% | {diff:+6.2f} | {p_str} | {sig}")
    print(f"\n*  p < {alpha}   ** p < {bonf:.4f} (Bonferroni, {n_tests} comparisons)\n")


# ---- PRIMARY TEST 1: selection vs aggregation (punctual vs continuous) ----
comparison_table("MINN-max", "MINN-mean",
                 "Selection vs aggregation — NEURAL (MINN)")

comparison_table("PoolLR-max", "PoolLR-mean",
                 "Selection vs aggregation — LINEAR (pooled features + LogReg)")


# ### Primary test 2 — vote- vs pool-based aggregation
# 
# Having established that the choice of pooling operator (max vs mean) does not
# affect performance, the remaining question is whether the *mechanism* used to
# combine segment-level evidence matters. Two comparisons are made, each holding
# the underlying classifier fixed so that only the aggregation mechanism differs:
# 
# - **Neural:** FFNN with majority vote vs MINN with max-pooling. Both use an
#   identical 32–64–128 ReLU encoder; they differ only in whether a decision is
#   made per segment and then voted on, or whether segment representations are
#   pooled before a single decision.
# - **Linear:** Logistic Regression with majority vote vs Logistic Regression
#   applied to max-pooled raw embeddings.

# 

# In[4]:


# ---- PRIMARY TEST 2: voting vs pooling ----
comparison_table("MINN-max", "FFNN-SIL",
                 "Pooling vs voting — NEURAL (same encoder, differs only in aggregation)")

comparison_table("PoolLR-max", "LogReg-SIL",
                 "Pooling vs voting — LINEAR (same classifier, differs only in aggregation)")


# ### Exploratory test — degradation with granularity
# 
# The continuous account predicts that performance should not depend strongly on
# segment length, since depression-relevant information is expected to be present
# throughout the interview. Each method is therefore tested for a change in
# performance between the coarsest (N = 1) and finest (N = 64) granularity.
# These tests are exploratory and are not corrected for multiple comparisons.

# In[5]:


print("Change in F1 from N=1 to N=64, per method")
print(f"{'Method':<13} {'Agg':<6} {'N=1':>8} {'N=64':>8} {'change':>8} {'p':>9} sig")
print("-" * 62)

for agg_type in ["vote", "pool"]:
    for name, m in METHODS.items():
        if m["agg"] != agg_type:
            continue
        a = np.array(m["data"][1]["fold_f1s"])
        b = np.array(m["data"][64]["fold_f1s"])
        _, p = stats.ttest_rel(b, a)
        change = (b.mean() - a.mean()) * 100
        sig = "*" if p < 0.05 else ""
        print(f"{name:<13} {agg_type:<6} {a.mean()*100:7.2f}% {b.mean()*100:7.2f}% "
              f"{change:+7.2f} {p:9.4f} {sig}")
    print()


# ## 3. Summary figure
# 
# Recording-level F1 as a function of segmentation granularity, with methods
# grouped by aggregation mechanism. The horizontal axis is logarithmic in N.

# In[6]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

styles = {
    "vote": {"color": "#c0392b", "linestyle": "--", "marker": "o"},
    "pool": {"color": "#2471a3", "linestyle": "-",  "marker": "s"},
}

for name, m in METHODS.items():
    s = styles[m["agg"]]
    means = [m["data"][n]["f1_mean"] * 100 for n in N_LEVELS]
    stds  = [m["data"][n]["f1_std"]  * 100 for n in N_LEVELS]
    ax.errorbar(N_LEVELS, means, yerr=stds,
                label=f"{name} ({m['agg']})",
                color=s["color"], linestyle=s["linestyle"], marker=s["marker"],
                capsize=3, alpha=0.75, markersize=5, linewidth=1.4)

ax.set_xscale("log", base=2)
ax.set_xticks(N_LEVELS)
ax.set_xticklabels(N_LEVELS)
ax.set_xlabel("Number of segments per interview (N)")
ax.set_ylabel("Recording-level F1 (%)")
ax.set_title("Depression detection performance vs segmentation granularity")
ax.set_ylim(70, 100)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower left", ncol=2)

plt.tight_layout()
plt.savefig("../Models/results_by_granularity.png", dpi=200)
plt.show()


# ## 4. Summary of findings
# 
# **1. Selection and aggregation pooling are equivalent.**
# Max-pooling (which implements the standard MIL assumption that a bag is positive
# if at least one instance is positive) and mean-pooling (which combines evidence
# across all instances) perform equivalently at every granularity, in both the
# neural and the linear setting. No comparison survives correction for multiple
# tests. If depression-relevant evidence were concentrated in a small number of
# segments, selection would be expected to outperform aggregation; it does not.
# 
# **2. Performance degrades with granularity only for vote-based methods.**
# All three majority-vote methods show a statistically significant drop in F1
# between N = 1 and N = 64 (−6.9 to −10.2 points, all p < 0.05). None of the four
# pooling methods shows a significant drop (−0.2 to −3.9 points, all p > 0.14).
# The separation is complete across both model families.
# 
# **3. Pooling outperforms voting at fine granularity.**
# Averaged across methods, vote-based aggregation falls from 91.4% F1 at N = 1 to
# 82.5% at N = 64, whereas pooling-based aggregation falls only from 91.4% to
# 89.6%. In the neural setting, where the encoder is held identical and only the
# aggregation mechanism differs, MINN-max significantly outperforms FFNN-SIL at
# N = 32 (+6.7 points, p = 0.0058, Bonferroni-corrected). The same difference in
# direction and magnitude is observed at N = 64 and in the linear setting without
# reaching significance.
# 
# ### Interpretation
# 
# These results support the **continuous** account of depression-related linguistic
# evidence. The equivalence of selection and aggregation pooling indicates that
# depression-relevant information is distributed across the interview rather than
# concentrated in a limited number of passages.
# 
# The apparent advantage of Multiple Instance Learning over Single Instance
# Learning at fine granularity is therefore better attributed to the aggregation
# mechanism than to evidence localisation. Majority voting requires each segment
# to be assigned a hard binary label before evidence is combined; at N = 64 each
# segment contains roughly ten words, which is insufficient for a reliable
# segment-level decision, and voting propagates this unreliability. Pooling
# combines continuous segment representations before any decision is taken and is
# consequently unaffected.
# 
# This conclusion is consistent with Alsarrani et al. (2025), who also report
# continuous traces for the language modality using LIWC representations. The
# present results additionally indicate that the performance drop they observe
# beyond N = 32 is attributable to the representation and the aggregation
# mechanism rather than to the distribution of the evidence itself.
# 
# ### Limitations
# 
# - Five folds provide limited statistical power; non-significant results indicate
#   that a difference could not be detected, not that none exists.
# - The fold splits supplied with the corpus are unstratified (Fold 4 has a
#   markedly skewed HC/PT test distribution), which contributes to the wide
#   cross-fold standard deviations visible in the figure.
# - Linear models are evaluated on a single run per fold, whereas neural models are
#   averaged over five random initialisations within each fold. Linear per-fold
#   estimates are therefore noisier, which reduces the sensitivity of the paired
#   tests in that setting.
# - Segments are defined by equal word counts rather than equal duration, so the
#   granularity axis operationalises position within the transcript rather than
#   position in time.

# 
