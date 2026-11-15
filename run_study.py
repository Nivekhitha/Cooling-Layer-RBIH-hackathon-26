"""Multi-seed, multi-variant study of the Cooling Layer.

Thresholds are FIXED in advance (verify >= 0.5, hold >= 0.9; chosen on seed 7 of the base
simulator only), so every other seed/variant is out-of-sample for the thresholds.
Model scores are out-of-fold, grouped by ring, so a ring never appears in both train and test.
"""
import sys, time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from sim import simulate
from replay import (build_table, ring_group, oof_scores, evaluate_policy, FEATS, GRAPH_ONLY,
                    wilson_upper, prevalence_rows)

warnings.filterwarnings("ignore")
SEEDS = [7, 11, 13, 17, 19]
VARIANTS = ["base", "slow", "split", "structure"]
TV, TH = 0.5, 0.9

rows = []
for variant in VARIANTS:
    for seed in SEEDS:
        t0 = time.time()
        txns, acc = simulate(seed=seed, variant=variant)
        T = build_table(txns, acc)
        y = T.fraud.to_numpy()
        g = ring_group(T, np.random.default_rng(0))
        s = oof_scores(T[FEATS], y, g)
        s_ng = oof_scores(T[[f for f in FEATS if f not in GRAPH_ONLY]], y, g)
        r = dict(variant=variant, seed=seed, gated=len(T), fraud_share_gated=y.mean(),
                 pr_auc=average_precision_score(y, s), pr_auc_no_graph=average_precision_score(y, s_ng))
        for pc in (1.0, 0.8, 0.5):
            o, _ = evaluate_policy(T, txns, s, TV, TH, p_confirm=pc)
            r[f"value_stopped_pc{pc}"] = o["value_reduction_pct"]
        o, _ = evaluate_policy(T, txns, s, TV, TH, p_confirm=1.0)
        r.update({k: o[k] for k in ["legit_gated_total", "legit_held", "legit_verified", "fraud_gated_total", "fraud_held",
                                    "chains", "chains_stopped_exp", "stopped_at_victim_pct", "stopped_at_first_hop_pct",
                                    "stopped_at_cashout_pct", "value_never_scored_pct"]})
        rows.append(r)
        print(f"{variant:9s} seed {seed:2d} done in {time.time()-t0:4.0f}s  value stopped (p_confirm=1) {r['value_stopped_pc1.0']:.1f}%", flush=True)

R = pd.DataFrame(rows)
R.to_csv("study_results.csv", index=False)

def ms(col):
    return R.groupby("variant")[col].agg(["mean", "std"]).reindex(VARIANTS)

print("\n=== value stopped (%), mean +- sd over seeds ===")
for pc in ("1.0", "0.8", "0.5"):
    m = ms(f"value_stopped_pc{pc}")
    print(f"p_confirm={pc}: " + "  ".join(f"{v}: {m.loc[v,'mean']:.1f}+-{m.loc[v,'std']:.1f}" for v in VARIANTS))
print("\n=== where value is stopped, % of cash-out-bound value (p_confirm=1.0) ===")
for col in ["stopped_at_victim_pct", "stopped_at_first_hop_pct", "stopped_at_cashout_pct", "value_never_scored_pct"]:
    m = ms(col); print(f"{col:26s} " + "  ".join(f"{v}: {m.loc[v,'mean']:.1f}" for v in VARIANTS))
print("\n=== PR-AUC with / without graph+device features ===")
for col in ["pr_auc", "pr_auc_no_graph"]:
    m = ms(col); print(f"{col:16s} " + "  ".join(f"{v}: {m.loc[v,'mean']:.3f}+-{m.loc[v,'std']:.3f}" for v in VARIANTS))

B = R[R.variant == "base"]
k, n = int(B.legit_held.sum()), int(B.legit_gated_total.sum())
fpr = k / n; tpr = B.fraud_held.sum() / B.fraud_gated_total.sum()
print(f"\nbase variant pooled over {len(B)} seeds: legit held {k}/{n} = {fpr:.3%} (95% upper bound {wilson_upper(k, n):.2%}); "
      f"fraud payments held {tpr:.1%}")
print("fraud share of large payments in the simulator:", f"{B.fraud_share_gated.mean():.1%}")
print(prevalence_rows(tpr, fpr).round(4).to_string(index=False))
print("\nsame table using the 95% upper bound on the false-hold rate:")
print(prevalence_rows(tpr, wilson_upper(k, n)).round(4).to_string(index=False))
print("\ntotal legit gated payments pooled:", n, "| holds:", k)
