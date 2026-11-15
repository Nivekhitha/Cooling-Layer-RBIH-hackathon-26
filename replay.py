"""Time-aware scoring + Cooling Layer policy.

For every payment >= GATE we compute features using ONLY transactions with ts < t
(plus the payment itself), then a gradient-boosted model gives a fraud-flow risk score.
Payments below GATE settle instantly (a design choice: tiny payments are not worth a hold).
"""
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sim import simulate

GATE = 3000.0
WIN = 6 * 3600
DAY = 24 * 3600
LO = 0.85


class History:
    """Per-account sorted arrays so any 'as of time t' query is a binary search."""
    def __init__(self, txns, accounts):
        self.ins, self.outs = {}, {}
        for a, g in txns.groupby("dst"):
            self.ins[a] = (g.ts.to_numpy(), g.src.to_numpy(), g.amt.to_numpy())
        for a, g in txns.groupby("src"):
            self.outs[a] = (g.ts.to_numpy(), g.dst.to_numpy(), g.amt.to_numpy())
        first = txns.groupby(["src", "dst"]).ts.min()
        self.pair_first = first.to_dict()
        self.dev = dict(zip(accounts.acct, accounts.device))
        self.dev_members = accounts.groupby("device").acct.apply(list).to_dict()
        self._prior = {}

    def unrelated_on_device(self, a, t):
        n = 0
        for o in self.dev_members[self.dev[a]]:
            if o == a:
                continue
            if self.pair_first.get((a, o), 1e18) >= t and self.pair_first.get((o, a), 1e18) >= t:
                n += 1
        return n

    def burst(self, a, t):
        """distinct senders into a within the last 6h, including a payment at t."""
        if a not in self.ins:
            return 0
        ts, src, _ = self.ins[a]
        lo, hi = np.searchsorted(ts, t - WIN, "left"), np.searchsorted(ts, t, "right")
        return len(set(src[lo:hi]))

    def prior_fwd(self, a, t):
        """(n_big_inflows, forwarded_share, median_delay) over big inflows older than 30 min, as of t."""
        key = (a, int(t // 600))
        if key in self._prior:
            return self._prior[key]
        res = (0, 0.0, DAY)
        if a in self.ins:
            its, _, iam = self.ins[a]
            m = (iam >= GATE) & (its < t - 1800)
            if m.any():
                ots, _, oam = self.outs.get(a, (np.empty(0), None, np.empty(0)))
                delays = []
                for ti, ai in zip(its[m], iam[m]):
                    if len(ots):
                        k = (ots > ti) & (ots <= min(ti + DAY, t)) & (oam >= LO * ai) & (oam <= ai)
                        if k.any():
                            delays.append(ots[k].min() - ti)
                n = int(m.sum())
                res = (n, len(delays) / n, float(np.median(delays)) if delays else DAY)
        self._prior[key] = res
        return res

    def forwarding_now(self, a, t, amt):
        """does this payment forward a recent inflow? -> (flag, delay_sec, fraction_of_inflow)"""
        if a not in self.ins:
            return 0, DAY, 0.0
        its, _, iam = self.ins[a]
        lo, hi = np.searchsorted(its, t - DAY, "left"), np.searchsorted(its, t, "left")
        best = None
        for ti, ai in zip(its[lo:hi][::-1], iam[lo:hi][::-1]):
            if ai >= GATE and LO * ai <= amt <= ai:
                best = (1, t - ti, amt / ai)
                break
        return best if best else (0, DAY, 0.0)

    def big_partners(self, a, t):
        out = set()
        if a in self.ins:
            ts, s, am = self.ins[a]; k = np.searchsorted(ts, t, "left")
            out |= set(s[:k][am[:k] >= GATE])
        if a in self.outs:
            ts, d, am = self.outs[a]; k = np.searchsorted(ts, t, "left")
            out |= set(d[:k][am[:k] >= GATE])
        return out


FEATS = ["amt", "new_payee", "d_burst", "d_in_deg", "d_hist", "d_dev_unrel", "d_prior_share", "d_prior_pass",
         "d_nbr_fast", "s_fwd_flag", "s_fwd_delay", "s_fwd_frac", "s_prior_share", "s_burst", "s_dev_unrel"]
GRAPH_ONLY = ["d_dev_unrel", "s_dev_unrel", "d_nbr_fast"]


def txn_features(H, t, s, d, amt):
    f = {"amt": amt}
    f["new_payee"] = int(H.pair_first.get((s, d), 1e18) >= t)
    f["d_burst"] = H.burst(d, t)
    ts = H.ins[d][0]; k = np.searchsorted(ts, t, "right")
    f["d_in_deg"] = len(set(H.ins[d][1][:k]))
    f["d_hist"] = k + (np.searchsorted(H.outs[d][0], t, "left") if d in H.outs else 0)
    f["d_dev_unrel"] = H.unrelated_on_device(d, t)
    n, sh, med = H.prior_fwd(d, t)
    f["d_prior_share"], f["d_prior_pass"] = sh, med
    nb = [x for x in H.big_partners(d, t) if x != s]
    f["d_nbr_fast"] = float(np.mean([H.prior_fwd(x, t)[1] >= 0.5 and H.prior_fwd(x, t)[0] >= 1 for x in nb])) if nb else 0.0
    f["s_fwd_flag"], f["s_fwd_delay"], f["s_fwd_frac"] = H.forwarding_now(s, t, amt)
    f["s_prior_share"] = H.prior_fwd(s, t)[1]
    f["s_burst"] = H.burst(s, t) if s in H.ins else 0
    f["s_dev_unrel"] = H.unrelated_on_device(s, t)
    return f


def build_table(txns, accounts):
    H = History(txns, accounts)
    cand = txns[txns.amt >= GATE]
    rows = [dict(idx=i, **txn_features(H, r.ts, r.src, r.dst, r.amt)) for i, r in zip(cand.index, cand.itertuples())]
    T = pd.DataFrame(rows).set_index("idx")
    T = T.join(txns[["ts", "src", "dst", "kind", "chain", "stage", "part"]])
    T["fraud"] = (T.kind != "legit").astype(int)
    return T


def ring_group(T, rng):
    def g(r):
        for a in (r.src, r.dst):
            if a.startswith("R") and "_" in a:
                return int(a.split("_")[0][1:])
        return 1000 + int(rng.integers(0, 60))
    return np.array([g(r) for r in T.itertuples()])


def oof_scores(X, y, groups, n_splits=5):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200,
                                           class_weight="balanced", random_state=0).fit(X.iloc[tr], y[tr])
        s[te] = m.predict_proba(X.iloc[te])[:, 1]
    return s


# ---------------------------------------------------------------- policy
def decide(score, t_verify, t_hold):
    return np.where(score >= t_hold, "hold", np.where(score >= t_verify, "verify", "settle"))


def evaluate_policy(T, txns, score, t_verify, t_hold, p_confirm=1.0, p_verify=0.3, hold_minutes=30):
    """Expected outcome of the Cooling Layer on fraud chains (victim_in -> mule_hop -> cashout).

    ASSUMPTIONS (state these in the note):
      hold    : payment is parked for `hold_minutes`. With probability p_confirm the bank's review
                confirms fraud inside that window and freezes it; otherwise it is released, late.
                p_confirm=1.0 is the optimistic upper bound (every hold is confirmed).
      verify  : sender is warned + step-up check; stops the payment with probability p_verify.
      Payments below GATE are never scored and settle instantly. A chain whose payments are all
      below GATE therefore escapes completely.
    Value is measured as money that would reach the cash-out account. With split hops each part is
    followed on its own path, and the stopped value is attributed to the stage where it was stopped.
    """
    T = T.copy()
    T["score"], T["action"] = score, decide(score, t_verify, t_hold)
    T["p_stop"] = np.where(T.action == "hold", p_confirm, np.where(T.action == "verify", p_verify, 0.0))
    ps = T.p_stop.to_dict()

    n_all_legit = int((txns.kind == "legit").sum())
    leg, frd = T[T.fraud == 0], T[T.fraud == 1]
    out = {
        "legit_gated_total": len(leg),
        "legit_held": int((leg.action == "hold").sum()),
        "legit_verified": int((leg.action == "verify").sum()),
        "legit_held_gated_pct": 100 * (leg.action == "hold").mean(),
        "legit_verified_gated_pct": 100 * (leg.action == "verify").mean(),
        "legit_held_all_pct": 100 * (leg.action == "hold").sum() / n_all_legit,
        "legit_delay_minutes": int((leg.action == "hold").sum()) * hold_minutes,
        "fraud_gated_total": len(frd),
        "fraud_held": int((frd.action == "hold").sum()),
    }

    ch = txns[txns.chain >= 0]
    base = surv = ungated = chains_stopped = 0.0
    stopped = np.zeros(3)
    for c, g in ch.groupby("chain"):
        i0 = g.index[g.stage == 0][0]
        s0 = 1 - ps.get(i0, 0.0)
        cb = ca = 0.0
        for j, gp in g[g.stage > 0].groupby("part"):
            i1, i2 = gp.index[gp.stage == 1][0], gp.index[gp.stage == 2][0]
            v = float(txns.at[i2, "amt"])
            s1, s2 = 1 - ps.get(i1, 0.0), 1 - ps.get(i2, 0.0)
            stopped += v * np.array([1 - s0, s0 * (1 - s1), s0 * s1 * (1 - s2)])
            cb += v
            ca += v * s0 * s1 * s2
        base += cb
        surv += ca
        chains_stopped += 1 - ca / cb
        if not any(i in ps for i in g.index):
            ungated += cb
    n_chain = ch.chain.nunique()
    out.update(chains=n_chain, chains_stopped_exp=chains_stopped,
               value_reduction_pct=100 * (1 - surv / base),
               stopped_at_victim_pct=100 * stopped[0] / base,
               stopped_at_first_hop_pct=100 * stopped[1] / base,
               stopped_at_cashout_pct=100 * stopped[2] / base,
               value_never_scored_pct=100 * ungated / base)
    return out, T


# ---------------------------------------------------------------- base-rate adjustment
def wilson_upper(k, n, z=1.96):
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c + r) / d


def prevalence_rows(tpr, fpr, prevs=(0.196, 0.01, 0.001, 0.0001), n=100_000):
    """What the same hold rule looks like if fraud is a smaller share of large payments.
    Precision = share of holds that are real fraud. (tpr/fpr are measured on the simulator.)"""
    rows = []
    for pi in prevs:
        tp, fp = tpr * pi, fpr * (1 - pi)
        rows.append(dict(fraud_share=pi, hold_precision=tp / (tp + fp), holds_per_100k=n * (tp + fp),
                         legit_held_per_100k=n * fp))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- reason codes
def reasons(r):
    out = []
    if r.s_fwd_flag:
        out.append(f"Sender is forwarding {r.s_fwd_frac:.0%} of an inflow received {r.s_fwd_delay/60:.0f} min ago")
    if r.d_burst >= 4:
        out.append(f"Receiver got money from {int(r.d_burst)} distinct senders in the last 6h")
    if r.d_prior_share >= 0.5:
        out.append(f"Receiver has forwarded {r.d_prior_share:.0%} of past large inflows (median {r.d_prior_pass/60:.0f} min)")
    if r.d_nbr_fast >= 0.5:
        out.append(f"{r.d_nbr_fast:.0%} of receiver's large-payment counterparties are fast pass-through accounts")
    if r.d_dev_unrel >= 1 or r.s_dev_unrel >= 1:
        out.append(f"Shares a device with {int(max(r.d_dev_unrel, r.s_dev_unrel))} account(s) it has no payment history with")
    if r.new_payee and not out:
        out.append("First payment to this payee")
    return out or ["Elevated combined risk score"]


if __name__ == "__main__":
    # single-seed quick run; use run_study.py for the multi-seed / adversarial study
    txns, acc = simulate()
    T = build_table(txns, acc)
    y = T.fraud.to_numpy()
    g = ring_group(T, np.random.default_rng(0))
    s = oof_scores(T[FEATS], y, g)
    from sklearn.metrics import average_precision_score, roc_auc_score
    print(f"{len(T)} gated payments, {T.fraud.sum()} fraud-flow ({T.fraud.mean():.1%})")
    print(f"txn-level ROC-AUC {roc_auc_score(y, s):.3f}  PR-AUC {average_precision_score(y, s):.3f}")
    for pc in (1.0, 0.8, 0.5):
        o, _ = evaluate_policy(T, txns, s, 0.5, 0.9, p_confirm=pc)
        print(f"p_confirm={pc}: value stopped {o['value_reduction_pct']:.1f}%  legit held {o['legit_held']}/{o['legit_gated_total']}")
