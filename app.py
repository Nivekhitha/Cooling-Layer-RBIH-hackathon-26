"""Cooling Layer - interactive demo (simulated data only).

Run locally:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from sim import simulate
from replay import build_table, ring_group, oof_scores, evaluate_policy, reasons, FEATS
from viz import tradeoff, tradeoff_figure, ring_figure, ring_ids, ring_of

st.set_page_config(page_title="Cooling Layer demo", layout="wide")


@st.cache_data(show_spinner="Simulating the network and scoring payments (about 10-20 seconds the first time)...")
def load(variant, seed):
    txns, acc = simulate(seed=seed, variant=variant)
    T = build_table(txns, acc)
    y = T.fraud.to_numpy()
    s = oof_scores(T[FEATS], y, ring_group(T, np.random.default_rng(0)))
    return txns, T, s


def when(ts):
    return f"day {int(ts // 86400) + 1}, {int(ts % 86400 // 3600):02d}:{int(ts % 3600 // 60):02d}"


st.title("Cooling Layer")
st.caption("Risk-scored settlement against mule-account fraud: fast for honest payments, slow for mule money.")
st.warning(
    "**Prototype on simulated data.** Not a real system, and not connected to any bank or payment network. "
    "Review and verification outcomes below are assumptions you can change.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Scenario")
    VARIANTS = {
        "Base": "base",
        "Slow hops (4-20 h)": "slow",
        "Split hops (2-3 partial forwards)": "split",
        "Structured (chunks under Rs 2,900)": "structure",
    }
    variant = VARIANTS[st.selectbox("How do the mule operators behave?", list(VARIANTS))]
    seed = st.selectbox("Simulation seed", [7, 11, 13, 17, 19])
    st.header("Policy")
    t_verify = st.slider("Verify at score >=", 0.05, 0.95, 0.5, 0.05)
    t_hold = st.slider("Hold at score >=", 0.10, 0.99, 0.9, 0.01)
    st.header("Assumptions")
    p_confirm = st.slider("Bank review confirms a held fraud payment (probability)", 0.0, 1.0, 1.0, 0.05)
    p_verify = st.slider("Sender verification stops a fraud payment (probability)", 0.0, 1.0, 0.3, 0.05)

if t_verify > t_hold:
    st.sidebar.error("Verify threshold is above the hold threshold; using the hold value for both.")
    t_verify = t_hold

txns, T, s = load(variant, seed)
out, TS = evaluate_policy(T, txns, s, t_verify, t_hold, p_confirm=p_confirm, p_verify=p_verify)
action = TS.action.to_dict()

tab1, tab2, tab3, tab4 = st.tabs(["Policy trade-off", "Inspect a payment", "Ring view", "About and limits"])

# ------------------------------------------------------------------ tab 1
with tab1:
    c = st.columns(4)
    c[0].metric("Cash-out value stopped", f"{out['value_reduction_pct']:.1f}%")
    c[1].metric(
        "Legit large payments held",
        f"{out['legit_held']} of {out['legit_gated_total']}",
        f"{out['legit_held_gated_pct']:.2f}%",
        delta_color="off",
    )
    c[2].metric(
        "Legit sent to verification",
        f"{out['legit_verified']}",
        f"{out['legit_verified_gated_pct']:.2f}%",
        delta_color="off",
    )
    c[3].metric(
        "Value never scored",
        f"{out['value_never_scored_pct']:.1f}%",
        help="Chains whose payments are all below the Rs 3,000 scoring gate.",
    )
    left, right = st.columns([1.1, 1])
    with left:
        curve = tradeoff(T, txns, s, t_verify, p_confirm, p_verify)
        st.pyplot(
            tradeoff_figure(curve, (out["legit_held_gated_pct"], out["value_reduction_pct"])),
            clear_figure=True,
        )
    with right:
        st.markdown("**Where the stopped value is stopped**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Stage": ["At the victim's payment", "At the first mule hop", "At the cash-out hop"],
                    "% of cash-out-bound value": [
                        round(out["stopped_at_victim_pct"], 1),
                        round(out["stopped_at_first_hop_pct"], 1),
                        round(out["stopped_at_cashout_pct"], 1),
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption("Earlier is better: money stopped at the victim's payment never reaches a mule account.")
        with st.expander("Base-rate check: what if real fraud is much rarer?"):
            SHARES = {"21% (simulator)": 0.21, "5%": 0.05, "1%": 0.01, "0.1%": 0.001, "0.01%": 0.0001}
            share = SHARES[
                st.select_slider("Fraud share of large payments", options=list(SHARES), value="1%")
            ]
            tpr = out["fraud_held"] / max(out["fraud_gated_total"], 1)
            fpr = out["legit_held"] / max(out["legit_gated_total"], 1)
            denominator = tpr * share + fpr * (1 - share)
            prec = tpr * share / denominator if denominator > 0 else float("nan")
            st.write(
                f"With this policy, **about {prec:.0%}** of held payments would be real fraud. "
                "The simulator has far more fraud than real life, so most holds at realistic rates would be false alarms."
            )

# ------------------------------------------------------------------ tab 2
with tab2:
    st.write("Pick a scored payment to see the decision and the reasons an investigator would get.")
    mode = st.radio(
        "Show",
        ["Highest-risk payments", "A random sample", "Fraud payments the policy missed"],
        horizontal=True,
    )
    D = TS.copy()
    if mode == "Highest-risk payments":
        D = D.sort_values("score", ascending=False).head(40)
    elif mode == "A random sample":
        D = D.sample(min(40, len(D)), random_state=1).sort_values("score", ascending=False)
    else:
        D = D[(D.fraud == 1) & (D.action == "settle")].sort_values("score", ascending=False).head(40)
    if D.empty:
        st.info("None in this scenario: no fraud payment was settled at these thresholds.")
    else:
        labels = {
            i: (
                f"{when(r.ts)} | {r.src} to {r.dst} | Rs {r.amt:,.0f} | "
                f"score {r.score:.2f} | {r.action.upper()}"
            )
            for i, r in D.iterrows()
        }
        by_label = {v: k for k, v in labels.items()}
        pick = by_label[st.selectbox("Payment", list(by_label))]
        r = TS.loc[pick]
        a, b = st.columns([1, 1.4])
        with a:
            st.markdown(f"### {r.action.upper()}")
            st.write(
                f"**Risk score:** {r.score:.2f}  \n**Amount:** Rs {r.amt:,.0f}  \n"
                f"**When:** {when(r.ts)}  \n**From / to:** {r.src} to {r.dst}"
            )
            msg = {
                "settle": "Settled instantly.",
                "verify": "Sender is warned and asked for a step-up check before the payment goes through.",
                "hold": "Payment is parked for 30 minutes while the bank reviews it.",
            }[r.action]
            st.info(msg)
        with b:
            st.markdown("**Reason codes**")
            row = TS.loc[[pick]].itertuples().__next__()
            for line in reasons(row):
                st.write("- " + line)
            with st.expander("Reveal the simulation's label"):
                if r.chain >= 0:
                    st.write(
                        f"This payment is stage {int(r.stage) + 1} of 3 in a simulated mule chain "
                        "(victim, first hop, cash-out)."
                    )
                else:
                    st.write("This is a legitimate payment in the simulation.")

# ------------------------------------------------------------------ tab 3
with tab3:
    rings = ring_ids(txns)
    ring = int(st.selectbox("Simulated mule ring", [f"Ring {r}" for r in rings]).split()[1])
    st.pyplot(ring_figure(txns, action, ring), clear_figure=True)
    sub = TS[[ring_of(a, b) == ring for a, b in zip(TS.src, TS.dst)] if len(TS) else []]
    sub = sub[sub.chain >= 0]
    st.write(
        "Payments in this ring that were scored: "
        + ", ".join(f"{int(n)} {k}" for k, n in sub.action.value_counts().items())
        + "."
    )
    st.caption(
        "Each arrow is a payment. Red = held, amber = verify, green = settled, "
        "grey dashed = below the scoring gate (never scored)."
    )

# ------------------------------------------------------------------ tab 4
with tab4:
    st.markdown(
        """
**What this is.** A prototype of a settlement layer: each payment of Rs 3,000 or more gets a risk score from information available *before* that moment, and a policy chooses settle, verify or hold.

**What the numbers are.** All results come from a simulator written for this project, with a gradient-boosted model scored out-of-fold (a ring never appears in both training and test).

**Limitations**
- Synthetic data only; the simulator, fraud patterns and features were designed by the same person.
- Review confirmation and the verification effect are assumptions (sliders in the sidebar).
- Fraud is about 21% of large payments here, far above real life (see the base-rate check).
- Payments under Rs 3,000 are never scored; chunking below it escapes (try the Structured scenario).
- Graph and device features add little in this simulator (PR-AUC 0.963 with, 0.955 without).

Source code: see the project repository.
"""
    )
