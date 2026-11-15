"""Plot helpers for the interactive demo."""
import re

import matplotlib.pyplot as plt
import numpy as np

from replay import evaluate_policy

COL = {"hold": "#d62728", "verify": "#ffbf00", "settle": "#2ca02c", "ungated": "#888888"}


def tradeoff(T, txns, scores, t_verify, p_confirm, p_verify):
    """Evaluate the policy over hold thresholds for the selected verification threshold."""
    hold_thresholds = np.linspace(max(t_verify, 0.1), 0.99, 30)
    rows = []
    for t_hold in hold_thresholds:
        result, _ = evaluate_policy(
            T, txns, scores, t_verify, t_hold, p_confirm=p_confirm, p_verify=p_verify
        )
        rows.append(
            (result["legit_held_gated_pct"], result["value_reduction_pct"])
        )
    return np.asarray(rows)


def tradeoff_figure(curve, selected):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve[:, 0], curve[:, 1], marker=".", linewidth=1.5, label="Hold-threshold sweep")
    ax.scatter(*selected, color="#1f77b4", edgecolor="white", s=70, zorder=3, label="Selected policy")
    ax.set_xlabel("Legitimate large payments held (%)")
    ax.set_ylabel("Cash-out-bound value stopped (%)")
    ax.set_title("Policy trade-off")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def ring_ids(txns):
    rings = set()
    for account in set(txns.src) | set(txns.dst):
        match = re.match(r"R(\d+)_", account)
        if match:
            rings.add(int(match.group(1)))
    return sorted(rings)


def ring_of(src, dst):
    for account in (src, dst):
        match = re.match(r"R(\d+)_", account)
        if match:
            return int(match.group(1))
    return None


def _node_position(account, ring, y_by_node):
    if account.startswith(f"R{ring}_a"):
        x = 0.30
    elif account.startswith(f"R{ring}_b"):
        x = 0.68
    elif account == f"R{ring}_cash":
        x = 0.94
    else:
        x = 0.05 if account.startswith("U") else 1.0
    return x, y_by_node[account]


def ring_figure(txns, actions, ring):
    """Draw payments associated with one simulated mule ring."""
    ring_txns = txns[
        [ring_of(src, dst) == ring for src, dst in zip(txns.src, txns.dst)]
    ]
    nodes = sorted(set(ring_txns.src) | set(ring_txns.dst))
    y_by_node = {node: i for i, node in enumerate(nodes)}
    fig, ax = plt.subplots(figsize=(10, max(4, len(nodes) * 0.35)))

    for idx, row in ring_txns.iterrows():
        src_pos = _node_position(row.src, ring, y_by_node)
        dst_pos = _node_position(row.dst, ring, y_by_node)
        action = actions.get(idx, "ungated")
        color = COL.get(action, COL["ungated"])
        ax.annotate(
            "",
            xy=dst_pos,
            xytext=src_pos,
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "alpha": 0.7,
                "lw": 1.5 if action != "ungated" else 1.0,
                "linestyle": "--" if action == "ungated" else "-",
                "connectionstyle": "arc3,rad=0.08",
            },
        )

    for node in nodes:
        x, y = _node_position(node, ring, y_by_node)
        ax.scatter(x, y, color="#555555", s=70, zorder=3)
        ax.text(x, y + 0.12, node, ha="center", va="bottom", fontsize=8)

    ax.set_title(f"Payments associated with ring {ring}")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.5, max(len(nodes) - 0.5, 0.5))
    ax.axis("off")
    fig.tight_layout()
    return fig
