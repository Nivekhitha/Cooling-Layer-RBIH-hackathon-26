"""Synthetic UPI-style transaction simulator.

Produces (txns, accounts):
  txns:     ts (sec), src, dst, amt, kind  (kind: legit | victim_in | mule_hop | cashout)
  accounts: acct, device, is_mule, role

variant (how the mule operators behave):
  base       as originally designed
  slow       every ring waits 4-20h between hops (looks like legit slow pass-through)
  split      each hop is split into 2-3 partial transfers (no single forward matches the inflow)
  structure  half of the victims are drained in chunks under the Rs 2,900 mark (below the scoring gate)

Hard negatives are included on purpose so the model cannot win by just
flagging "high fan-in": busy merchants, and chit-fund style collectors that
receive from many people but pay out slowly.
"""
import numpy as np
import pandas as pd


def simulate(seed=7, n_users=3000, n_merchants=60, n_collectors=15, n_rings=14, days=7, hard=True, variant="base"):
    rng = np.random.default_rng(seed)
    H = days * 86400
    users = [f"U{i}" for i in range(n_users)]
    merchants = [f"M{i}" for i in range(n_merchants)]
    collectors = [f"C{i}" for i in range(n_collectors)]

    # ---- devices: ~8% of users share a device with an earlier user (family phones)
    dev = {}
    for i, u in enumerate(users):
        if i > 0 and rng.random() < 0.08:
            dev[u] = dev[users[rng.integers(0, i)]]
        else:
            dev[u] = f"D{i}"
    for a in merchants + collectors:
        dev[a] = f"D{a}"

    rows = []  # (ts, src, dst, amt, kind)

    # ---- legit traffic: merchants (zipf popularity) + a few personal contacts
    pop = 1.0 / np.arange(1, n_merchants + 1)
    pop /= pop.sum()
    for u in users:
        contacts = rng.choice(users, size=5, replace=False)
        for _ in range(rng.poisson(25)):
            ts = rng.uniform(0, H)
            amt = float(np.round(rng.lognormal(5.5, 1.0), 2))
            if rng.random() < 0.7:
                dst = merchants[rng.choice(n_merchants, p=pop)]
            else:
                dst = contacts[rng.integers(0, 5)]
                if dst == u:
                    continue
            rows.append((ts, u, dst, amt, "legit"))

    # ---- hard negative: collectors (chit fund / society dues): many senders, slow single payout
    for c in collectors:
        senders = rng.choice(users, size=rng.integers(15, 30), replace=False)
        t_last = 0
        for s in senders:
            ts = rng.uniform(0, H * 0.5)
            t_last = max(t_last, ts)
            rows.append((ts, s, c, 2000.0, "legit"))
        total = 2000.0 * len(senders)
        rows.append((t_last + rng.uniform(86400, 2 * 86400), c, users[rng.integers(0, n_users)], total, "legit"))

    # ---- mule rings
    mule_accts, roles = {}, {}
    chain_key, cid = {}, 0
    for r in range(n_rings):
        stealthy = rng.random() < 0.3          # slower, smaller rings: harder to catch
        n_t1, n_t2 = (2, 1) if stealthy else (3, 2)
        t1 = [f"R{r}_a{k}" for k in range(n_t1)]
        t2 = [f"R{r}_b{k}" for k in range(n_t2)]
        cash = f"R{r}_cash"
        ring_dev = f"DRING{r}"
        for a, role in [(x, "tier1") for x in t1] + [(x, "tier2") for x in t2] + [(cash, "cashout")]:
            mule_accts[a] = True
            roles[a] = role
            # operator reuses the ring device ~80% of the time
            dev[a] = ring_dev if rng.random() < (0.45 if hard else 0.8) else f"D{a}"

        t0 = rng.uniform(0, H * 0.8)
        n_victims = int(rng.integers(3, 6)) if stealthy else int(rng.integers(6, 13))
        victims = rng.choice(users, size=n_victims, replace=False)
        pass_lo, pass_hi = (1800, 14400) if stealthy else (60, 900)   # seconds
        if variant == "slow":
            pass_lo, pass_hi = 4 * 3600, 20 * 3600
        for v in victims:
            structured = variant == "structure" and rng.random() < 0.5
            n_chunks = 1
            if structured:
                total = float(rng.uniform(5000, 15000))
                n_chunks = int(np.ceil(total / 2900))
            for _ in range(n_chunks):
                ts = t0 + rng.uniform(0, 4 * 3600)
                amt = float(np.round(total / n_chunks, 2)) if structured else float(np.round(rng.uniform(5000, 60000), 2))
                m1 = t1[rng.integers(0, n_t1)]
                rows.append((ts, v, m1, amt, "victim_in")); chain_key[(ts, v, m1)] = (cid, 0, 0)
                # layered forwarding: m1 -> m2 -> cashout, keeping ~3-8% as "commission"
                ts2 = ts + rng.uniform(pass_lo, pass_hi)
                amt2 = amt * rng.uniform(0.92, 0.97)
                m2 = t2[rng.integers(0, n_t2)]
                if variant == "split":
                    k = int(rng.integers(2, 4))
                    parts = list(np.round(amt2 * rng.dirichlet(np.ones(k)), 2))
                else:
                    parts = [amt2]
                for j, p in enumerate(parts):
                    tsj = ts2 + j * rng.uniform(5, 120) if variant == "split" else ts2
                    rows.append((tsj, m1, m2, float(p), "mule_hop")); chain_key[(tsj, m1, m2)] = (cid, 1, j)
                    ts3 = tsj + rng.uniform(pass_lo, pass_hi)
                    rows.append((ts3, m2, cash, float(p) * rng.uniform(0.95, 0.99), "cashout")); chain_key[(ts3, m2, cash)] = (cid, 2, j)
                cid += 1
        # mules also make a few normal-looking payments to blend in
        for m in t1 + t2:
            for _ in range(rng.integers(1, 4)):
                rows.append((rng.uniform(0, H), m, merchants[rng.integers(0, n_merchants)],
                             float(np.round(rng.lognormal(5.0, 0.7), 2)), "legit"))


    if hard:
        # ---- hard negative: legit "pass-through" users (resellers, agents, brokers).
        # They receive large payments from several different people and forward most of it quickly.
        for k in range(40):
            agent = users[rng.integers(0, n_users)]
            fast = rng.random() < 0.5
            for _ in range(int(rng.integers(4, 14))):
                ts = rng.uniform(0, H * 0.95)
                amt = float(np.round(rng.uniform(4000, 50000), 2))
                rows.append((ts, users[rng.integers(0, n_users)], agent, amt, "legit"))
                lag = rng.uniform(120, 3600) if fast else rng.uniform(4 * 3600, 20 * 3600)
                rows.append((ts + lag, agent, merchants[rng.integers(0, n_merchants)] if rng.random() < 0.5
                             else users[rng.integers(0, n_users)], amt * rng.uniform(0.9, 0.99), "legit"))
        # ---- mules are real people's accounts too: ordinary inbound AND outbound activity
        for m in list(mule_accts):
            for _ in range(int(rng.poisson(20))):
                rows.append((rng.uniform(0, H), m, merchants[rng.choice(n_merchants, p=pop)]
                             if rng.random() < 0.7 else users[rng.integers(0, n_users)],
                             float(np.round(rng.lognormal(5.5, 1.0), 2)), "legit"))
        for m in list(mule_accts):
            for _ in range(int(rng.integers(5, 25))):
                rows.append((rng.uniform(0, H), users[rng.integers(0, n_users)], m,
                             float(np.round(rng.lognormal(5.5, 1.0), 2)), "legit"))

    txns = pd.DataFrame(rows, columns=["ts", "src", "dst", "amt", "kind"])
    txns = txns[txns.ts < H].sort_values("ts").reset_index(drop=True)

    ck = [chain_key.get((a, b, c), (-1, -1, -1)) for a, b, c in zip(txns.ts, txns.src, txns.dst)]
    txns["chain"] = [x[0] for x in ck]
    txns["stage"] = [x[1] for x in ck]
    txns["part"] = [x[2] for x in ck]
    all_accts = sorted(set(txns.src) | set(txns.dst))
    accounts = pd.DataFrame({"acct": all_accts})
    accounts["device"] = accounts.acct.map(dev)
    accounts["is_mule"] = accounts.acct.map(lambda a: int(a in mule_accts))
    accounts["role"] = accounts.acct.map(lambda a: roles.get(a, "normal"))
    return txns, accounts


if __name__ == "__main__":
    t, a = simulate()
    print(t.kind.value_counts().to_string())
    print(f"\n{len(a)} accounts, {a.is_mule.sum()} mules ({a.is_mule.mean():.1%})")
