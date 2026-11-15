# Cooling Layer

Risk-scored dynamic settlement against mule-account fraud. Built for the St Aloysius - RBIH FinTech Innovation Hackathon 2026
(track: Digital Payments Fraud Detection and Prevention). **Prototype on synthetic data. Not a production system.**

## Idea
Every payment of Rs 3,000 or more is scored using only information available before that moment. A policy then chooses:

| Score | Action | Assumption in the evaluation |
|---|---|---|
| below 0.5 | Settle instantly | none |
| 0.5 to 0.9 | Verify (sender warned, step-up check) | stops the payment 30% of the time |
| 0.9 or above | Hold 30 minutes for bank review | review confirms and freezes fraud with probability `p_confirm` |

Payments under Rs 3,000 are never scored and settle instantly (a known weakness, see Limitations).

## Files
| File | What it does |
|---|---|
| `sim.py` | Synthetic UPI-style network: legit traffic, hard negatives (merchants, collectors, fast legit pass-through agents), mule rings. Variants: `base`, `slow`, `split`, `structure` |
| `replay.py` | Time-aware features (computed only from earlier transactions), gradient-boosted scorer, policy, evaluation, reason codes |
| `run_study.py` | 5 seeds x 4 variants, thresholds fixed in advance, out-of-fold scores grouped by ring |
| `results/` | Output of the study (`study_results.csv`, `study.log`) |
| `docs/` | Concept note (PDF) |

## Run
```bash
pip install -r requirements.txt
python replay.py        # quick single-seed check (about 10 seconds)
python run_study.py     # full study (a few minutes), writes study_results.csv
```

## Results (mean over 5 seeds, simulator only)
Share of cash-out-bound value stopped, by how often a held payment is confirmed as fraud by bank review:

| Scenario | all holds confirmed | 80% confirmed | 50% confirmed | never scored |
|---|---|---|---|---|
| Base | 96.7% | 93.9% | 80.3% | 0% |
| Slow hops | 98.6% | 95.3% | 81.0% | 0% |
| Split hops | 98.1% | 94.7% | 80.1% | 0% |
| Structured (below gate) | 72.4% | 70.4% | 60.4% | 22.9% |

Base scenario: 0.49% of legitimate large payments held (28 of 5,721; 95% upper bound 0.71%); about 69% of value stopped at the victim's own payment.

## Limitations (please read)
- **Synthetic data only.** The simulator, fraud patterns and features were designed by the same person; results show the mechanism works, not that it transfers to real traffic.
- **Assumed outcomes.** Review confirmation and the 30% verification effect are assumptions.
- **Base rate.** Fraud is about 21% of large payments in the simulator. At 1% real fraud, about 61% of holds are real fraud; at 0.01%, about 1.5%.
- **Fixed gate.** Chunking payments below Rs 3,000 evades scoring (23% of value in the structured scenario). A cumulative-inflow gate is planned, not built.
- **Graph features are marginal here.** PR-AUC 0.963 with vs 0.955 without graph/device features (within seed noise).
- Feature computation took about 0.1 ms per payment in-process; this is not an end-to-end latency claim.

## Related work
RBIH's MuleHunter.AI (announced December 2024) focuses on detecting mule accounts. This project asks a neighbouring question: what to do with a payment in flight.
It does not use or replicate MuleHunter.AI.
