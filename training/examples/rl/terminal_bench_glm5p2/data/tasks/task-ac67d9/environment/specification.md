# FairMix Fairness Metric — Mathematical Specification

## Problem Overview

Optimize a vector of 96 local peer-to-peer (P2P) electricity prices for a Local Energy Community (LEC) of 10 households over a 24-hour horizon (15-minute resolution). The objective is to minimize the **FairMix** composite fairness metric, which balances benefit coverage, equity of gains, and loss moderation.

## Community Structure

| Household | Type | PV Peak (kW) |
|-----------|------|-------------|
| 0–1 | Prosumer | 2.0, 2.5 |
| 2–3 | Prosumer | 4.0, 4.5 |
| 4–5 | Prosumer | 6.0, 6.5 |
| 6–9 | Consumer | 0.0 |

All energy flows (grid import, grid export, P2P buy, P2P sell) are pre-computed by a MILP dispatch and **fixed**. Only prices change.

## Decision Variable

A price vector **p** of length 96:

    p = [p_0, p_1, ..., p_95]

where p_t is the local P2P electricity price (EUR/kWh) at timestep t.

**Bounds:** feed_in_tariff[t] ≤ p[t] ≤ grid_tariff[t] for all t ∈ {0, ..., 95}.

## Bill Computation

For household h, the total bill given price vector **p**:

    bill_h(p) = Σ_t [ grid_import[h,t] × grid_tariff[t]
                     − grid_export[h,t] × feed_in_tariff[t]
                     + (p2p_buy[h,t] − p2p_sell[h,t]) × p[t] ]

## Business-as-Usual (BAU) Reference

The BAU bill uses the mid-market price:

    mid_price[t] = (grid_tariff[t] + feed_in_tariff[t]) / 2
    bau_bill_h = bill_h(mid_price)

## Household Savings

    savings_h = bau_bill_h − bill_h(p)

Positive savings = household benefits from optimized pricing.

**Key property:** Total savings sum to zero (P2P is zero-sum), so the optimizer redistributes costs rather than reducing them globally.

## FairMix Composite Metric

### Component 1 — Benefit Coverage (C1)

    N = total number of households (10)
    N⁺ = count of households with savings_h > 0

    C1 = 1 − N⁺ / N

Measures what fraction of the community does NOT benefit. Lower is better.

### Component 2 — Equity Balance (C2)

Coefficient of variation among benefiting households, using **population** standard deviation (ddof=0):

    If N⁺ ≥ 2:
        positive_savings = {savings_h : savings_h > 0}
        C2 = std(positive_savings) / mean(positive_savings)
    Else:
        C2 = 0

Measures how unevenly benefits are distributed among winners. Lower is better.

### Component 3 — Loss Burden (C3)

Mean absolute loss normalized by mean BAU bill:

    If any savings_h < 0:
        losses = {|savings_h| : savings_h < 0}
        C3 = mean(losses) / mean(bau_bills)
    Else:
        C3 = 0

Measures the severity of losses relative to typical household costs. Lower is better.

### Composite Score

    FairMix = (C1 + C2 + C3) / 3

**Minimize** FairMix. The theoretical minimum is 0 (perfect fairness). The mid-market baseline gives FairMix = 1/3 ≈ 0.333.

## Optimizer Specification

Implement a self-adaptive Differential Evolution (DE) variant with:

1. **Population**: NP individuals, each a 96-dimensional price vector.
2. **Self-adaptive parameters**: Per-individual F (3 columns: F1, F2, F3) and CR, each regenerated with 10% probability per generation. When regenerated, F_i ~ Uniform(0.1, 1.0) and CR_i ~ Uniform(0, 1).
3. **HyDE-DF mutation**: For individual x with best solution x_best and random individuals r1, r2:

        a = (max_iter − gen) / max_iter
        decay = exp(1 − 1/a²)     [a > 0; decay = 0 when a = 0]

        v = x + F3 × (x_r1 − x_r2) + decay × F1 × (x_best × (F2 + randn()) − x)

4. **Binomial crossover**: Each dimension crossed with probability CR; at least one dimension always crossed (j_rand).
5. **Boundary repair**: Clamp to violated bound.
6. **Selection**: Greedy — keep trial if fitness < parent fitness; update F, CR for improved individuals.
7. **Budget**: Maximum 5,000 function evaluations per trial. Evaluations count each call to the fitness function.
