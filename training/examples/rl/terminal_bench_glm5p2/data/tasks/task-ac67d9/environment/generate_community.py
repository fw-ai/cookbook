#!/usr/bin/env python3
"""Generate deterministic Local Energy Community data for fair pricing optimization."""
import json
import math
import random
import os

random.seed(42)

N_HH = 10
N_T = 96

# Grid tariff (EUR/kWh) - time-of-use pattern
grid_tariff = []
for t in range(N_T):
    hour = t * 0.25
    if hour < 6:
        grid_tariff.append(0.08)
    elif hour < 9:
        grid_tariff.append(0.12)
    elif hour < 12:
        grid_tariff.append(0.16)
    elif hour < 17:
        grid_tariff.append(0.14)
    elif hour < 21:
        grid_tariff.append(0.19)
    else:
        grid_tariff.append(0.10)

# Feed-in tariff (flat)
feed_in_tariff = [0.04] * N_T

# PV peak capacities (kW) - households 0-5 are prosumers, 6-9 are consumers
pv_peak = [2.0, 2.5, 4.0, 4.5, 6.0, 6.5, 0.0, 0.0, 0.0, 0.0]

# Solar irradiance profile (normalized bell curve)
solar = []
for t in range(N_T):
    h = t * 0.25
    if 6 <= h <= 18:
        solar.append(math.sin(math.pi * (h - 6) / 12) ** 1.5)
    else:
        solar.append(0.0)

# PV generation per household (kWh per 15 min)
pv_generation = []
for hh in range(N_HH):
    row = []
    for t in range(N_T):
        if pv_peak[hh] > 0 and solar[t] > 0:
            cloud = 0.80 + 0.20 * random.random()
            val = pv_peak[hh] * 0.25 * solar[t] * cloud
            row.append(round(val, 6))
        else:
            row.append(0.0)
    pv_generation.append(row)

# Consumption profiles (kWh per 15 min)
base_load = [0.30, 0.35, 0.40, 0.45, 0.50, 0.35, 0.45, 0.50, 0.30, 0.55]
consumption = []
for hh in range(N_HH):
    row = []
    for t in range(N_T):
        hour = t * 0.25
        if hour < 6:
            tf = 0.4
        elif hour < 9:
            tf = 1.2
        elif hour < 12:
            tf = 0.8
        elif hour < 17:
            tf = 0.7
        elif hour < 21:
            tf = 1.5
        else:
            tf = 0.9
        noise = 1.0 + 0.08 * random.gauss(0, 1)
        val = max(0.05, base_load[hh] * tf * noise)
        row.append(round(val, 6))
    consumption.append(row)

# Simplified MILP dispatch: proportional P2P matching
grid_import = [[0.0] * N_T for _ in range(N_HH)]
grid_export = [[0.0] * N_T for _ in range(N_HH)]
p2p_buy = [[0.0] * N_T for _ in range(N_HH)]
p2p_sell = [[0.0] * N_T for _ in range(N_HH)]

for t in range(N_T):
    net = [consumption[h][t] - pv_generation[h][t] for h in range(N_HH)]
    buyers = [(h, net[h]) for h in range(N_HH) if net[h] > 1e-9]
    sellers = [(h, -net[h]) for h in range(N_HH) if net[h] < -1e-9]

    total_demand = sum(d for _, d in buyers)
    total_supply = sum(s for _, s in sellers)

    if total_demand > 1e-9 and total_supply > 1e-9:
        matched = min(total_demand, total_supply)
        for h, d in buyers:
            share = d / total_demand
            p2p_amount = matched * share
            p2p_buy[h][t] = round(p2p_amount, 6)
            grid_import[h][t] = round(max(0, d - p2p_amount), 6)
        for h, s in sellers:
            share = s / total_supply
            p2p_amount = matched * share
            p2p_sell[h][t] = round(p2p_amount, 6)
            grid_export[h][t] = round(max(0, s - p2p_amount), 6)
    elif total_demand > 1e-9:
        for h, d in buyers:
            grid_import[h][t] = round(d, 6)
    elif total_supply > 1e-9:
        for h, s in sellers:
            grid_export[h][t] = round(s, 6)

data = {
    "n_households": N_HH,
    "n_timesteps": N_T,
    "grid_tariff": grid_tariff,
    "feed_in_tariff": feed_in_tariff,
    "consumption": consumption,
    "pv_generation": pv_generation,
    "grid_import": grid_import,
    "grid_export": grid_export,
    "p2p_buy": p2p_buy,
    "p2p_sell": p2p_sell,
}

os.makedirs("/app/data", exist_ok=True)
with open("/app/data/community.json", "w") as f:
    json.dump(data, f)

print("Community data generated successfully.")
