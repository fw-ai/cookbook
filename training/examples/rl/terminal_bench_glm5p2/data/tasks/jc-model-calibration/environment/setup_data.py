#!/usr/bin/env python3
"""Generate synthetic experimental tensile test data for JC model calibration.

Ground truth Johnson-Cook parameters (Ti-6Al-4V inspired):
  Hardening: sigma = [A + B*ep^n] * [1 + C*ln(edot/edot0)] * [1 - T*^m]
  Fracture:  ef = D1 + D2*exp(D3*eta)
"""
import json
import csv
import os
import math
import random

# --- Ground truth Johnson-Cook parameters ---
A_GT = 985.0      # MPa, initial yield
B_GT = 680.0      # MPa, hardening modulus
N_GT = 0.47       # hardening exponent
C_GT = 0.015      # strain-rate sensitivity
M_GT = 0.75       # thermal softening exponent
D1_GT = 0.05      # fracture param 1
D2_GT = 0.40      # fracture param 2
D3_GT = -0.90     # fracture param 3

# --- Material properties ---
E_MAT = 113800.0    # MPa
RHO = 4430.0        # kg/m3
CP = 526.0          # J/(kg*K)
T_MELT = 1933.0     # K
T_ROOM = 298.0      # K
CHI = 0.9           # Taylor-Quinney coefficient
EDOT_REF = 0.001    # /s reference strain rate

# --- Specimen geometry ---
L0 = 25.4           # mm gauge length
D0 = 6.35           # mm gauge diameter
A0 = math.pi / 4.0 * D0 * D0  # mm^2

DATA_DIR = '/opt/task_data'


def jc_stress(ep, edot, T):
    """Johnson-Cook flow stress in MPa."""
    h = A_GT + (B_GT * math.pow(ep, N_GT) if ep > 1e-12 else 0.0)
    r = 1.0 + C_GT * math.log(edot / EDOT_REF) if edot > EDOT_REF else 1.0
    Ts = max(0.0, (T - T_ROOM) / (T_MELT - T_ROOM))
    t = 1.0 - math.pow(Ts, M_GT) if Ts > 1e-15 else 1.0
    return h * r * t


def generate_tensile(edot, adiabatic, seed_val):
    """Generate force-displacement data for a smooth round-bar tensile test."""
    rng = random.Random(seed_val)

    # Fracture strain at eta = 1/3 (smooth bar)
    eta = 1.0 / 3.0
    ef = D1_GT + D2_GT * math.exp(D3_GT * eta)

    rows = []

    # --- Elastic region (30 points) ---
    sigma_y = jc_stress(0.0, edot, T_ROOM)
    ey = sigma_y / E_MAT
    for i in range(1, 31):
        ee = ey * i / 30.0
        se = E_MAT * ee
        F = se * A0
        d = ee * L0
        F *= (1.0 + 0.003 * rng.gauss(0, 1))
        d *= (1.0 + 0.001 * rng.gauss(0, 1))
        d = max(d, 1e-8)
        rows.append((d, F))

    # --- Plastic region (400 points) via forward-Euler adiabatic integration ---
    n_plastic = 400
    dep = ef / n_plastic
    T = T_ROOM

    for i in range(n_plastic):
        ep = dep * (i + 1)
        sigma_t = jc_stress(ep, edot, T)

        # Store data point
        et = ep + sigma_t / E_MAT
        ee = math.exp(et) - 1.0
        se = sigma_t * math.exp(-et)
        F = se * A0
        d = ee * L0
        F *= (1.0 + 0.005 * rng.gauss(0, 1))
        d *= (1.0 + 0.001 * rng.gauss(0, 1))
        rows.append((d, F))

        # Adiabatic temperature update (forward Euler)
        if adiabatic:
            dT = CHI * sigma_t * 1e6 * dep / (RHO * CP)
            T += dT

    # Ensure monotonic displacement
    for i in range(1, len(rows)):
        if rows[i][0] <= rows[i - 1][0]:
            rows[i] = (rows[i - 1][0] + 1e-6, rows[i][1])

    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Material properties file ---
    with open(os.path.join(DATA_DIR, 'material_properties.json'), 'w') as f:
        json.dump({
            'material': 'Ti-6Al-4V (mill-annealed)',
            'youngs_modulus_MPa': E_MAT,
            'poissons_ratio': 0.342,
            'density_kg_m3': RHO,
            'specific_heat_J_per_kgK': CP,
            'melting_temperature_K': T_MELT,
            'room_temperature_K': T_ROOM,
            'taylor_quinney_coefficient': CHI,
            'reference_strain_rate_per_s': EDOT_REF,
        }, f, indent=2)

    # --- Specimen geometry file ---
    with open(os.path.join(DATA_DIR, 'specimen_geometry.json'), 'w') as f:
        json.dump({
            'specimen_type': 'round_tensile_bar',
            'gauge_length_mm': L0,
            'gauge_diameter_mm': D0,
            'cross_section_area_mm2': round(A0, 4),
            'columns': ['displacement_mm', 'force_N'],
            'notes': 'Force in Newtons, displacement in mm measured by extensometer over gauge length.'
        }, f, indent=2)

    # --- Tensile test data ---
    configs = [
        ('tensile_QS', 0.001, False, 3, [100, 137, 174]),
        ('tensile_IR', 1.0,   False, 2, [200, 237]),
        ('tensile_HR', 1000.0, True,  2, [300, 337]),
    ]

    for prefix, edot, adiab, ntests, seeds in configs:
        for j in range(ntests):
            data = generate_tensile(edot, adiab, seeds[j])
            fname = os.path.join(DATA_DIR, '{}_{:02d}.csv'.format(prefix, j + 1))
            with open(fname, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['displacement_mm', 'force_N', 'strain_rate_per_s'])
                for d, frc in data:
                    w.writerow(['{:.6f}'.format(d), '{:.4f}'.format(frc), '{:.3e}'.format(edot)])

    # --- Fracture data from notched specimens ---
    a = D0 / 2.0  # minimum cross-section radius
    specimens = [
        ('smooth_round',  1.0 / 3.0),
        ('notched_R10',   1.0 / 3.0 + math.log(1.0 + a / (2.0 * 10.0))),
        ('notched_R6',    1.0 / 3.0 + math.log(1.0 + a / (2.0 * 6.0))),
        ('notched_R3',    1.0 / 3.0 + math.log(1.0 + a / (2.0 * 3.0))),
        ('notched_R1p5',  1.0 / 3.0 + math.log(1.0 + a / (2.0 * 1.5))),
    ]

    rng_frac = random.Random(777)
    with open(os.path.join(DATA_DIR, 'fracture_data.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['specimen_type', 'stress_triaxiality', 'fracture_strain'])
        for name, eta in specimens:
            ef_true = D1_GT + D2_GT * math.exp(D3_GT * eta)
            ef_meas = ef_true * (1.0 + 0.02 * rng_frac.gauss(0, 1))
            w.writerow([name, '{:.6f}'.format(eta), '{:.6f}'.format(ef_meas)])

    # --- Test matrix ---
    with open(os.path.join(DATA_DIR, 'test_matrix.json'), 'w') as f:
        json.dump({
            'tensile_tests': [
                {'file_prefix': 'tensile_QS', 'strain_rate_per_s': 0.001,
                 'n_specimens': 3, 'thermal_condition': 'isothermal'},
                {'file_prefix': 'tensile_IR', 'strain_rate_per_s': 1.0,
                 'n_specimens': 2, 'thermal_condition': 'isothermal'},
                {'file_prefix': 'tensile_HR', 'strain_rate_per_s': 1000.0,
                 'n_specimens': 2, 'thermal_condition': 'adiabatic'},
            ],
            'fracture_tests': {
                'file': 'fracture_data.csv',
                'notes': 'Fracture strains measured at quasi-static rate. Triaxiality from Bridgman analysis: eta = 1/3 + ln(1 + a/(2R)), where a = specimen min radius, R = notch root radius.'
            }
        }, f, indent=2)

    print('Data generation complete. Files in ' + DATA_DIR)


if __name__ == '__main__':
    main()
