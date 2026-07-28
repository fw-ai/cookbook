#!/usr/bin/env python3
"""Generate Hamiltonian matrices and configuration for the compilation task."""
import numpy as np
import json
import os

# Pauli matrices
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

# --- 2-qubit XXZ Heisenberg model (anisotropy Delta=0.5) ---
# H = XX + YY + 0.5*ZZ
h_xxz_2q = (
    np.kron(sx, sx) + np.kron(sy, sy) + 0.5 * np.kron(sz, sz)
)

# --- 2-qubit transverse-field Ising model (h=0.7) ---
# H = ZZ + 0.7*(XI + IX)
h_tfim_2q = (
    np.kron(sz, sz)
    + 0.7 * (np.kron(sx, I2) + np.kron(I2, sx))
)

# --- 3-qubit Heisenberg XXX chain ---
# H = X1X2 + Y1Y2 + Z1Z2 + X2X3 + Y2Y3 + Z2Z3
h_heis_3q = (
    np.kron(np.kron(sx, sx), I2)
    + np.kron(np.kron(sy, sy), I2)
    + np.kron(np.kron(sz, sz), I2)
    + np.kron(I2, np.kron(sx, sx))
    + np.kron(I2, np.kron(sy, sy))
    + np.kron(I2, np.kron(sz, sz))
)

# --- 3-qubit transverse-field Ising model (h=0.5) ---
# H = Z1Z2 + Z2Z3 + 0.5*(X1 + X2 + X3)
h_tfim_3q = (
    np.kron(np.kron(sz, sz), I2)
    + np.kron(I2, np.kron(sz, sz))
    + 0.5 * (
        np.kron(np.kron(sx, I2), I2)
        + np.kron(np.kron(I2, sx), I2)
        + np.kron(np.kron(I2, I2), sx)
    )
)

# Verify all are Hermitian
for name, h in [("h_xxz_2q", h_xxz_2q), ("h_tfim_2q", h_tfim_2q),
                ("h_heis_3q", h_heis_3q), ("h_tfim_3q", h_tfim_3q)]:
    assert np.allclose(h, h.conj().T), f"{name} is not Hermitian"

os.makedirs('/app', exist_ok=True)
np.savez(
    '/app/hamiltonians.npz',
    h_xxz_2q=h_xxz_2q,
    h_tfim_2q=h_tfim_2q,
    h_heis_3q=h_heis_3q,
    h_tfim_3q=h_tfim_3q,
)

config = {
    "evolution_time": 1.0,
    "gate_set": ["CZGate", "RZGate", "SXGate"],
    "max_hilbert_schmidt_distance": 1e-5,
    "max_two_qubit_gates": {
        "2_qubit": 8,
        "3_qubit": 25
    },
}
with open('/app/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Task data generated at /app/")
