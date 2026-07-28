#!/usr/bin/env python3
"""Generate deterministic test data for NIST SP 800-22 statistical testing task."""
import hashlib
import struct
import random
import os

NUM_BYTES = 125000  # 1,000,000 bits

def gen_sha256_stream():
    """SHA-256 in counter mode - cryptographic quality."""
    result = bytearray()
    ctr = 0
    while len(result) < NUM_BYTES:
        result.extend(hashlib.sha256(struct.pack('>Q', ctr)).digest())
        ctr += 1
    return bytes(result[:NUM_BYTES])

def gen_lcg_stream(seed=1):
    """glibc-style LCG, outputting only the low byte - known weak."""
    result = bytearray()
    x = seed
    for _ in range(NUM_BYTES):
        x = (1103515245 * x + 12345) % (2**31)
        result.append(x & 0xFF)
    return bytes(result)

def gen_mt_stream(seed=42):
    """Mersenne Twister (Python random) - good statistical quality."""
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(NUM_BYTES))

os.makedirs('/app/data', exist_ok=True)

with open('/app/data/sha256_stream.bin', 'wb') as f:
    f.write(gen_sha256_stream())

with open('/app/data/lcg_stream.bin', 'wb') as f:
    f.write(gen_lcg_stream())

with open('/app/data/mt_stream.bin', 'wb') as f:
    f.write(gen_mt_stream())
