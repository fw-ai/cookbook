#!/usr/bin/env python3
"""Generate challenge data: binary public key, SQLite database, hash target."""
import hashlib
import sqlite3
import struct
import os

# Parameters
q, n, m, o, k = 31, 8, 4, 4, 2

# Public key matrices (upper triangular entries, row-major order)
matrices = [
    [26,20,29,2,17,28,1,20, 8,28,16,3,27,2,3, 2,16,23,19,20,12,
     22,14,2,16,4, 9,7,12,18, 20,3,26, 26,23, 11],
    [0,19,29,0,9,15,27,2, 19,30,4,15,26,12,13, 12,25,30,24,18,28,
     24,24,16,28,20, 26,6,3,23, 15,13,24, 10,28, 12],
    [8,1,17,28,17,10,27,9, 24,12,16,15,1,25,5, 6,2,10,3,25,24,
     29,20,26,29,15, 15,0,9,22, 18,24,20, 11,16, 2],
    [5,9,30,25,27,10,4,24, 12,30,30,29,8,21,5, 13,30,9,4,27,2,
     12,28,11,9,30, 28,14,24,19, 28,10,5, 10,18, 16],
]

message = "Vinaigrette is not cryptographically secure"

def shake256_hash(msg, m_count, field_q):
    bits_per = 5
    total_bits = bits_per * m_count
    total_bytes = (total_bits + 7) // 8
    h = hashlib.shake_256(msg.encode()).digest(total_bytes)
    bits = "".join(format(b, "08b") for b in h)
    return [int(bits[i * bits_per:(i + 1) * bits_per], 2) % field_q
            for i in range(m_count)]

def write_binary_key(path):
    with open(path, 'wb') as f:
        f.write(b'VNRT')
        f.write(struct.pack('B', 0x01))  # version
        f.write(struct.pack('B', q))
        f.write(struct.pack('B', n))
        f.write(struct.pack('B', m))
        f.write(struct.pack('B', o))
        f.write(struct.pack('B', k))
        f.write(b'\x00\x00')  # reserved
        for mat in matrices:
            f.write(bytes(mat))

def write_hash_target(path):
    target = shake256_hash(message, m, q)
    with open(path, 'wb') as f:
        f.write(bytes(target))

def create_database(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('CREATE TABLE params (key TEXT PRIMARY KEY, value INTEGER)')
    for key, val in [('q', q), ('n', n), ('m', m), ('o', o), ('k', k)]:
        c.execute('INSERT INTO params VALUES (?, ?)', (key, val))
    c.execute('CREATE TABLE message (id INTEGER PRIMARY KEY, content TEXT)')
    c.execute('INSERT INTO message VALUES (1, ?)', (message,))
    target = shake256_hash(message, m, q)
    c.execute('CREATE TABLE hash_target (idx INTEGER PRIMARY KEY, value INTEGER)')
    for i, v in enumerate(target):
        c.execute('INSERT INTO hash_target VALUES (?, ?)', (i, v))
    conn.commit()
    conn.close()

os.makedirs('/challenge', exist_ok=True)
write_binary_key('/challenge/public_key.bin')
write_hash_target('/challenge/hash_target.bin')
create_database('/challenge/challenge.db')
print("Challenge data generated successfully.")
