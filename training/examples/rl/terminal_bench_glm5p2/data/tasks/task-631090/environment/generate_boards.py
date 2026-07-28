#!/usr/bin/env python3
"""Generate deterministic SameGame board instances using a linear congruential generator."""
import os
import sys

def generate_board(seed, rows=15, cols=15, colors=3):
    """Generate a board using glibc LCG parameters."""
    board = []
    s = seed
    for _ in range(rows):
        row = []
        for _ in range(cols):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            color = (s >> 16) % colors
            row.append(color)
        board.append(row)
    return board

os.makedirs('/app/instances', exist_ok=True)
os.makedirs('/app/solutions', exist_ok=True)

seeds = [42, 137, 2024, 31415, 65537]
for i, seed in enumerate(seeds, 1):
    board = generate_board(seed)
    path = f'/app/instances/board_{i}.txt'
    with open(path, 'w') as f:
        f.write('15 15 3\n')
        for row in board:
            f.write(' '.join(map(str, row)) + '\n')
    # Verify the file was written
    if not os.path.exists(path):
        print(f"ERROR: Failed to create {path}", file=sys.stderr)
        sys.exit(1)
    sz = os.path.getsize(path)
    if sz < 100:
        print(f"ERROR: {path} too small ({sz} bytes)", file=sys.stderr)
        sys.exit(1)

# Final verification
for i in range(1, 6):
    path = f'/app/instances/board_{i}.txt'
    assert os.path.exists(path), f"Missing {path}"
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 16, f"{path} has {len(lines)} lines, expected 16"
    print(f"Created {path} ({os.path.getsize(path)} bytes)")
