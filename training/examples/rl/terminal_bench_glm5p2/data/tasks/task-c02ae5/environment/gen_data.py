#!/usr/bin/env python3
"""Generate binary columnar data, SQLite catalog (with workload), and column statistics."""
import struct
import random
import os
import json
import sqlite3

SEED = 20180314

# Workload queries stored in the catalog database.
# (batch_id, query_id, query_text)
WORKLOAD_QUERIES = [
    # Batch 1: Two-way equi-joins
    (1, 1, "0 1|0.1=1.0|0.0 1.2"),
    (1, 2, "1 2|0.1=1.0|0.2 1.4"),
    (1, 3, "0 3|0.2=1.0|0.3 1.0"),
    (1, 4, "2 4|0.3=1.0|0.2 1.2"),
    (1, 5, "3 1|0.2=1.0|0.0 1.2"),
    # Batch 2: Two-way with filters
    (2, 1, "0 1|0.1=1.0&0.3>500000|0.0 1.2"),
    (2, 2, "0 2|0.0=1.5&1.1<50|0.3 1.2"),
    (2, 3, "4 0|0.3=1.0&0.4=0|0.2 1.3"),
    (2, 4, "1 2|0.1=1.0&0.2>30000&1.4<50000|0.2 1.2"),
    (2, 5, "0 1|0.1=1.0&0.4=25|0.0 0.3"),
    # Batch 3: Three-way chain joins
    (3, 1, "0 1 2|0.1=1.0&1.1=2.0|0.0 2.4"),
    (3, 2, "1 2 4|0.1=1.0&1.3=2.0|0.2 2.2"),
    (3, 3, "0 3 1|0.2=1.0&1.2=2.0|0.3 2.2"),
    (3, 4, "0 1 2|0.1=1.0&1.1=2.0&0.3>700000|0.3 1.2 2.2"),
    (3, 5, "3 0 1|0.1=1.0&1.1=2.0|0.0 2.2"),
    # Batch 4: Cycles and self-joins
    (4, 1, "0 0|0.1=1.1|0.0 1.0"),
    (4, 2, "0 1 0|0.1=1.0&1.3=2.0|0.0 2.3"),
    (4, 3, "2 4 2|0.3=1.0&1.1=2.0|0.2 2.4"),
    (4, 4, "0 3|0.2=1.0&1.1=0.0|0.3 1.0"),
    (4, 5, "1 0|0.0=1.1&1.0=0.3|0.2 1.3"),
    # Batch 5: Four-way joins
    (5, 1, "0 1 2 4|0.1=1.0&1.1=2.0&2.3=3.0|0.0 3.2"),
    (5, 2, "0 1 3 2|0.1=1.0&0.2=2.0&1.1=3.0|0.3 3.4"),
    (5, 3, "0 1 2 4|0.1=1.0&1.1=2.0&2.3=3.0&3.3=0.0|0.3 2.2"),
    (5, 4, "0 1 2 3|0.1=1.0&1.1=2.0&0.2=3.0&2.5=0.0|0.0 3.0"),
    (5, 5, "4 0 1 2|0.3=1.0&1.1=2.0&2.1=3.0&3.4<20000&0.4<3|0.2 3.2"),
    # Batch 6: Five-way joins and edge cases
    (6, 1, "0 1 2 3 4|0.1=1.0&1.1=2.0&0.2=3.0&2.3=4.0|0.0 4.2"),
    (6, 2, "0 1 2 4 3|0.1=1.0&1.1=2.0&2.3=3.0&0.2=4.0&3.3=0.0|0.3 2.4"),
    (6, 3, "0 1 2 4 0|0.1=1.0&1.1=2.0&2.3=3.0&3.3=4.0|0.0 4.3"),
    (6, 4, "0 2|0.0=1.5&0.3>9999999|0.0 1.2"),
    (6, 5, "0 1 2 3 4|0.1=1.0&1.1=2.0&0.2=3.0&2.3=4.0&4.3=0.0&0.3>800000|0.3 4.2"),
]


def write_relation(filepath, num_rows, num_cols, columns):
    """Write relation in binary columnar format:
    uint64 num_tuples | uint64 num_columns | col0[0..N-1] | col1[0..N-1] | ...
    All values little-endian uint64.
    """
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<QQ', num_rows, num_cols))
        for col in columns:
            for val in col:
                f.write(struct.pack('<Q', val))


def create_catalog(relations):
    """Create SQLite catalog database with schema metadata and workload."""
    cat_path = '/app/catalog.db'
    if os.path.exists(cat_path):
        os.remove(cat_path)

    db = sqlite3.connect(cat_path)

    db.execute('''CREATE TABLE relations (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        filepath TEXT NOT NULL,
        num_tuples INTEGER NOT NULL,
        num_columns INTEGER NOT NULL
    )''')

    db.execute('''CREATE TABLE columns (
        relation_id INTEGER NOT NULL,
        col_index INTEGER NOT NULL,
        col_type TEXT NOT NULL DEFAULT 'UINT64',
        min_val INTEGER,
        max_val INTEGER,
        ndv INTEGER,
        PRIMARY KEY (relation_id, col_index),
        FOREIGN KEY (relation_id) REFERENCES relations(id)
    )''')

    db.execute('''CREATE TABLE foreign_keys (
        src_rel INTEGER NOT NULL,
        src_col INTEGER NOT NULL,
        dst_rel INTEGER NOT NULL,
        dst_col INTEGER NOT NULL,
        FOREIGN KEY (src_rel) REFERENCES relations(id),
        FOREIGN KEY (dst_rel) REFERENCES relations(id)
    )''')

    db.execute('''CREATE TABLE workload (
        batch_id INTEGER NOT NULL,
        query_id INTEGER NOT NULL,
        query_text TEXT NOT NULL,
        PRIMARY KEY (batch_id, query_id)
    )''')

    fk_defs = [
        (0, 1, 1, 0), (0, 2, 3, 0),
        (1, 1, 2, 0), (1, 3, 0, 0),
        (2, 3, 4, 0), (2, 5, 0, 0),
        (3, 1, 0, 0), (3, 2, 1, 0),
        (4, 1, 2, 0), (4, 3, 0, 0),
    ]

    for name in ['r0', 'r1', 'r2', 'r3', 'r4']:
        rid = int(name[1:])
        nr, nc, cols = relations[name]
        filepath = f'/app/data/{name}'

        db.execute('INSERT INTO relations VALUES (?, ?, ?, ?, ?)',
                   (rid, name, filepath, nr, nc))

        for ci in range(nc):
            col = cols[ci]
            db.execute('INSERT INTO columns VALUES (?, ?, ?, ?, ?, ?)',
                       (rid, ci, 'UINT64', min(col), max(col), len(set(col))))

    for src_r, src_c, dst_r, dst_c in fk_defs:
        db.execute('INSERT INTO foreign_keys VALUES (?, ?, ?, ?)',
                   (src_r, src_c, dst_r, dst_c))

    for batch_id, query_id, query_text in WORKLOAD_QUERIES:
        db.execute('INSERT INTO workload VALUES (?, ?, ?)',
                   (batch_id, query_id, query_text))

    db.commit()
    db.close()


def create_statistics(relations):
    """Create statistics JSON with equi-width histograms."""
    stats = {}
    for name in ['r0', 'r1', 'r2', 'r3', 'r4']:
        nr, nc, cols = relations[name]
        col_stats = {}
        for j in range(nc):
            col_data = cols[j]
            sorted_vals = sorted(col_data)
            bucket_size = max(1, len(sorted_vals) // 10)
            histogram = []
            for b in range(10):
                start = b * bucket_size
                end = start + bucket_size if b < 9 else len(sorted_vals)
                bucket_vals = sorted_vals[start:end]
                histogram.append({
                    'lo': bucket_vals[0],
                    'hi': bucket_vals[-1],
                    'count': len(bucket_vals),
                    'ndv': len(set(bucket_vals))
                })
            col_stats[f'c{j}'] = {
                'min': min(col_data),
                'max': max(col_data),
                'ndv': len(set(col_data)),
                'histogram': histogram,
            }
        stats[name] = {'num_rows': nr, 'columns': col_stats}

    with open('/app/data/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)


def main():
    random.seed(SEED)
    os.makedirs('/app/data', exist_ok=True)

    relations = {}

    # R0: 10000 rows, 5 columns
    n0 = 10000
    cat_weights = [1.0 / ((k + 1) ** 1.2) for k in range(50)]
    r0 = [
        list(range(n0)),
        [random.randint(0, 5999) for _ in range(n0)],
        [random.randint(0, 3999) for _ in range(n0)],
        [random.randint(1, 1000000) for _ in range(n0)],
        random.choices(range(50), weights=cat_weights, k=n0),
    ]
    write_relation('/app/data/r0', n0, 5, r0)
    relations['r0'] = (n0, 5, r0)

    # R1: 6000 rows, 4 columns
    n1 = 6000
    r1 = [
        list(range(n1)),
        [random.randint(0, 14999) for _ in range(n1)],
        [random.randint(1, 50000) for _ in range(n1)],
        [random.randint(0, 9999) for _ in range(n1)],
    ]
    write_relation('/app/data/r1', n1, 4, r1)
    relations['r1'] = (n1, 4, r1)

    # R2: 15000 rows, 6 columns
    n2 = 15000
    r2 = [
        list(range(n2)),
        [random.randint(0, 199) for _ in range(n2)],
        [random.randint(1, 1000000) for _ in range(n2)],
        [random.randint(0, 24999) for _ in range(n2)],
        [random.randint(1, 99999) for _ in range(n2)],
        [random.randint(0, 9999) for _ in range(n2)],
    ]
    write_relation('/app/data/r2', n2, 6, r2)
    relations['r2'] = (n2, 6, r2)

    # R3: 4000 rows, 3 columns
    n3 = 4000
    r3 = [
        list(range(n3)),
        [random.randint(0, 9999) for _ in range(n3)],
        [random.randint(0, 5999) for _ in range(n3)],
    ]
    write_relation('/app/data/r3', n3, 3, r3)
    relations['r3'] = (n3, 3, r3)

    # R4: 25000 rows, 5 columns
    n4 = 25000
    status_weights = [1.0 / ((k + 1) ** 2.0) for k in range(10)]
    r4 = [
        list(range(n4)),
        [random.randint(0, 14999) for _ in range(n4)],
        [random.randint(1, 500000) for _ in range(n4)],
        [random.randint(0, 9999) for _ in range(n4)],
        random.choices(range(10), weights=status_weights, k=n4),
    ]
    write_relation('/app/data/r4', n4, 5, r4)
    relations['r4'] = (n4, 5, r4)

    # Write manifest
    with open('/app/data/relations.txt', 'w') as f:
        for i in range(5):
            f.write(f'/app/data/r{i}\n')

    create_catalog(relations)
    create_statistics(relations)

    sizes = ', '.join(f'{n}({relations[n][0]}x{relations[n][1]})'
                      for n in ['r0', 'r1', 'r2', 'r3', 'r4'])
    print(f"Generated 5 relations: {sizes}")
    print("SQLite catalog (with workload): /app/catalog.db")
    print("Statistics: /app/data/statistics.json")


if __name__ == '__main__':
    main()
