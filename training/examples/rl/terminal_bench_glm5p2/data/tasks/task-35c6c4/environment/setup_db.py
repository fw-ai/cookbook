#!/usr/bin/env python3
"""Create forensic investigation environment with multiple recovery challenges."""
import sqlite3
import struct
import os
import random
import shutil


def create_evidence_db():
    """Create the evidence database with financial investigation data."""
    os.makedirs('/app', exist_ok=True)
    conn = sqlite3.connect('/app/evidence.db')
    c = conn.cursor()

    c.execute('PRAGMA page_size = 4096')
    c.execute('PRAGMA journal_mode = DELETE')

    c.execute('''CREATE TABLE suspects (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        alias TEXT NOT NULL,
        last_known_city TEXT NOT NULL
    )''')

    c.execute('''CREATE TABLE accounts (
        id INTEGER PRIMARY KEY,
        suspect_id INTEGER NOT NULL,
        bank TEXT NOT NULL,
        account_no TEXT NOT NULL,
        balance REAL NOT NULL,
        FOREIGN KEY (suspect_id) REFERENCES suspects(id)
    )''')

    c.execute('''CREATE TABLE transfers (
        id INTEGER PRIMARY KEY,
        from_acct INTEGER NOT NULL,
        to_acct INTEGER NOT NULL,
        amount REAL NOT NULL,
        ts TEXT NOT NULL,
        note TEXT NOT NULL,
        FOREIGN KEY (from_acct) REFERENCES accounts(id),
        FOREIGN KEY (to_acct) REFERENCES accounts(id)
    )''')

    suspects = [
        (1, 'Marcus Blackwell', 'The Broker', 'Zurich'),
        (2, 'Lena Petrova', 'Nadia', 'Vienna'),
        (3, 'Chen Wei', 'Ghost', 'Singapore'),
        (4, 'Ricardo Fontes', 'El Reloj', 'Panama City'),
        (5, 'Yuki Tanaka', 'Kitsune', 'Tokyo'),
    ]
    c.executemany('INSERT INTO suspects VALUES (?,?,?,?)', suspects)

    accounts = [
        (1, 1, 'Swiss National', 'CH93-0076-2011-6238-5295-7', 2450000.00),
        (2, 1, 'Deutsche Bank', 'DE89-3704-0044-0532-0130-00', 890000.50),
        (3, 2, 'Raiffeisen', 'AT61-1904-3002-3457-3201', 1230000.00),
        (4, 3, 'OCBC', 'SG72-0000-0048-1263', 3780000.75),
        (5, 3, 'HSBC', 'HK93-1239-0000-0000-1234', 560000.00),
        (6, 4, 'Banco Nacional', 'PA49-0001-0120-0000-1234-5678', 4100000.00),
        (7, 5, 'MUFG', 'JP39-0009-0210-0000-0123-4567', 1850000.25),
        (8, 5, 'Sumitomo Mitsui', 'JP84-0200-0312-0000-9876-5432', 720000.00),
    ]
    c.executemany('INSERT INTO accounts VALUES (?,?,?,?,?)', accounts)

    transfers = [
        (1, 1, 4, 250000.00, '2024-01-15T09:23:00', 'consulting fee'),
        (2, 4, 6, 175000.50, '2024-01-22T14:05:00', 'equipment purchase'),
        (3, 2, 3, 500000.00, '2024-02-03T11:30:00', 'investment return'),
        (4, 6, 7, 890000.00, '2024-02-14T16:45:00', 'property acquisition'),
        (5, 3, 1, 120000.75, '2024-02-28T08:12:00', 'loan repayment'),
        (6, 5, 8, 340000.00, '2024-03-10T13:20:00', 'fund transfer'),
        (7, 7, 2, 445000.25, '2024-03-18T10:55:00', 'settlement'),
        (8, 8, 5, 210000.00, '2024-04-01T15:30:00', 'rebalancing'),
        (9, 1, 6, 780000.00, '2024-04-15T09:00:00', 'acquisition deposit'),
        (10, 4, 7, 135000.50, '2024-04-22T11:45:00', 'service payment'),
        (11, 6, 3, 920000.00, '2024-05-05T14:15:00', 'capital injection'),
        (12, 2, 8, 65000.00, '2024-05-18T16:30:00', 'administrative'),
        (13, 7, 1, 1100000.00, '2024-06-02T08:45:00', 'partnership buyout'),
        (14, 3, 4, 275000.75, '2024-06-15T12:00:00', 'interest payment'),
        (15, 8, 6, 430000.00, '2024-07-01T10:20:00', 'bridge loan'),
        (16, 5, 2, 185000.50, '2024-07-14T14:35:00', 'dividend distribution'),
        (17, 1, 7, 620000.00, '2024-08-03T09:10:00', 'trust allocation'),
        (18, 6, 8, 310000.25, '2024-08-22T11:50:00', 'operational expense'),
        (19, 4, 1, 545000.00, '2024-09-10T15:05:00', 'profit sharing'),
        (20, 7, 3, 750000.00, '2024-09-28T13:40:00', 'syndicate contribution'),
    ]
    c.executemany('INSERT INTO transfers VALUES (?,?,?,?,?,?)', transfers)

    conn.commit()
    conn.close()
    print(f"evidence.db created: {os.path.getsize('/app/evidence.db')} bytes")


def build_btree_page():
    """Build a valid SQLite B-tree leaf table page (type 0x0d) with 5 records.

    Schema: (id INTEGER PRIMARY KEY, code TEXT, destination TEXT, amount REAL, timestamp TEXT)
    INTEGER PRIMARY KEY is stored as NULL in the record; the actual value is the rowid.
    """
    PAGE_SIZE = 4096

    records_data = [
        (1, 'OMEGA-7', 'Cayman Islands', 2500000.0, '2024-03-15T03:00:00'),
        (2, 'THETA-3', 'Isle of Man', 1750000.0, '2024-04-20T02:30:00'),
        (3, 'SIGMA-9', 'Liechtenstein', 890000.5, '2024-06-01T04:15:00'),
        (4, 'DELTA-1', 'Seychelles', 3200000.0, '2024-07-18T01:45:00'),
        (5, 'PHI-12', 'Mauritius', 1450000.75, '2024-09-05T03:30:00'),
    ]

    def encode_varint(value):
        if value <= 127:
            return bytes([value])
        parts = []
        temp = value
        while temp > 0:
            parts.append(temp & 0x7F)
            temp >>= 7
        parts.reverse()
        result = bytearray()
        for i in range(len(parts)):
            if i < len(parts) - 1:
                result.append(parts[i] | 0x80)
            else:
                result.append(parts[i])
        return bytes(result)

    cells = []
    for rowid, code, dest, amount, ts in records_data:
        code_bytes = code.encode('utf-8')
        dest_bytes = dest.encode('utf-8')
        ts_bytes = ts.encode('utf-8')

        # Serial types: NULL(0) for id, text for code/dest/ts, float(7) for amount
        serial_types = [
            0,                          # NULL (id is rowid)
            len(code_bytes) * 2 + 13,   # TEXT
            len(dest_bytes) * 2 + 13,   # TEXT
            7,                          # IEEE 754 float
            len(ts_bytes) * 2 + 13,     # TEXT
        ]

        # Build header: header_size varint + serial type varints
        header_content = b''
        for st in serial_types:
            header_content += encode_varint(st)
        header_size = 1 + len(header_content)  # +1 for the header_size byte itself
        header = encode_varint(header_size) + header_content

        # Build body
        body = b''  # NULL: 0 bytes
        body += code_bytes
        body += dest_bytes
        body += struct.pack('>d', amount)
        body += ts_bytes

        record = header + body
        payload_size = len(record)

        cell = encode_varint(payload_size) + encode_varint(rowid) + record
        cells.append(cell)

    # Build the page
    page = bytearray(PAGE_SIZE)
    num_cells = len(cells)

    # Place cells from end of page working backward
    cell_offsets = []
    pos = PAGE_SIZE
    for cell in reversed(cells):
        pos -= len(cell)
        page[pos:pos + len(cell)] = cell
        cell_offsets.insert(0, pos)

    cell_content_start = pos

    # Page header (8 bytes for leaf table page, no right-child pointer)
    page[0] = 0x0d  # Leaf table B-tree page
    struct.pack_into('>H', page, 1, 0)  # First freeblock offset
    struct.pack_into('>H', page, 3, num_cells)  # Number of cells
    struct.pack_into('>H', page, 5, cell_content_start)  # Cell content area start
    page[7] = 0  # Fragmented free bytes

    # Cell pointer array (starts at offset 8)
    for i, offset in enumerate(cell_offsets):
        struct.pack_into('>H', page, 8 + i * 2, offset)

    return bytes(page)


def create_disk_image():
    """Create a 256 KiB disk fragment with a hidden B-tree page among noise and decoys."""
    PAGE_SIZE = 4096
    NUM_PAGES = 64  # 256 KiB total
    TARGET_PAGE = 37  # Page where the real B-tree data is hidden

    # Build the B-tree page
    btree_page = build_btree_page()
    assert len(btree_page) == PAGE_SIZE, f"B-tree page wrong size: {len(btree_page)}"
    assert btree_page[0] == 0x0d, f"B-tree page wrong type: 0x{btree_page[0]:02x}"

    # Fill image with deterministic pseudo-random noise (seeded for reproducibility)
    rng = random.Random(0xDEADBEEF)
    image = bytearray(rng.randbytes(PAGE_SIZE * NUM_PAGES))

    # Clear any accidental B-tree page type markers at page-aligned boundaries
    valid_btree_types = {0x02, 0x05, 0x0a, 0x0d}
    for p in range(NUM_PAGES):
        offset = p * PAGE_SIZE
        if image[offset] in valid_btree_types:
            image[offset] = 0x00

    # Decoy 1: interior table B-tree page (type 0x05) at page 7
    image[7 * PAGE_SIZE] = 0x05

    # Decoy 2: leaf index B-tree page (type 0x0a) at page 19
    image[19 * PAGE_SIZE] = 0x0a

    # Decoy 3: leaf table type (0x0d) at page 28 but cell count = 0
    image[28 * PAGE_SIZE] = 0x0d
    struct.pack_into('>H', image, 28 * PAGE_SIZE + 3, 0)

    # Decoy 4: leaf table (0x0d) at page 52, cell count = 3,
    # but cell pointers beyond page boundary (invalid)
    image[52 * PAGE_SIZE] = 0x0d
    struct.pack_into('>H', image, 52 * PAGE_SIZE + 3, 3)
    for j in range(3):
        struct.pack_into('>H', image, 52 * PAGE_SIZE + 8 + j * 2, 0xFFFF)

    # Place the actual B-tree leaf page at the target offset
    offset = TARGET_PAGE * PAGE_SIZE
    image[offset:offset + PAGE_SIZE] = btree_page

    with open('/app/disk_fragment.img', 'wb') as f:
        f.write(image)

    size = os.path.getsize('/app/disk_fragment.img')
    print(f"disk_fragment.img created: {size} bytes")
    assert size == PAGE_SIZE * NUM_PAGES, f"disk_fragment.img size wrong: {size}"


def create_ledger_db():
    """Create a ledger database in WAL mode with uncheckpointed transactions in a detached WAL."""

    initial_entries = [
        (1, '2024-01-10', 'Operating expenses Q1', 'Expenses', 'Cash', 85000.00),
        (2, '2024-02-15', 'Revenue from services', 'Cash', 'Revenue', 142000.00),
        (3, '2024-03-20', 'Payroll disbursement', 'Salaries', 'Cash', 63000.00),
        (4, '2024-04-05', 'Equipment acquisition', 'Assets', 'Cash', 210000.00),
        (5, '2024-05-12', 'Client payment received', 'Cash', 'Receivables', 95000.00),
    ]

    wal_entries = [
        (6, '2024-06-18', 'Offshore transfer alpha', 'Transfers_Out', 'Operating', 1500000.00),
        (7, '2024-07-03', 'Offshore transfer beta', 'Transfers_Out', 'Reserve', 2300000.00),
        (8, '2024-08-21', 'Final liquidation', 'Transfers_Out', 'Emergency', 750000.00),
    ]

    # Phase 1: Create database with initial entries, fully checkpointed
    conn = sqlite3.connect('/app/ledger.db')
    c = conn.cursor()
    c.execute('PRAGMA page_size = 4096')
    c.execute('PRAGMA journal_mode = WAL')
    c.execute('PRAGMA wal_autocheckpoint = 0')

    c.execute('''CREATE TABLE journal_entries (
        id INTEGER PRIMARY KEY,
        date TEXT NOT NULL,
        description TEXT NOT NULL,
        debit_account TEXT NOT NULL,
        credit_account TEXT NOT NULL,
        amount REAL NOT NULL
    )''')

    c.executemany('INSERT INTO journal_entries VALUES (?,?,?,?,?,?)', initial_entries)
    conn.commit()

    # Checkpoint to flush initial entries into the main database file
    c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    conn.close()
    print(f"ledger.db phase 1 done: {os.path.getsize('/app/ledger.db')} bytes")

    # Phase 2: Save base state (connection closed, clean state)
    shutil.copy('/app/ledger.db', '/tmp/ledger_base.db')

    # Clean up any WAL/SHM files left from phase 1
    for ext in ['-wal', '-shm']:
        path = '/app/ledger.db' + ext
        if os.path.exists(path):
            os.unlink(path)

    # Phase 3: Reopen and create WAL-only entries
    conn = sqlite3.connect('/app/ledger.db')
    c = conn.cursor()
    c.execute('PRAGMA wal_autocheckpoint = 0')

    # Verify WAL mode is active
    mode = c.execute('PRAGMA journal_mode').fetchone()[0]
    assert mode == 'wal', f"Expected WAL mode, got {mode}"

    c.executemany('INSERT INTO journal_entries VALUES (?,?,?,?,?,?)', wal_entries)
    conn.commit()

    # Verify WAL file exists
    wal_path = '/app/ledger.db-wal'
    assert os.path.exists(wal_path), "WAL file not created after inserting entries"
    wal_size = os.path.getsize(wal_path)
    assert wal_size > 0, "WAL file is empty"

    # Copy WAL while connection is open (WAL file exists)
    shutil.copy(wal_path, '/app/seized_journal.wal')
    print(f"seized_journal.wal created: {os.path.getsize('/app/seized_journal.wal')} bytes")

    # Close connection (triggers checkpoint and WAL deletion)
    conn.close()

    # Phase 4: Restore base database (entries 1-5 only)
    shutil.copy('/tmp/ledger_base.db', '/app/ledger.db')
    for ext in ['-wal', '-shm']:
        path = '/app/ledger.db' + ext
        if os.path.exists(path):
            os.unlink(path)

    # Phase 5: Verify WAL recovery actually works
    shutil.copy('/app/seized_journal.wal', '/app/ledger.db-wal')
    conn = sqlite3.connect('/app/ledger.db')
    count = conn.execute('SELECT COUNT(*) FROM journal_entries').fetchone()[0]
    conn.close()
    assert count == 8, f"WAL recovery verification failed: expected 8, got {count}"
    print(f"WAL recovery verified: {count} entries recovered")

    # Phase 6: Final restore to base state for the task
    shutil.copy('/tmp/ledger_base.db', '/app/ledger.db')
    for ext in ['-wal', '-shm']:
        path = '/app/ledger.db' + ext
        if os.path.exists(path):
            os.unlink(path)
    os.unlink('/tmp/ledger_base.db')

    print(f"ledger.db final: {os.path.getsize('/app/ledger.db')} bytes (5 entries)")
    print(f"seized_journal.wal: {os.path.getsize('/app/seized_journal.wal')} bytes (3 WAL entries)")


def corrupt_evidence_db():
    """Introduce 5 specific corruptions into the evidence database header."""
    with open('/app/evidence.db', 'r+b') as f:
        data = bytearray(f.read())

        # Corruption 1: Zero out bytes 6-15 of the magic header string
        for i in range(6, 16):
            data[i] = 0x00

        # Corruption 2: Wrong page size at offset 16-17 (big-endian uint16)
        struct.pack_into('>H', data, 16, 2048)

        # Corruption 3: Invalid text encoding at offset 56-59
        struct.pack_into('>I', data, 56, 0)

        # Corruption 4: Wrong database size in pages at offset 28-31
        struct.pack_into('>I', data, 28, 9999)

        # Corruption 5: Invalid schema format number at offset 44-47
        struct.pack_into('>I', data, 44, 0)

        f.seek(0)
        f.write(data)

    print("evidence.db corrupted (5 header fields)")


if __name__ == '__main__':
    create_evidence_db()
    create_disk_image()
    create_ledger_db()
    corrupt_evidence_db()

    # Verify all files exist and have expected sizes
    expected_files = {
        '/app/evidence.db': None,
        '/app/disk_fragment.img': 4096 * 64,
        '/app/ledger.db': None,
        '/app/seized_journal.wal': None,
    }
    for path, expected_size in expected_files.items():
        assert os.path.exists(path), f"MISSING: {path}"
        actual_size = os.path.getsize(path)
        if expected_size is not None:
            assert actual_size == expected_size, \
                f"SIZE MISMATCH {path}: expected {expected_size}, got {actual_size}"
        assert actual_size > 0, f"EMPTY: {path}"
        print(f"  OK: {path} ({actual_size} bytes)")

    print("Setup complete: all forensic sources created and verified")
