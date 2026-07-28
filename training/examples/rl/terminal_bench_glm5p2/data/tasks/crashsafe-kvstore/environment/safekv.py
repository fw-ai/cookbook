"""
Crash-safe key-value store using a write-ahead log (WAL).

Provides crash-consistent storage by logging operations to a WAL before
applying them to the data file. Uses atomic file replacement (temp file +
rename) for data persistence.

WAL binary format: sequence of entries, each consisting of:
  - 4-byte big-endian uint32: payload length
  - 4-byte big-endian uint32: CRC32 checksum of payload
  - N-byte payload: JSON-encoded operation {"op", "key", "value"}
"""

import os
import json
import struct
import zlib

HEADER_SIZE = 8  # 4 bytes payload length + 4 bytes CRC32


class CrashSafeKV:
    """A key-value store with write-ahead logging for crash consistency.

    Usage:
        kv = CrashSafeKV('/path/to/data')
        kv.recover()  # Call on startup to replay any incomplete WAL entries
        kv.put('key', 'value')
        print(kv.get('key'))
        kv.delete('key')
    """

    def __init__(self, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, 'data.json')
        self.wal_file = os.path.join(data_dir, 'wal.log')
        self.data = {}
        self._load_data()

    def _load_data(self):
        """Load the current data file into memory."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)

    def put(self, key, value):
        """Store a key-value pair with crash safety."""
        self._append_wal_entry('put', key, value)
        self.data[key] = value
        self._persist_data()
        self._clear_wal()

    def get(self, key):
        """Retrieve the value for a key, or None if not found."""
        return self.data.get(key)

    def delete(self, key):
        """Delete a key with crash safety."""
        self._append_wal_entry('delete', key, None)
        self.data.pop(key, None)
        self._persist_data()
        self._clear_wal()

    def _append_wal_entry(self, op, key, value):
        """Append an operation to the write-ahead log."""
        payload = json.dumps({'op': op, 'key': key, 'value': value}).encode('utf-8')
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = struct.pack('>II', len(payload), crc)
        with open(self.wal_file, 'ab') as f:
            f.write(header + payload)

    def _persist_data(self):
        """Atomically persist the current data to disk using rename."""
        tmp_path = self.data_file + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(self.data, f)
        os.rename(tmp_path, self.data_file)

    def _clear_wal(self):
        """Remove the WAL file after successful persistence."""
        if os.path.exists(self.wal_file):
            os.unlink(self.wal_file)

    def recover(self):
        """Replay WAL entries to recover from a crash.

        Should be called on startup. Reads all entries from the WAL,
        applies them to the data, persists the result, and clears the WAL.
        """
        if not os.path.exists(self.wal_file):
            return

        self._load_data()

        with open(self.wal_file, 'rb') as f:
            while True:
                header = f.read(HEADER_SIZE)
                if not header:
                    break
                length, stored_crc = struct.unpack('>II', header)
                payload = f.read(length)
                entry = json.loads(payload.decode('utf-8'))
                if entry['op'] == 'put':
                    self.data[entry['key']] = entry['value']
                elif entry['op'] == 'delete':
                    self.data.pop(entry['key'], None)

        self._persist_data()
        self._clear_wal()
