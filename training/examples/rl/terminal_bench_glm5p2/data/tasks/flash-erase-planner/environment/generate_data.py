#!/usr/bin/env python3
"""Generate deterministic test data for flash programming planner debug task.

Creates firmware.ihex (Intel HEX), firmware.bin (raw binary for test verification),
current_flash.bin, and config.json with carefully crafted data to test all flash
programming optimization scenarios:

  Sector 0 (16KB):  Different data requiring 0->1 bit transitions -> ERASE needed
  Sector 1 (16KB):  Identical firmware and current -> SKIP entirely
  Sector 2 (16KB):  Current is bitwise superset of firmware -> PROGRAM only (no erase)
  Sector 3 (16KB):  Different constant data requiring erase -> ERASE needed
  Sector 4 (64KB):  Current is erased (0xFF) -> PROGRAM only (no erase)
  Sector 5 (128KB): Partially covered, current is 0xFF -> PROGRAM only (partial page)
  Sectors 6-11:     No firmware data -> SKIP
"""
import json
import os
import sys

FLASH_BASE = 0x08000000
BANK1_SIZE = 0x100000  # 1MB
FIRMWARE_SIZE = 153800  # Creates a partial last page (200 bytes + 824 fill)


def fw_byte(offset):
    """Deterministic firmware byte. Range: 1-253 (never 0x00 or 0xFF)."""
    return ((offset * 7 + 3) % 253) + 1


def make_firmware():
    data = bytearray(FIRMWARE_SIZE)
    for i in range(FIRMWARE_SIZE):
        if 0xC000 <= i < 0x10000:
            # Sector 3: constant 0x42
            data[i] = 0x42
        else:
            data[i] = fw_byte(i)
    return bytes(data)


def make_current_flash(firmware):
    current = bytearray([0xFF] * BANK1_SIZE)

    # Sector 0: different data (will require 0->1 transitions -> erase)
    for i in range(0x0000, 0x4000):
        current[i] = ((i * 13 + 7) % 253) + 1

    # Sector 1: identical to firmware (skip)
    for i in range(0x4000, 0x8000):
        current[i] = firmware[i]

    # Sector 2: bitwise superset of firmware (program without erase)
    # current[i] = firmware[i] | 0xC0 ensures firmware & ~current == 0
    for i in range(0x8000, 0xC000):
        current[i] = firmware[i] | 0xC0

    # Sector 3: constant 0x24, firmware is 0x42 (erase needed: 0x42 & ~0x24 != 0)
    for i in range(0xC000, 0x10000):
        current[i] = 0x24

    # Sectors 4-11: remain 0xFF (erased)
    return bytes(current)


def make_ihex_line(record_type, address, data=b''):
    """Create one Intel HEX record line."""
    byte_count = len(data)
    addr_hi = (address >> 8) & 0xFF
    addr_lo = address & 0xFF
    all_bytes = bytes([byte_count, addr_hi, addr_lo, record_type]) + bytes(data)
    checksum = (~sum(all_bytes) + 1) & 0xFF
    return ':' + all_bytes.hex().upper() + f'{checksum:02X}'


def firmware_to_ihex(firmware, base_address):
    """Convert raw firmware binary to Intel HEX format."""
    lines = []
    CHUNK_SIZE = 16  # standard 16 bytes per data record
    current_segment = -1
    offset = 0

    while offset < len(firmware):
        abs_addr = base_address + offset
        segment = (abs_addr >> 16) & 0xFFFF

        if segment != current_segment:
            # Extended Linear Address record (type 04)
            ext_data = bytes([(segment >> 8) & 0xFF, segment & 0xFF])
            lines.append(make_ihex_line(0x04, 0x0000, ext_data))
            current_segment = segment

        local_addr = abs_addr & 0xFFFF
        remaining_in_segment = 0x10000 - local_addr
        chunk_size = min(CHUNK_SIZE, len(firmware) - offset, remaining_in_segment)

        chunk = firmware[offset:offset + chunk_size]
        lines.append(make_ihex_line(0x00, local_addr, chunk))
        offset += chunk_size

    # EOF record
    lines.append(make_ihex_line(0x01, 0x0000))
    return '\n'.join(lines) + '\n'


def verify(firmware, current):
    # Sector 0: needs erase
    found = False
    for i in range(0x0000, 0x4000):
        if firmware[i] & (~current[i] & 0xFF):
            found = True
            break
    assert found, "Sector 0 must require erase"

    # Sector 1: identical
    assert firmware[0x4000:0x8000] == current[0x4000:0x8000], "Sector 1 must be identical"

    # Sector 2: no erase needed
    for i in range(0x8000, 0xC000):
        assert (firmware[i] & (~current[i] & 0xFF)) == 0, \
            f"Sector 2 byte {i:#x} must not need erase"

    # Sector 3: needs erase
    assert (0x42 & (~0x24 & 0xFF)) != 0, "Sector 3 must require erase"

    # Sector 4: current is all 0xFF
    assert all(current[i] == 0xFF for i in range(0x10000, 0x20000)), \
        "Sector 4 must be erased"

    # Sector 5 covered portion: current is 0xFF
    assert all(current[i] == 0xFF for i in range(0x20000, FIRMWARE_SIZE)), \
        "Sector 5 covered portion must be erased"

    # Firmware size creates partial page
    sector5_bytes = FIRMWARE_SIZE - 0x20000  # 22728
    assert sector5_bytes % 1024 != 0, "Must have a partial last page"
    partial = sector5_bytes % 1024
    fill = 1024 - partial
    assert partial == 200, f"Partial page should have 200 data bytes, got {partial}"
    assert fill == 824, f"Fill should be 824 bytes, got {fill}"

    print("All verifications passed")
    print(f"  Firmware size: {FIRMWARE_SIZE}")
    print(f"  Sector 5 coverage: {sector5_bytes} bytes")
    print(f"  Last page: {partial} data + {fill} fill")


def verify_ihex(firmware, ihex_content, base_address):
    """Verify that IHEX content round-trips back to the original firmware."""
    byte_map = {}
    ba = 0

    for line in ihex_content.strip().split('\n'):
        line = line.strip()
        if not line.startswith(':'):
            continue
        byte_count = int(line[1:3], 16)
        address = int(line[3:7], 16)
        record_type = int(line[7:9], 16)
        payload = line[9:9 + byte_count * 2]

        if record_type == 0x00:
            for i in range(byte_count):
                val = int(payload[i*2:i*2+2], 16)
                byte_map[ba + address + i] = val
        elif record_type == 0x04:
            ext_addr = int(payload, 16)
            ba = ext_addr << 16
        elif record_type == 0x01:
            break

    min_addr = min(byte_map.keys())
    max_addr = max(byte_map.keys())
    assert min_addr == base_address, f"IHEX min addr {min_addr:#x} != expected {base_address:#x}"
    size = max_addr - min_addr + 1
    assert size == len(firmware), f"IHEX size {size} != firmware size {len(firmware)}"

    for i in range(len(firmware)):
        addr = base_address + i
        assert byte_map.get(addr) == firmware[i], \
            f"IHEX mismatch at {addr:#x}: {byte_map.get(addr):#x} != {firmware[i]:#x}"

    print("IHEX verification passed")


def main():
    os.makedirs("/app/output", exist_ok=True)

    firmware = make_firmware()
    current = make_current_flash(firmware)
    verify(firmware, current)

    ihex_content = firmware_to_ihex(firmware, FLASH_BASE)
    verify_ihex(firmware, ihex_content, FLASH_BASE)

    # Write raw binary (used by test suite for CRC verification)
    with open("/app/firmware.bin", "wb") as f:
        f.write(firmware)

    # Write Intel HEX (used by the planner)
    with open("/app/firmware.ihex", "w") as f:
        f.write(ihex_content)

    with open("/app/current_flash.bin", "wb") as f:
        f.write(current)

    config = {
        "current_flash_address": "0x08000000"
    }
    with open("/app/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Generated firmware.bin: {len(firmware)} bytes")
    print(f"Generated firmware.ihex: {len(ihex_content)} bytes")
    print(f"Generated current_flash.bin: {len(current)} bytes")
    print("Generated config.json")


if __name__ == "__main__":
    main()
