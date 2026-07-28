# Flash Programming Plan Output Specification

## Overview

The flash planner generates a JSON plan at `/app/output/plan.json` describing
the sequence of flash operations needed to program firmware onto a target
microcontroller's flash memory. An optimal planner minimizes unnecessary
erase operations by exploiting the physical properties of flash memory.

## Output Format

```json
{
  "operations": [...],
  "statistics": {...}
}
```

### Operations

An ordered array of flash operations. Two operation types exist:

#### `erase_sector`

| Field | Type | Description |
|-------|------|-------------|
| type | string | `"erase_sector"` |
| address | string | Sector start address as hex string (e.g., `"0x08000000"`) with `0x` prefix and 8 uppercase hex digits |
| size | int | Sector size in bytes |
| algorithm | string | Flash algorithm name from target description |
| timeout_ms | int | Erase timeout from algorithm parameters |

#### `program_page`

| Field | Type | Description |
|-------|------|-------------|
| type | string | `"program_page"` |
| address | string | Page start address as hex string (same format as above) |
| size | int | Page size from algorithm parameters |
| algorithm | string | Flash algorithm name from target description |
| timeout_ms | int | Program timeout from algorithm parameters |
| data_crc32 | string | CRC32 of the complete page data (including any fill bytes), formatted as `"0x"` followed by 8 uppercase hexadecimal digits |
| fill_bytes | int | Number of padding bytes appended to fill a partial page |

### Statistics

| Field | Type | Description |
|-------|------|-------------|
| total_sectors_analyzed | int | Total sectors enumerated across all flash regions |
| sectors_erased | int | Number of sectors that required erasing |
| sectors_skipped_identical | int | Sectors where firmware matches current flash exactly |
| sectors_skipped_no_firmware | int | Sectors with no firmware data to program |
| sectors_programmed_without_erase | int | Sectors reprogrammed without prior erase |
| total_pages_programmed | int | Total program_page operations emitted |
| total_bytes_programmed | int | Total bytes across all programmed pages |
| fill_bytes_added | int | Total padding bytes added across all partial pages |
| estimated_time_ms | int | Sum of all operation timeout values |
| firmware_size | int | Total firmware image size in bytes |

### Ordering Constraints

- All operations must appear in ascending address order.
- For the same address, `erase_sector` must precede `program_page`.
- Each operation must reference the flash algorithm for the region containing its address.
- Partial pages at firmware boundaries are padded with the flash algorithm's erased byte value to fill the complete page size.
