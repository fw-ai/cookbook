# ADC0 — 12-bit Successive Approximation ADC

## Overview

The ADC0 peripheral is a 12-bit successive approximation analog-to-digital converter with 4 regular conversion channels, 4 injected conversion slots, and a programmable calibration engine. It was validated on silicon but its SVD definition was never written.

Base address: `0x40070000`
Address block size: `0x80`

## Global Control and Status

The ADC has three global registers at the start of its address space:

- **CR** (offset `0x00`, 32-bit, read-write): Master control register. Bits control ADC enable, conversion start, calibration trigger, and power-down mode.
- **CFGR** (offset `0x04`, 32-bit, read-write): Configuration register. Controls resolution (6/8/10/12-bit), data alignment, scan mode, and DMA request generation.
- **SR** (offset `0x08`, 32-bit, read-only): Status register. Flags for end-of-conversion (EOC), end-of-injected-conversion (JEOC), analog watchdog (AWD), and overrun (OVR).

## Data Output

The ADC conversion result is accessible at offset `0x0C` through two views of the same physical 32-bit register:

- **DR_LEFT**: Left-aligned view. The 12-bit result is shifted to bits [31:20], useful for DSP pipelines that treat samples as Q1.31 fixed-point values.
- **DR_RIGHT**: Right-aligned view. The 12-bit result occupies bits [11:0], the standard unsigned integer representation.

Both are 32-bit, read-only. They occupy the same address. In SVD, one view must reference the other as its alternate register to indicate they share an address intentionally.

## Per-Channel Configuration

The ADC has 4 identical regular conversion channels (numbered 0 through 3). Each channel has its own configuration block of 2 registers occupying 8 bytes. The blocks are laid out consecutively starting at offset `0x10`:

- Channel 0: offset `0x10`
- Channel 1: offset `0x18`
- Channel 2: offset `0x20`
- Channel 3: offset `0x28`

Each channel block contains:

| Register | Offset within block | Width | Access | Description |
|----------|-------------------|-------|--------|-------------|
| SQR      | `0x00`            | 32    | R/W    | Sequence position and analog input mux select |
| OFR      | `0x04`            | 16    | R/W    | Signed offset correction applied to raw conversion result |

When modeled in SVD, the channel register names in the expanded register map must follow the `CH<n>.<register>` pattern (e.g., `CH0.SQR`, `CH2.OFR`).

## Injected Conversion Registers

Four injected data registers hold results from the injected (high-priority) conversion group. They are identically-structured 32-bit read-only registers at consecutive 4-byte-aligned addresses:

| Register | Offset |
|----------|--------|
| JDR0     | `0x30` |
| JDR1     | `0x34` |
| JDR2     | `0x38` |
| JDR3     | `0x3C` |

## Calibration

- **CALFACT** (offset `0x40`, 32-bit, read-write): Calibration factor register. Written by the hardware calibration engine on completion, or can be written directly by software to restore a previously saved calibration value.
