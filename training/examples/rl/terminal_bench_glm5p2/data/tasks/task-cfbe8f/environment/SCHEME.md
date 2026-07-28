# K7M Affine-MBA Obfuscation Scheme

This directory contains LLVM IR files obfuscated with the K7M Affine-MBA scheme.

## Overview

The scheme replaces simple `i8` arithmetic operations (`add`, `sub`) with
semantically equivalent Mixed Boolean Arithmetic (MBA) expressions wrapped in
invertible affine layers.

## Structure

Each obfuscated operation follows this template:

    result = (mba_expr(a, b) * alpha + beta) * gamma + delta

where:

- `mba_expr(a, b)` is a purely bitwise expression equivalent to the original operation
- The affine layer `(x * alpha + beta) * gamma + delta` is invertible modulo 256
- Invertibility requires: `alpha * gamma = 1 (mod 256)` and `beta * gamma + delta = 0 (mod 256)`

## Variants

Two variants exist in the provided IR:

1. **Addition variant** — replaces `add i8 %a, %b`
2. **Subtraction variant** — replaces `sub i8 %a, %b`

Each variant uses a DIFFERENT MBA bitwise identity and DIFFERENT affine
constants (alpha, beta, gamma, delta). The subtraction identity is not a
trivial rearrangement of the addition identity.

## Notes

- All operations are on `i8` (8-bit integers, arithmetic modulo 256)
- Constants in LLVM IR may appear in signed form (values above 127 wrap to negative)
- Operand ordering may be commuted for commutative operations (add, mul, xor, and)
- `sub` is NOT commutative — operand order matters in the MBA identity
- Analyze `test_inputs/xvf_add.ll` and `test_inputs/xvf_sub.ll` for the simplest examples of each variant
