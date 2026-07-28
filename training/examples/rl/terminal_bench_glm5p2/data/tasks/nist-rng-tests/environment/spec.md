# Statistical Test Specifications

Based on NIST SP 800-22 Rev 1a. All tests operate on a binary sequence ε = (ε₁, ..., εₙ), n = 1,000,000.
Significance level α = 0.01. A test **passes** if P-value ≥ α.
Bits are read **MSB-first** from each byte: bit 7 of byte 0 is ε₁, bit 6 is ε₂, ..., bit 0 is ε₈, bit 7 of byte 1 is ε₉, etc.

## 1. Frequency (Monobit) Test

Determines whether the number of ones and zeros are approximately equal.

    Sₙ = Σᵢ₌₁ⁿ (2εᵢ − 1)
    s_obs = |Sₙ| / √n
    P-value = erfc(s_obs / √2)

## 2. Block Frequency Test (M = 128)

Determines whether the frequency of ones in M-bit non-overlapping blocks deviates from M/2.

    N = ⌊n/M⌋
    For block i (i = 0, ..., N−1): πᵢ = (count of ones in block i) / M
    χ²(obs) = 4M · Σᵢ₌₀ᴺ⁻¹ (πᵢ − ½)²
    P-value = Q(N/2, χ²/2)

where Q(a, x) = Γ(a,x)/Γ(a) is the regularized upper incomplete gamma function.

## 3. Runs Test

Determines whether the oscillation between zeros and ones is too fast or too slow.

    π = (Σᵢ₌₁ⁿ εᵢ) / n

Pre-test: if |π − ½| ≥ 2/√n, return P-value = 0.

    V_obs = 1 + Σᵢ₌₁ⁿ⁻¹ r(i)    where r(i) = 1 if εᵢ ≠ εᵢ₊₁, else 0.
    P-value = erfc(|V_obs − 2nπ(1−π)| / (2√(2n) · π(1−π)))

## 4. Longest Run of Ones in a Block (M = 10000)

Determines whether the longest run of ones within M-bit blocks is consistent with randomness.

    N = ⌊n/M⌋, K = 6

Partition the sequence into N non-overlapping blocks of M bits. Find the longest run of consecutive ones in each block. Classify each block's longest run into bins:

| Bin v | Longest run range | πᵥ       |
|-------|-------------------|----------|
| 0     | ≤ 10              | 0.0882   |
| 1     | 11                | 0.2092   |
| 2     | 12                | 0.2483   |
| 3     | 13                | 0.1933   |
| 4     | 14                | 0.1208   |
| 5     | 15                | 0.0675   |
| 6     | ≥ 16              | 0.0727   |

    χ² = Σᵥ₌₀ᴷ (νᵥ − N·πᵥ)² / (N·πᵥ)

where νᵥ is the count of blocks in bin v.

    P-value = Q(K/2, χ²/2)

## 5. Discrete Fourier Transform (Spectral) Test

Detects periodic features in the sequence via spectral analysis.

1. Map to ±1: Xᵢ = 2εᵢ − 1
2. Apply DFT: S = DFT(X)
3. Compute modulus of the first n/2 elements: Mₖ = |Sₖ| for k = 0, ..., n/2 − 1
4. Compute 95% threshold: T = √(ln(1/0.05) · n)    [ln(20) ≈ 2.995732274]
5. Count observed peaks below threshold: N₀ = #{k : Mₖ < T}
6. Expected count: N₁ = 0.95 · n/2
7. Normalized difference: d = (N₀ − N₁) / √(n/2 · 0.95 · 0.05)
8. P-value = erfc(|d| / √2)

## 6. Serial Test (m = 16)

Tests the uniformity of overlapping m-bit patterns.

For each block length r ∈ {m, m−1, m−2}:
- Extend ε to a circular sequence by appending the first (r−1) bits
- Count the frequency of each overlapping r-bit pattern over n positions: countᵢ for i = 0, ..., 2ʳ−1
- Compute: ψ²ᵣ = (2ʳ/n) · Σᵢ countᵢ² − n

Then:

    ∇ψ² = ψ²_m − ψ²_{m−1}
    ∇²ψ² = ψ²_m − 2·ψ²_{m−1} + ψ²_{m−2}

    P₁ = Q(2^(m−2), ∇ψ²/2)
    P₂ = Q(2^(m−3), ∇²ψ²/2)

Report P-value = min(P₁, P₂).

## 7. Approximate Entropy Test (m = 10)

Tests the frequency of overlapping patterns of length m and m+1.

For each block length r ∈ {m, m+1}:
- Extend ε to a circular sequence by appending the first (r−1) bits
- Count overlapping r-bit patterns: Cᵢ = countᵢ / n  (relative frequency)
- Compute: φ(r) = Σᵢ Cᵢ · ln(Cᵢ)    [where 0·ln(0) = 0]

Then:

    ApEn = φ(m) − φ(m+1)
    χ² = 2n · (ln 2 − ApEn)
    P-value = Q(2^(m−1), χ²/2)
