# Performance Analysis

Benchmark results on x86_64 (exact CPU details vary by system).
All benchmarks process 10,000 values.

## Summary: Best Methods by Use Case

| Use Case | Recommended | Time/10K | Accuracy | Notes |
|----------|-------------|----------|----------|-------|
| u8 pipeline (speed) | imageflow LUT | 16 µs | Perfect u8 | Not suitable for u16 |
| u8 pipeline (quality) | LUT8 + SIMD | 46 µs | Perfect | Good balance |
| u16 pipeline | LUT16 + scalar_powf | 66 µs | Perfect | Best for 16-bit |
| f32 sRGB→Linear | scalar_powf | 5.8 µs | <1 ULP | Fastest overall |
| f32 Linear→sRGB | SIMD dirty_pow | 33 µs | <8 ULP | 1.6x faster than scalar |

## Detailed Results

### sRGB → Linear

#### f32 → f32

| Method | Time | Code |
|--------|------|------|
| **scalar_powf** | **5.8 µs** | `srgb_to_linear(v)` → `((v + 0.055) / 1.055).powf(2.4)` |
| imageflow_powf | 6.2 µs | `imageflow::srgb_to_linear(v)` → `((v + 0.055) / 1.055).powf(2.4)` (textbook constants) |
| simd_dirty_pow | 28.6 µs | `simd::srgb_to_linear_x8(v)` → `wide::f32x8` with `pow_f32x8` approximation |
| lut12_interp | 42.4 µs | `lut_interp_linear_float(v, table)` → 4096-entry table + linear interpolation |

**Why scalar beats SIMD here:**
- `powf(2.4)` is well-optimized by LLVM for scalar code
- The `wide` crate's `pow_f32x8` uses a polynomial approximation with overhead
- No FMA/AVX2 dispatch in current SIMD code (missing multiversion)

```rust
// Scalar (5.8 µs) - transfer.rs
pub fn srgb_to_linear(gamma: f32) -> f32 {
    if gamma < 0.0 {
        0.0
    } else if gamma < SRGB_LINEAR_THRESHOLD_F32 {
        gamma * LINEAR_SCALE_F32
    } else if gamma < 1.0 {
        ((gamma + SRGB_A_F32) / SRGB_A_PLUS_1_F32).powf(2.4)
    } else {
        1.0
    }
}

// SIMD (28.6 µs) - simd.rs
pub fn srgb_to_linear_x8(srgb: f32x8) -> f32x8 {
    let linear_part = srgb * LINEAR_SCALE_F32;
    let gamma_part = pow_f32x8(
        (srgb + SRGB_A_F32) / SRGB_A_PLUS_1_F32,
        f32x8::splat(2.4)
    );
    // ... blend based on threshold
}
```

#### u8 → f32

| Method | Time | Code |
|--------|------|------|
| **lut8_direct** | **3.3 µs** | `lut8.lookup(i as usize)` → 256-entry f32 table |
| imageflow_lut8 | 3.5 µs | `imageflow_lut.lookup(i)` → 256-entry f32 table |
| simd_lut8_batch | 4.0 µs | `simd::srgb_u8_to_linear_batch(&lut8, input, output)` |
| scalar_powf | 52 µs | `srgb_to_linear(i as f32 / 255.0)` |

```rust
// LUT lookup (3.3 µs) - lut.rs
pub fn lookup(&self, index: usize) -> f32 {
    self.table[index.min(N - 1)]
}

// Table generation
for (i, entry) in table.iter_mut().enumerate() {
    let srgb = i as f64 / 255.0;
    *entry = srgb_to_linear_f64(srgb) as f32;
}
```

#### u16 → f32

| Method | Time | Code |
|--------|------|------|
| **lut16_direct** | **3.5 µs** | `lut16.lookup(i as usize)` → 65536-entry f32 table (256KB) |
| lut8_quantized | 6.1 µs | `lut8.lookup((i >> 8) as usize)` → loses precision |
| scalar_powf | 52 µs | `srgb_to_linear(i as f32 / 65535.0)` |

---

### Linear → sRGB

#### f32 → f32

| Method | Time | Code |
|--------|------|------|
| **simd_dirty_pow** | **33 µs** | `simd::linear_to_srgb_x8(v)` → `wide::f32x8` with dirty_pow |
| imageflow_fastpow | 34 µs | `imageflow::linear_to_srgb(v)` → bit-manipulation pow approximation |
| scalar_powf | 53 µs | `linear_to_srgb(v)` → `v.powf(1.0/2.4)` |
| lut16_interp | 58 µs | `lut_interp_linear_float(v, encode16)` |
| lut12_interp | 79 µs | `lut_interp_linear_float(v, encode12)` |

**Why SIMD wins here:**
- `powf(1/2.4 ≈ 0.4167)` is harder for LLVM to optimize than `powf(2.4)`
- The dirty_pow approximation amortizes overhead across 8 values
- Imageflow's fastpow is competitive (similar algorithm, scalar)

```rust
// SIMD dirty_pow (33 µs) - simd.rs
pub fn linear_to_srgb_x8(linear: f32x8) -> f32x8 {
    let linear_part = linear * f32x8::splat(12.92);
    let gamma_part = fmla(
        f32x8::splat(SRGB_A_PLUS_1_F32),
        pow_f32x8(linear, f32x8::splat(INV_GAMMA_F32)),  // 1/2.4
        f32x8::splat(-SRGB_A_F32)
    );
    // ... blend
}

// Imageflow fastpow (34 µs) - imageflow.rs
fn fastpow(x: f32, p: f32) -> f32 {
    fastpow2(p * fastlog2(x))
}

fn fastpow2(p: f32) -> f32 {
    // Bit manipulation approximation
    let v: UnionU32F32 = UnionU32F32 {
        i: ((1 << 23) as f32 * (p + 121.274 + 27.728 / (4.843 - z) - 1.490 * z)) as u32,
    };
    unsafe { v.f }
}
```

#### f32 → u8

| Method | Time | Code |
|--------|------|------|
| **imageflow_lut16k** | **14 µs** | `imageflow::linear_to_srgb_lut(v)` → 16384-entry u8 table |
| imageflow_fastpow | 52 µs | `imageflow::linear_to_srgb_u8_fastpow(v)` |
| simd_dirty_pow | 60 µs | `simd::linear_to_srgb_u8_batch(input, output)` |
| scalar_powf | 79 µs | `(linear_to_srgb(v) * 255.0 + 0.5) as u8` |
| lut12_interp | 80 µs | `(lut_interp_linear_float(v, encode12) * 255.0 + 0.5) as u8` |

```rust
// Imageflow LUT16K (14 µs) - imageflow.rs
pub fn linear_to_srgb_lut(linear: f32) -> u8 {
    let idx = (linear * 16383.0).clamp(0.0, 16383.0) as usize;
    LINEAR_TO_SRGB_LUT[idx]  // 16384-entry u8 table
}

// SIMD batch (60 µs) - simd.rs
pub fn linear_to_srgb_u8_x8(linear: f32x8) -> [u8; 8] {
    let srgb = linear_to_srgb_x8(linear);
    let scaled = srgb * f32x8::splat(255.0) + f32x8::splat(0.5);
    // ... convert to u8 array
}
```

#### f32 → u16

| Method | Time | Code |
|--------|------|------|
| **simd_dirty_pow** | **51 µs** | SIMD linear_to_srgb then `* 65535 + 0.5` |
| scalar_powf | 66 µs | `(linear_to_srgb(v) * 65535.0 + 0.5) as u16` |
| lut16_interp | 81 µs | `(lut_interp_linear_float(v, encode16) * 65535.0 + 0.5) as u16` |

---

### Roundtrip Benchmarks

#### u8 → f32 → u8

| Method | Time | Accuracy | Code Path |
|--------|------|----------|-----------|
| **imageflow_lut+lut16k** | **16 µs** | Perfect u8 | `lookup(i)` → `linear_to_srgb_lut(linear)` |
| lut8+simd | 46 µs | Perfect | `lookup(i)` → `linear_to_srgb_u8_batch` |
| lut8+scalar_powf | 70 µs | Perfect | `lookup(i)` → `(linear_to_srgb(v) * 255 + 0.5) as u8` |
| lut8+lut12 | 93 µs | Perfect | `lookup(i)` → `lut_interp_linear_float` → u8 |
| imageflow_lut+fastpow | 87 µs | Perfect | `lookup(i)` → `linear_to_srgb_u8_fastpow` |

#### u16 → f32 → u16

| Method | Time | Accuracy | Code Path |
|--------|------|----------|-----------|
| **lut16+scalar_powf** | **66 µs** | Perfect | `lut16.lookup(i)` → `(linear_to_srgb(v) * 65535 + 0.5) as u16` |
| lut16+lut16 | 78 µs | ±1 max | `lut16.lookup(i)` → `lut_interp_linear_float` → u16 |
| scalar_powf_both | 103 µs | Perfect | `srgb_to_linear(i/65535)` → `linear_to_srgb` → u16 |

---

## Observations

### Why Scalar Beats SIMD for sRGB→Linear

1. **LLVM optimizes `powf(2.4)` well** - The exponent 2.4 allows efficient scalar code generation
2. **SIMD pow overhead** - `wide::pow_f32x8` uses a polynomial approximation with setup cost
3. **No multiversion dispatch** - Current code doesn't enable AVX2/FMA at runtime
4. **Branch prediction** - Scalar code's threshold branch is well-predicted

### Why SIMD Wins for Linear→sRGB

1. **`powf(0.4167)` is harder** - The fractional exponent 1/2.4 is less optimizable
2. **Amortized overhead** - Processing 8 values amortizes the pow approximation setup
3. **Imageflow's fastpow** - Similar speed, shows bit-manipulation approach is effective

### LUT Trade-offs

| Table | Size | Speed | Accuracy |
|-------|------|-------|----------|
| 8-bit (256) | 1 KB | Fastest | Perfect for u8 input |
| 12-bit (4096) | 16 KB | Medium | ±1 at u16 |
| 16-bit (65536) | 256 KB | Medium | Perfect for u16 |
| 16K (16384) u8 | 16 KB | Fast | u8 output only |

### Recommendations

1. **For 8-bit pipelines**: Use imageflow's LUT approach (fastest)
2. **For 16-bit pipelines**: Use LUT16 + scalar_powf (perfect accuracy)
3. **For f32 pipelines**: Use scalar for sRGB→Linear, SIMD for Linear→sRGB
4. **Avoid**: LUT interpolation when you need exact values

---

## Code References

- `src/transfer.rs` - Scalar powf implementations
- `src/simd.rs` - SIMD dirty_pow implementations
- `src/lut.rs` - LUT tables and interpolation
- `src/imageflow.rs` - Imageflow's fastpow and LUT16K
- `benches/benchmarks.rs` - All benchmark code
