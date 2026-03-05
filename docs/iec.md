# IEC 61966-2-1 vs C0-Continuous Constants

## The problem

The sRGB transfer function is piecewise: a linear segment near black, a power
curve everywhere else. The two pieces must meet at a threshold. IEC 61966-2-1
chose constants that *almost* meet — but not quite.

The IEC spec uses threshold `0.04045` with offset `0.055` and scale `1.055`.
At the junction point, the linear segment evaluates to `0.003130804953560371`
and the power segment evaluates to `0.003130807283067685`. The gap is
**~2.3e-9** in the linearization direction and **~3.0e-8** in the encoding
direction.

This is invisible at 8-bit precision (the gap is ~600x smaller than one u8
level), but it creates a genuine discontinuity in the transfer function. For
high-precision workflows (16-bit, f32, or f64), the piecewise segments don't
connect.

## C0-continuous constants

The [moxcms](https://github.com/niclasberg/moxcms) project derived alternative
constants that maintain the same `12.92` linear scale and `2.4` gamma exponent
but adjust the offset to make the piecewise function exactly C0-continuous:

| Constant | IEC 61966-2-1 | C0 (moxcms) |
|----------|---------------|-------------|
| Offset (A) | 0.055 | 0.055010718947586600 |
| Scale (1+A) | 1.055 | 1.055010718947586600 |
| Gamma threshold | 0.04045 | 0.039293370676848 |
| Linear threshold | 0.003130804953560371 | 0.003041282560127521 |

With these constants, the gap at the junction drops from ~2.3e-9 to ~1.3e-18
(floating-point epsilon — effectively zero).

## When does it matter?

At **u8 precision**: never. Both constant sets produce identical 8-bit LUT
values for all 256 entries. The gap is invisible.

At **u16 precision**: barely. Near the threshold (around sRGB values
3200-3400 out of 65535), you might see a 1-level difference. In practice this
is below the noise floor of any real sensor or display.

At **f32/f64 precision**: the discontinuity shows up in derivatives and in
iterative algorithms that converge on the threshold region. If you're writing a
color management engine, ICC profile processor, or gamut mapping algorithm that
does repeated forward/inverse conversions near the junction, the C0 constants
give cleaner behavior.

## The gap across the threshold region

Both constant sets were evaluated at points near their respective thresholds.
The "gap" is `power_segment(x) - linear_segment(x)` — the signed difference
between what the power curve gives and what the linear segment gives at the
same input.

```
sRGB value   IEC gap (power-linear)    C0 gap (power-linear)
─────────    ──────────────────────    ─────────────────────
0.03800      +2.17e-07                 +9.59e-07
0.03929      -7.55e-07                 +1.30e-18  <── C0 threshold (zero gap)
0.04000      -4.76e-07                 +2.87e-07
0.04045      +2.33e-09                 +7.70e-07  <── IEC threshold
0.04080      +5.36e-07                 +1.31e-06
0.04200      +3.44e-06                 +4.22e-06
```

The IEC constants cross zero at `0.04045` (by construction), but the
transition isn't smooth — the gap changes sign abruptly. The C0 constants
cross zero at `0.03929`, and the transition is smooth because the function
is actually continuous there.

## What this crate does

By default, `linear-srgb` uses C0-continuous constants everywhere: the
`precise` powf-based functions, the `default` rational polynomial, the SIMD
paths, and all compile-time LUTs.

If you need exact IEC 61966-2-1 behavior (e.g., conformance testing against
the spec, or interop with software that implements the original constants),
enable the `iec` feature:

```toml
linear-srgb = { version = "0.6", features = ["iec"] }
```

Then use `linear_srgb::iec::srgb_to_linear` and `linear_srgb::iec::linear_to_srgb`.

## References

- IEC 61966-2-1:1999, "Multimedia systems and equipment — Colour measurement and management"
- [moxcms C0-continuous constants](https://github.com/niclasberg/moxcms) — derivation of the continuous constants
- [CSS Color Level 4, Section 10.2](https://drafts.csswg.org/css-color/#color-conversion-code) — also acknowledges the discontinuity
