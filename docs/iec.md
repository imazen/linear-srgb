# sRGB curve constants

The default and `precise` modules use the continuous curve also used by
[moxcms’s built-in sRGB transfer](https://github.com/awxkee/moxcms/blob/master/src/gamma.rs).
The `iec` module uses the textbook offset and threshold.

| Constant | Textbook sRGB | Default continuous curve |
|---|---:|---:|
| Offset | 0.055 | 0.0550107189475866 |
| Scale | 1.055 | 1.0550107189475866 |
| Encoded threshold | 0.04045 | 0.039293370676848 |
| Linear threshold | 0.0031308 | 0.003041282560127521 |

Changing the offset changes the entire power branch, not just the junction.
The default f32 table differs from the textbook curve at 244 of 256 u8 codes.
Code 12 has the largest relative difference, 2.28e-4 (3,595 f32 ULP).
All 256 default u8 decode→encode round trips return the original code.
That does not establish equality of intermediate linear values or encoded
values after filtering. See the [precision review](precision.md).

[ICC’s sRGB interpretation](https://registry.color.org/rgb-registry/files/sRGB.pdf)
and [CSS Color 4](https://www.w3.org/TR/css-color/#color-conversion-code) use
the textbook curve. An ICC transform must honor its actual profile’s TRC
parameters or samples; a CMS’s built-in sRGB helper does not define every
profile it processes.

For textbook interoperation, enable `iec` and call
`linear_srgb::iec::srgb_to_linear` / `linear_srgb::iec::linear_to_srgb`.
These evaluate floating-point formulae; they do not guarantee correctly
rounded transcendentals or cross-platform libm identity.
The current patch retains the continuous default. Switching curve families
needs a separate migration decision.
