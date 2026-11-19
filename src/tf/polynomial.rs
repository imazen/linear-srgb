//! Rational polynomial evaluation and transfer-kernel rounding policy.

use magetypes::simd::{backends::*, generic::*};
#[allow(unused_imports)]
use num_traits::Float;

/// Evaluate a rational polynomial P(x)/Q(x) using Horner's method.
///
/// Coefficients are stored lowest-degree-first: `p[0] + p[1]*x + p[2]*x^2 + ...`
#[inline(always)]
pub(crate) fn eval_rational_poly<const P: usize, const Q: usize>(
    x: f32,
    p: [f32; P],
    q: [f32; Q],
) -> f32 {
    let mut yp = p[P - 1];
    for i in (0..P - 1).rev() {
        yp = yp.mul_add(x, p[i]);
    }
    let mut yq = q[Q - 1];
    for i in (0..Q - 1).rev() {
        yq = yq.mul_add(x, q[i]);
    }
    yp / yq
}

/// Existing arithmetic contract: fused on native, opt-in guaranteed fusion on WASM.
/// Magetypes owns the hardware/engine probe and software fallback behind `_fused`.
pub(crate) const USE_FUSED_MATH: bool = !cfg!(all(
    target_arch = "wasm32",
    not(feature = "precise-portable")
));

/// Multiply-add in the kernels. Fused wherever that costs nothing or the scalar tier needs it, so
/// every native tier gives the same bits. On WASM, `mul_add_fused` is software unless the engine
/// fuses, so it stays relaxed there unless the `precise-portable` feature asks for the same bits.
pub(crate) trait Mla: Sized {
    fn mla(self, a: Self, b: Self) -> Self;
}

macro_rules! mla_for {
    ($vector:ident, $backend:ident) => {
        impl<T: $backend> Mla for $vector<T> {
            #[inline(always)]
            fn mla(self, a: Self, b: Self) -> Self {
                if USE_FUSED_MATH {
                    self.mul_add_fused(a, b)
                } else {
                    self.mul_add_relaxed(a, b)
                }
            }
        }
    };
}
mla_for!(f32x4, F32x4Backend);
mla_for!(f32x8, F32x8Backend);
mla_for!(f32x16, F32x16Backend);

macro_rules! polynomial_for {
    ($name:ident, $float:ident, $backend:ident) => {
        #[inline(always)]
        pub(crate) fn $name<T: $backend, const N: usize>(
            t: T,
            x: $float<T>,
            coefficients: [f32; N],
        ) -> $float<T> {
            let (&last, rest) = coefficients.split_last().expect("nonempty polynomial");
            rest.iter()
                .rev()
                .fold($float::splat_t(t, last), |acc, &coefficient| {
                    acc.mla(x, $float::splat_t(t, coefficient))
                })
        }
    };
}

polynomial_for!(eval_poly_x4, f32x4, F32x4Backend);
polynomial_for!(eval_poly_x8, f32x8, F32x8Backend);
polynomial_for!(eval_poly_x16, f32x16, F32x16Backend);
