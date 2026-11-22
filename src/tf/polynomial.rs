//! Rational polynomial evaluation and transfer-kernel rounding policy.

use magetypes::{
    scalar::F32ScalarBackend,
    simd::{backends::*, generic::*},
};

/// Evaluate a rational polynomial P(x)/Q(x) using Horner's method.
///
/// Coefficients are stored lowest-degree-first: `p[0] + p[1]*x + p[2]*x^2 + ...`
#[inline(always)]
pub(crate) fn eval_rational_poly<T: F32ScalarBackend, const P: usize, const Q: usize>(
    token: T,
    x: f32,
    p: [f32; P],
    q: [f32; Q],
) -> f32 {
    let mut yp = p[P - 1];
    for i in (0..P - 1).rev() {
        yp = mla_scalar(token, yp, x, p[i]);
    }
    let mut yq = q[Q - 1];
    for i in (0..Q - 1).rev() {
        yq = mla_scalar(token, yq, x, q[i]);
    }
    yp / yq
}

/// Existing arithmetic contract: fused on native, opt-in guaranteed fusion on WASM.
/// Magetypes owns the hardware/engine probe and software fallback behind `_fused`.
pub(crate) const USE_FUSED_MATH: bool = !cfg!(all(
    target_arch = "wasm32",
    not(feature = "precise-portable")
));

#[inline(always)]
pub(crate) fn mla_scalar<T: F32ScalarBackend>(token: T, a: f32, b: f32, c: f32) -> f32 {
    if USE_FUSED_MATH {
        crate::mlaf::fmla_f32_with_token(token, a, b, c)
    } else {
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "relaxed-simd",
            not(feature = "precise-portable")
        ))]
        if token.as_scalar().is_some() {
            use archmage::SimdToken;
            if let Some(w) = archmage::Wasm128Token::summon() {
                return magetypes::scalar::mul_add_relaxed_f32(w, a, b, c);
            }
        }
        magetypes::scalar::mul_add_relaxed_f32(token, a, b, c)
    }
}

pub(crate) use crate::mlaf::hw_sqrt_f32 as fast_sqrt_f32;

// x86 and WASM can select the non-NaN right operand directly. NEON's
// numeric max can return a NaN for a signalling input, so use an ordered
// mask there to guarantee positive zero for every NaN.
macro_rules! floor_zero {
    ($value:ident, $zero:ident) => {
        if cfg!(target_arch = "aarch64") {
            $value & $value.simd_gt($zero)
        } else {
            const NON_NAN_RHS: u32 = magetypes::numerics::ExtremaOptions::new()
                .non_nan_rhs()
                .bits();
            $value.max_with::<NON_NAN_RHS>($zero)
        }
    };
}
pub(crate) use floor_zero;

/// Multiply-add and hardware square-root in the kernels. Fused wherever that costs nothing or the
/// scalar tier needs it, so every native tier gives the same bits. On WASM, `mul_add_fused` is
/// software unless the engine fuses, so it stays relaxed there unless `precise-portable` asks for
/// the same bits.
pub(crate) trait Mla: Sized {
    fn mla(self, a: Self, b: Self) -> Self;
    fn fast_sqrt(self) -> Self;
}

macro_rules! mla_for {
    ($vector:ident, $backend:ident) => {
        impl<T: $backend> Mla for $vector<T> {
            #[inline(always)]
            fn mla(self, a: Self, b: Self) -> Self {
                if USE_FUSED_MATH {
                    self.mul_add_fused(a, b)
                } else {
                    #[cfg(all(
                        target_arch = "wasm32",
                        target_feature = "relaxed-simd",
                        not(feature = "precise-portable")
                    ))]
                    if self.token().as_scalar().is_some() {
                        use archmage::SimdToken;
                        if let Some(w) = archmage::Wasm128Token::summon() {
                            let (sa, aa, ba) = (self.to_array(), a.to_array(), b.to_array());
                            let mut out = [0.0f32; _];
                            for i in 0..sa.len() {
                                out[i] =
                                    magetypes::scalar::mul_add_relaxed_f32(w, sa[i], aa[i], ba[i]);
                            }
                            return Self::from_array_t(self.token(), out);
                        }
                    }
                    self.mul_add_relaxed(a, b)
                }
            }

            #[inline(always)]
            fn fast_sqrt(self) -> Self {
                let t = self.token();
                if t.as_scalar().is_some() {
                    Self::from_array_t(t, self.to_array().map(crate::mlaf::hw_sqrt_f32))
                } else {
                    self.sqrt()
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
        #[allow(dead_code)]
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

macro_rules! blended_polynomial_for {
    ($name:ident, $float:ident, $backend:ident) => {
        #[cfg(feature = "transfer")]
        #[inline(always)]
        pub(crate) fn $name<T: $backend, const N: usize>(
            t: T,
            x: $float<T>,
            mask: $float<T>,
            when_true: [f32; N],
            when_false: [f32; N],
        ) -> $float<T> {
            let last_t = when_true[N - 1];
            let last_f = when_false[N - 1];
            let mut acc = if last_t.to_bits() == last_f.to_bits() {
                $float::splat_t(t, last_t)
            } else {
                $float::blend(mask, $float::splat_t(t, last_t), $float::splat_t(t, last_f))
            };
            let mut i = N - 1;
            while i > 0 {
                i -= 1;
                let ct = when_true[i];
                let cf = when_false[i];
                let c = if ct.to_bits() == cf.to_bits() {
                    $float::splat_t(t, ct)
                } else {
                    $float::blend(mask, $float::splat_t(t, ct), $float::splat_t(t, cf))
                };
                acc = acc.mla(x, c);
            }
            acc
        }
    };
}

blended_polynomial_for!(eval_blended_poly_x4, f32x4, F32x4Backend);
blended_polynomial_for!(eval_blended_poly_x8, f32x8, F32x8Backend);
blended_polynomial_for!(eval_blended_poly_x16, f32x16, F32x16Backend);
