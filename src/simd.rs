//! SIMD-accelerated sRGB ↔ linear conversion.
//!
//! This module provides high-performance conversion functions using AVX2/SSE SIMD
//! instructions via the `wide` crate with runtime CPU feature detection.
//!
//! # API Overview
//!
//! ## x8 Functions (process 8 values at once)
//! - [`srgb_to_linear_x8`] - f32x8 sRGB → f32x8 linear
//! - [`linear_to_srgb_x8`] - f32x8 linear → f32x8 sRGB
//! - [`srgb_u8_to_linear_x8`] - \[u8; 8\] sRGB → f32x8 linear
//! - [`linear_to_srgb_u8_x8`] - f32x8 linear → \[u8; 8\] sRGB
//!
//! ## Slice Functions (process entire slices)
//! - [`srgb_to_linear_slice`] - &mut \[f32\] sRGB → linear in-place
//! - [`linear_to_srgb_slice`] - &mut \[f32\] linear → sRGB in-place
//! - [`srgb_u8_to_linear_slice`] - &\[u8\] sRGB → &mut \[f32\] linear
//! - [`linear_to_srgb_u8_slice`] - &\[f32\] linear → &mut \[u8\] sRGB

use multiversed::multiversed;
use wide::{CmpLt, f32x8};

use crate::fast_math::pow_x8;

// sRGB transfer function constants (IEC 61966-2-1)
const SRGB_LINEAR_THRESHOLD: f32x8 = f32x8::splat(0.039_293_37);
const LINEAR_THRESHOLD: f32x8 = f32x8::splat(0.003_041_282_6);
const LINEAR_SCALE: f32x8 = f32x8::splat(1.0 / 12.92);
const SRGB_OFFSET: f32x8 = f32x8::splat(0.055);
const SRGB_SCALE: f32x8 = f32x8::splat(1.055);
const TWELVE_92: f32x8 = f32x8::splat(12.92);
const ZERO: f32x8 = f32x8::splat(0.0);
const ONE: f32x8 = f32x8::splat(1.0);
const U8_MAX: f32x8 = f32x8::splat(255.0);
const HALF: f32x8 = f32x8::splat(0.5);

/// Precomputed sRGB u8 → linear f32 lookup table.
/// Uses the same constants as the transfer module (C0-continuous IEC 61966-2-1).
/// Computed at compile time to embed in binary.
const SRGB_U8_TO_LINEAR_LUT: [f32; 256] = [
    0e0_f32, 3.035269835488375e-4_f32, 6.07053967097675e-4_f32, 9.105809506465125e-4_f32,
    1.21410793419535e-3_f32, 1.5176349177441874e-3_f32, 1.821161901293025e-3_f32, 2.1246888848418626e-3_f32,
    2.4282158683907e-3_f32, 2.7317428519395373e-3_f32, 3.035269835488375e-3_f32, 3.3473314611229448e-3_f32,
    3.677344441366452e-3_f32, 4.0255958774635e-3_f32, 4.3923629356734835e-3_f32, 4.777916693130672e-3_f32,
    5.182522482094411e-3_f32, 5.606440203642381e-3_f32, 6.0499246144396545e-3_f32, 6.513225589681857e-3_f32,
    6.996588364869239e-3_f32, 7.500253758701098e-3_f32, 8.02445837907288e-3_f32, 8.569434813899632e-3_f32,
    9.135411808271226e-3_f32, 9.722614429258631e-3_f32, 1.033126421953247e-2_f32, 1.0961579340818595e-2_f32,
    1.1613774708098979e-2_f32, 1.2288062115364455e-2_f32, 1.2984650353638553e-2_f32, 1.3703745321915007e-2_f32,
    1.4445550131584857e-2_f32, 1.5210265204870508e-2_f32, 1.5998088367732796e-2_f32, 1.680921493767176e-2_f32,
    1.7643837806801566e-2_f32, 1.8502147520544895e-2_f32, 1.9384332352260048e-2_f32, 2.029057837408647e-2_f32,
    2.1221069524268714e-2_f32, 2.217598767119691e-2_f32, 2.3155512674381193e-2_f32, 2.4159822442559955e-2_f32,
    2.5189092989124896e-2_f32, 2.624349848503163e-2_f32, 2.7323211309351038e-2_f32, 2.842840209760458e-2_f32,
    2.9559239788015824e-2_f32, 3.0715891665800646e-2_f32, 3.189852340560932e-2_f32, 3.31072991122257e-2_f32,
    3.4342381359620935e-2_f32, 3.560393122845253e-2_f32, 3.6892108342093125e-2_f32, 3.820707090126763e-2_f32,
    3.954897571737209e-2_f32, 4.091797824454262e-2_f32, 4.2314232610538585e-2_f32, 4.373789164649983e-2_f32,
    4.518910691563392e-2_f32, 4.666802874088623e-2_f32, 4.8174806231641464e-2_f32, 4.9709587309503984e-2_f32,
    5.127251873319947e-2_f32, 5.2863746122639076e-2_f32, 5.448341398218497e-2_f32, 5.613166572315294e-2_f32,
    5.780864368558672e-2_f32, 5.9514489159335944e-2_f32, 6.124934240446855e-2_f32, 6.301334267104575e-2_f32,
    6.480662821828785e-2_f32, 6.662933633315535e-2_f32, 6.848160334837076e-2_f32, 7.036356465990364e-2_f32,
    7.227535474394083e-2_f32, 7.421710717336279e-2_f32, 7.618895463374566e-2_f32, 7.819102893890731e-2_f32,
    8.022346104601631e-2_f32, 8.228638107027908e-2_f32, 8.437991829922274e-2_f32, 8.65042012065882e-2_f32,
    8.865935746584809e-2_f32, 9.084551396336377e-2_f32, 9.306279681119425e-2_f32, 9.53113313595697e-2_f32,
    9.759124220904188e-2_f32, 9.99026532223228e-2_f32, 1.0224568753582215e-1_f32, 1.0462046757089492e-1_f32,
    1.0702711504480827e-1_f32, 1.0946575098143785e-1_f32, 1.119364957217027e-1_f32, 1.1443946893374674e-1_f32,
    1.1697478962287669e-1_f32, 1.1954257614126311e-1_f32, 1.2214294619741281e-1_f32, 1.2477601686542021e-1_f32,
    1.2744190459400445e-1_f32, 1.3014072521533915e-1_f32, 1.328725939536815e-1_f32, 1.3563762543380634e-1_f32,
    1.3843593368925236e-1_f32, 1.4126763217038515e-1_f32, 1.441328337522831e-1_f32, 1.4703165074245136e-1_f32,
    1.4996419488836965e-1_f32, 1.5293057738487753e-1_f32, 1.559309088814034e-1_f32, 1.589652994890404e-1_f32,
    1.62033858787475e-1_f32, 1.6513669583177162e-1_f32, 1.6827391915901724e-1_f32, 1.7144563679483038e-1_f32,
    1.7465195625973928e-1_f32, 1.7789298457543018e-1_f32, 1.8116882827087194e-1_f32, 1.8447959338831948e-1_f32,
    1.8782538548919864e-1_f32, 1.9120630965987687e-1_f32, 1.9462247051732165e-1_f32, 1.9807397221465087e-1_f32,
    2.0156091844657667e-1_f32, 2.0508341245474676e-1_f32, 2.0864155703298495e-1_f32, 2.1223545453243392e-1_f32,
    2.1586520686660335e-1_f32, 2.1953091551632387e-1_f32, 2.2323268153461176e-1_f32, 2.269706055514448e-1_f32,
    2.3074478777845225e-1_f32, 2.3455532801352041e-1_f32, 2.3840232564531783e-1_f32, 2.4228587965773846e-1_f32,
    2.4620608863426877e-1_f32, 2.5016305076227785e-1_f32, 2.541568638372337e-1_f32, 2.5818762526684713e-1_f32,
    2.6225543207514446e-1_f32, 2.663603809064713e-1_f32, 2.705025680294297e-1_f32, 2.7468208934074806e-1_f32,
    2.7889904036908775e-1_f32, 2.831535162787864e-1_f32, 2.874456118735393e-1_f32, 2.917754216000216e-1_f32,
    2.961430395514514e-1_f32, 3.0054855947109427e-1_f32, 3.0499207475571455e-1_f32, 3.0947367845896817e-1_f32,
    3.1399346329474426e-1_f32, 3.1855152164045336e-1_f32, 3.2314794554026405e-1_f32, 3.277828267082896e-1_f32,
    3.3245625653172595e-1_f32, 3.3716832607393943e-1_f32, 3.419191260775102e-1_f32, 3.467087469672274e-1_f32,
    3.5153727885303987e-1_f32, 3.564048115329629e-1_f32, 3.6131143449594066e-1_f32, 3.6625723692466716e-1_f32,
    3.712423076983654e-1_f32, 3.7626673539552463e-1_f32, 3.8133060829660037e-1_f32, 3.8643401438667196e-1_f32,
    3.915770413580644e-1_f32, 3.967597766129312e-1_f32, 4.0198230726580075e-1_f32, 4.0724472014608687e-1_f32,
    4.125471018005641e-1_f32, 4.178895384958065e-1_f32, 4.232721162205954e-1_f32, 4.286949206882912e-1_f32,
    4.341580373391733e-1_f32, 4.39661551342749e-1_f32, 4.4520554760002823e-1_f32, 4.5079011074577e-1_f32,
    4.564153251506974e-1_f32, 4.6208127492368195e-1_f32, 4.677880439139006e-1_f32, 4.7353571571296305e-1_f32,
    4.793243736570098e-1_f32, 4.8515410082878474e-1_f32, 4.9102498005967876e-1_f32, 4.969370939317475e-1_f32,
    5.028905247797038e-1_f32, 5.088853546928813e-1_f32, 5.149216655171772e-1_f32, 5.209995388569664e-1_f32,
    5.271190560769934e-1_f32, 5.332802983042395e-1_f32, 5.39483346429767e-1_f32, 5.457282811105392e-1_f32,
    5.520151827712187e-1_f32, 5.583441316059429e-1_f32, 5.647152075800775e-1_f32, 5.711284904319499e-1_f32,
    5.775840596745581e-1_f32, 5.840819945972618e-1_f32, 5.906223742674525e-1_f32, 5.972052775322011e-1_f32,
    6.038307830198896e-1_f32, 6.104989691418194e-1_f32, 6.172099140938033e-1_f32, 6.239636958577368e-1_f32,
    6.307603922031522e-1_f32, 6.376000806887527e-1_f32, 6.44482838663932e-1_f32, 6.514087432702708e-1_f32,
    6.583778714430215e-1_f32, 6.653902999125717e-1_f32, 6.724461052058939e-1_f32, 6.795453636479757e-1_f32,
    6.866881513632371e-1_f32, 6.938745442769281e-1_f32, 7.011046181165127e-1_f32, 7.083784484130379e-1_f32,
    7.156961105024852e-1_f32, 7.230576795271089e-1_f32, 7.304632304367586e-1_f32, 7.379128379901879e-1_f32,
    7.454065767563474e-1_f32, 7.529445211156647e-1_f32, 7.605267452613104e-1_f32, 7.681533232004496e-1_f32,
    7.758243287554806e-1_f32, 7.835398355652586e-1_f32, 7.912999170863098e-1_f32, 7.991046465940284e-1_f32,
    8.069540971838643e-1_f32, 8.148483417724954e-1_f32, 8.227874530989894e-1_f32, 8.307715037259527e-1_f32,
    8.388005660406673e-1_f32, 8.468747122562166e-1_f32, 8.549940144125975e-1_f32, 8.631585443778235e-1_f32,
    8.713683738490147e-1_f32, 8.796235743534773e-1_f32, 8.879242172497716e-1_f32, 8.962703737287706e-1_f32,
    9.046621148147056e-1_f32, 9.130995113662014e-1_f32, 9.215826340773045e-1_f32, 9.301115534784967e-1_f32,
    9.38686339937702e-1_f32, 9.473070636612796e-1_f32, 9.559737946950111e-1_f32, 9.646866029250757e-1_f32,
    9.734455580790162e-1_f32, 9.822507297266954e-1_f32, 9.911021872812447e-1_f32, 1e0_f32,
];

#[inline]
fn get_lut() -> &'static [f32; 256] {
    &SRGB_U8_TO_LINEAR_LUT
}

// ============================================================================
// x8 Functions - Process 8 values at once
// ============================================================================

/// Convert 8 sRGB f32 values to linear.
///
/// Input values are clamped to \[0, 1\].
///
/// # Example
/// ```
/// use linear_srgb::simd::srgb_to_linear_x8;
/// use wide::f32x8;
///
/// let srgb = f32x8::from([0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.5]);
/// let linear = srgb_to_linear_x8(srgb);
/// ```
#[multiversed]
#[inline]
pub fn srgb_to_linear_x8(srgb: f32x8) -> f32x8 {
    let srgb = srgb.max(ZERO).min(ONE);
    let linear_result = srgb * LINEAR_SCALE;
    let power_result = pow_x8((srgb + SRGB_OFFSET) / SRGB_SCALE, 2.4);
    let mask = srgb.simd_lt(SRGB_LINEAR_THRESHOLD);
    mask.blend(linear_result, power_result)
}

/// Convert 8 linear f32 values to sRGB.
///
/// Input values are clamped to \[0, 1\].
///
/// # Example
/// ```
/// use linear_srgb::simd::linear_to_srgb_x8;
/// use wide::f32x8;
///
/// let linear = f32x8::from([0.0, 0.1, 0.2, 0.5, 1.0, 0.01, 0.05, 0.8]);
/// let srgb = linear_to_srgb_x8(linear);
/// ```
#[multiversed]
#[inline]
pub fn linear_to_srgb_x8(linear: f32x8) -> f32x8 {
    let linear = linear.max(ZERO).min(ONE);
    let linear_result = linear * TWELVE_92;
    let power_result = SRGB_SCALE * pow_x8(linear, 1.0 / 2.4) - SRGB_OFFSET;
    let mask = linear.simd_lt(LINEAR_THRESHOLD);
    mask.blend(linear_result, power_result)
}

/// Convert 8 sRGB u8 values to linear f32 using LUT lookup.
///
/// This is the fastest method for u8 input as it uses a precomputed lookup table.
///
/// # Example
/// ```
/// use linear_srgb::simd::srgb_u8_to_linear_x8;
///
/// let srgb = [0u8, 64, 128, 192, 255, 32, 96, 160];
/// let linear = srgb_u8_to_linear_x8(srgb);
/// ```
#[inline]
pub fn srgb_u8_to_linear_x8(srgb: [u8; 8]) -> f32x8 {
    let lut = get_lut();
    f32x8::from([
        lut[srgb[0] as usize],
        lut[srgb[1] as usize],
        lut[srgb[2] as usize],
        lut[srgb[3] as usize],
        lut[srgb[4] as usize],
        lut[srgb[5] as usize],
        lut[srgb[6] as usize],
        lut[srgb[7] as usize],
    ])
}

/// Convert 8 linear f32 values to sRGB u8.
///
/// Input values are clamped to \[0, 1\], output is rounded to nearest u8.
///
/// # Example
/// ```
/// use linear_srgb::simd::linear_to_srgb_u8_x8;
/// use wide::f32x8;
///
/// let linear = f32x8::from([0.0, 0.1, 0.2, 0.5, 1.0, 0.01, 0.05, 0.8]);
/// let srgb = linear_to_srgb_u8_x8(linear);
/// ```
#[multiversed]
#[inline]
pub fn linear_to_srgb_u8_x8(linear: f32x8) -> [u8; 8] {
    let srgb = linear_to_srgb_x8(linear);
    let scaled = srgb * U8_MAX + HALF;
    let arr: [f32; 8] = scaled.into();
    [
        arr[0] as u8,
        arr[1] as u8,
        arr[2] as u8,
        arr[3] as u8,
        arr[4] as u8,
        arr[5] as u8,
        arr[6] as u8,
        arr[7] as u8,
    ]
}

// ============================================================================
// Slice Functions - Process entire slices
// ============================================================================

/// Convert sRGB f32 values to linear in-place.
///
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
///
/// # Example
/// ```
/// use linear_srgb::simd::srgb_to_linear_slice;
///
/// let mut values = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
/// srgb_to_linear_slice(&mut values);
/// ```
#[multiversed]
#[inline]
pub fn srgb_to_linear_slice(values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let result = srgb_to_linear_x8(f32x8::from(*chunk));
        *chunk = result.into();
    }

    for v in remainder {
        *v = crate::srgb_to_linear(*v);
    }
}

/// Convert linear f32 values to sRGB in-place.
///
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
///
/// # Example
/// ```
/// use linear_srgb::simd::linear_to_srgb_slice;
///
/// let mut values = vec![0.0f32, 0.1, 0.2, 0.5, 1.0];
/// linear_to_srgb_slice(&mut values);
/// ```
#[multiversed]
#[inline]
pub fn linear_to_srgb_slice(values: &mut [f32]) {
    let (chunks, remainder) = values.as_chunks_mut::<8>();

    for chunk in chunks {
        let result = linear_to_srgb_x8(f32x8::from(*chunk));
        *chunk = result.into();
    }

    for v in remainder {
        *v = crate::linear_to_srgb(*v);
    }
}

/// Convert sRGB u8 values to linear f32.
///
/// Uses a precomputed LUT for each u8 value, processed in SIMD batches of 8.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
///
/// # Example
/// ```
/// use linear_srgb::simd::srgb_u8_to_linear_slice;
///
/// let input: Vec<u8> = (0..=255).collect();
/// let mut output = vec![0.0f32; 256];
/// srgb_u8_to_linear_slice(&input, &mut output);
/// ```
#[inline]
pub fn srgb_u8_to_linear_slice(input: &[u8], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let lut = get_lut();

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = [
            lut[inp[0] as usize],
            lut[inp[1] as usize],
            lut[inp[2] as usize],
            lut[inp[3] as usize],
            lut[inp[4] as usize],
            lut[inp[5] as usize],
            lut[inp[6] as usize],
            lut[inp[7] as usize],
        ];
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        *out = lut[*inp as usize];
    }
}

/// Convert linear f32 values to sRGB u8.
///
/// Processes 8 values at a time using SIMD, with scalar fallback for remainder.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
///
/// # Example
/// ```
/// use linear_srgb::simd::linear_to_srgb_u8_slice;
///
/// let input: Vec<f32> = (0..=255).map(|i| i as f32 / 255.0).collect();
/// let mut output = vec![0u8; 256];
/// linear_to_srgb_u8_slice(&input, &mut output);
/// ```
#[multiversed]
#[inline]
pub fn linear_to_srgb_u8_slice(input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());

    let (in_chunks, in_remainder) = input.as_chunks::<8>();
    let (out_chunks, out_remainder) = output.as_chunks_mut::<8>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        *out = linear_to_srgb_u8_x8(f32x8::from(*inp));
    }

    for (inp, out) in in_remainder.iter().zip(out_remainder.iter_mut()) {
        let srgb = crate::linear_to_srgb(*inp);
        *out = (srgb * 255.0 + 0.5) as u8;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- x8 function tests ----

    #[test]
    fn test_srgb_to_linear_x8() {
        let input = [0.0f32, 0.25, 0.5, 0.75, 1.0, 0.1, 0.9, 0.04];
        let result = srgb_to_linear_x8(f32x8::from(input));
        let result_arr: [f32; 8] = result.into();

        for (i, &inp) in input.iter().enumerate() {
            let expected = crate::srgb_to_linear(inp);
            assert!(
                (result_arr[i] - expected).abs() < 1e-5,
                "srgb_to_linear_x8 mismatch at {}: got {}, expected {}",
                i,
                result_arr[i],
                expected
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_x8() {
        let input = [0.0f32, 0.1, 0.2, 0.5, 1.0, 0.01, 0.001, 0.8];
        let result = linear_to_srgb_x8(f32x8::from(input));
        let result_arr: [f32; 8] = result.into();

        for (i, &inp) in input.iter().enumerate() {
            let expected = crate::linear_to_srgb(inp);
            assert!(
                (result_arr[i] - expected).abs() < 1e-5,
                "linear_to_srgb_x8 mismatch at {}: got {}, expected {}",
                i,
                result_arr[i],
                expected
            );
        }
    }

    #[test]
    fn test_srgb_u8_to_linear_x8() {
        let input: [u8; 8] = [0, 64, 128, 192, 255, 32, 96, 160];
        let result = srgb_u8_to_linear_x8(input);
        let result_arr: [f32; 8] = result.into();

        for (i, &inp) in input.iter().enumerate() {
            let expected = crate::srgb_u8_to_linear(inp);
            assert!(
                (result_arr[i] - expected).abs() < 1e-6,
                "srgb_u8_to_linear_x8 mismatch at {}: got {}, expected {}",
                i,
                result_arr[i],
                expected
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_u8_x8() {
        let input = [0.0f32, 0.1, 0.2, 0.5, 1.0, 0.01, 0.05, 0.8];
        let result = linear_to_srgb_u8_x8(f32x8::from(input));

        for (i, &inp) in input.iter().enumerate() {
            let expected = (crate::linear_to_srgb(inp) * 255.0 + 0.5) as u8;
            assert!(
                (result[i] as i16 - expected as i16).abs() <= 1,
                "linear_to_srgb_u8_x8 mismatch at {}: got {}, expected {}",
                i,
                result[i],
                expected
            );
        }
    }

    // ---- Slice function tests ----

    #[test]
    fn test_srgb_to_linear_slice() {
        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let expected: Vec<f32> = values.iter().map(|&v| crate::srgb_to_linear(v)).collect();

        srgb_to_linear_slice(&mut values);

        for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "srgb_to_linear_slice mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_slice() {
        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let expected: Vec<f32> = values.iter().map(|&v| crate::linear_to_srgb(v)).collect();

        linear_to_srgb_slice(&mut values);

        for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "linear_to_srgb_slice mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_srgb_u8_to_linear_slice() {
        let input: Vec<u8> = (0..=255).collect();
        let mut output = vec![0.0f32; 256];

        srgb_u8_to_linear_slice(&input, &mut output);

        for (i, &out) in output.iter().enumerate() {
            let expected = crate::srgb_u8_to_linear(i as u8);
            assert!(
                (out - expected).abs() < 1e-6,
                "srgb_u8_to_linear_slice mismatch at {}: got {}, expected {}",
                i,
                out,
                expected
            );
        }
    }

    #[test]
    fn test_linear_to_srgb_u8_slice() {
        let input: Vec<f32> = (0..=255).map(|i| i as f32 / 255.0).collect();
        let mut output = vec![0u8; 256];

        linear_to_srgb_u8_slice(&input, &mut output);

        for i in 0..256 {
            let expected = (crate::linear_to_srgb(input[i]) * 255.0 + 0.5) as u8;
            assert!(
                (output[i] as i16 - expected as i16).abs() <= 1,
                "linear_to_srgb_u8_slice mismatch at {}: got {}, expected {}",
                i,
                output[i],
                expected
            );
        }
    }

    // ---- Roundtrip tests ----

    #[test]
    fn test_f32_roundtrip() {
        let mut values: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let original = values.clone();

        srgb_to_linear_slice(&mut values);
        linear_to_srgb_slice(&mut values);

        for (i, (&orig, &conv)) in original.iter().zip(values.iter()).enumerate() {
            assert!(
                (orig - conv).abs() < 1e-4,
                "f32 roundtrip failed at {}: {} -> {}",
                i,
                orig,
                conv
            );
        }
    }

    #[test]
    fn test_u8_roundtrip() {
        let input: Vec<u8> = (0..=255).collect();
        let mut linear = vec![0.0f32; 256];
        let mut back = vec![0u8; 256];

        srgb_u8_to_linear_slice(&input, &mut linear);
        linear_to_srgb_u8_slice(&linear, &mut back);

        for i in 0..256 {
            assert!(
                (input[i] as i16 - back[i] as i16).abs() <= 1,
                "u8 roundtrip failed at {}: {} -> {} -> {}",
                i,
                input[i],
                linear[i],
                back[i]
            );
        }
    }

    // ---- Edge case tests ----

    #[test]
    fn test_clamping() {
        // Test that out-of-range values are clamped
        let input = f32x8::from([-0.5, -0.1, 0.0, 0.5, 1.0, 1.5, 2.0, 10.0]);
        let result = srgb_to_linear_x8(input);
        let arr: [f32; 8] = result.into();

        assert_eq!(arr[0], 0.0, "negative should clamp to 0");
        assert_eq!(arr[1], 0.0, "negative should clamp to 0");
        assert!(arr[4] > 0.99 && arr[4] <= 1.0, "1.0 should stay ~1.0");
        assert!(arr[5] > 0.99 && arr[5] <= 1.0, "values > 1 should clamp");
    }

    #[test]
    fn test_linear_segment() {
        // Test values in the linear segment (< 0.04045)
        let input = f32x8::from([0.0, 0.01, 0.02, 0.03, 0.04, 0.005, 0.015, 0.035]);
        let result = srgb_to_linear_x8(input);
        let arr: [f32; 8] = result.into();
        let input_arr: [f32; 8] = input.into();

        for i in 0..8 {
            let expected = input_arr[i] / 12.92;
            assert!(
                (arr[i] - expected).abs() < 1e-6,
                "linear segment mismatch at {}: got {}, expected {}",
                i,
                arr[i],
                expected
            );
        }
    }

    #[test]
    fn test_empty_slice() {
        let mut empty: Vec<f32> = vec![];
        srgb_to_linear_slice(&mut empty);
        assert!(empty.is_empty());

        let empty_u8: Vec<u8> = vec![];
        let mut empty_out: Vec<f32> = vec![];
        srgb_u8_to_linear_slice(&empty_u8, &mut empty_out);
    }

    #[test]
    fn test_non_multiple_of_8() {
        // Test slices that aren't multiples of 8
        for len in [1, 3, 7, 9, 15, 17, 100] {
            let mut values: Vec<f32> = (0..len).map(|i| i as f32 / len as f32).collect();
            let expected: Vec<f32> = values.iter().map(|&v| crate::srgb_to_linear(v)).collect();

            srgb_to_linear_slice(&mut values);

            for (i, (&got, &exp)) in values.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-5,
                    "len={} mismatch at {}: got {}, expected {}",
                    len,
                    i,
                    got,
                    exp
                );
            }
        }
    }
}
