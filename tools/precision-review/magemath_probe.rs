//! Included by magemath_probe.py in an isolated package, never the shipped crate.
use archmage::{SimdToken, attune};
use magemath::{expf as e, logf as l, powf as p};
use magetypes::simd::generic::f32x4;

#[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
compile_error!("the WASM review must enable +simd128");

const PORTABLE: u32 = p::PowfOptions::PORTABLE_GAMMA_MIDP
    .with_execution(p::PowfExecution::Caller)
    .bits();
const BACKEND: u32 = p::PowfOptions::BACKEND_GAMMA_MIDP
    .with_execution(p::PowfExecution::Caller)
    .bits();
const LOG: u32 = l::LogfOptions::PORTABLE_LOWP
    .with_execution(l::LogfExecution::Caller)
    .bits();
const EXP: u32 = e::ExpfOptions::PORTABLE_LOWP
    .with_execution(e::ExpfExecution::Caller)
    .bits();

#[attune(_v3_t, _neon_t, _wasm128_t, _wasm128_relaxed_fma_t, _scalar_t)]
fn probe(token: Token) {
    let mut powers = 0;
    let mut power_diffs = 0;
    let mut policy_diffs = 0;
    let mut logs = 0;
    let mut log_diffs = 0;
    let mut exp_diffs = 0;
    for i in 1..=4096 {
        let xs = core::array::from_fn(|j| (4 * (i - 1) + j + 1) as f32 / 16384.0);
        let input = f32x4::from_array_t(token, xs);
        for n in [0.45, 1.0 / 2.2, 2.2, 2.4, 4.0] {
            let old = legacy::pow_x4(token, input, n).to_array();
            let portable = p::try_powf_f32x4_with::<PORTABLE, _>(token, input, n)
                .unwrap()
                .to_array();
            let backend = p::try_powf_f32x4_with::<BACKEND, _>(token, input, n)
                .unwrap()
                .to_array();
            for j in 0..4 {
                powers += 1;
                power_diffs += usize::from(old[j].to_bits() != portable[j].to_bits());
                policy_diffs += usize::from(backend[j].to_bits() != portable[j].to_bits());
                assert_eq!(
                    portable[j].to_bits(),
                    p::try_powf_with_token::<PORTABLE, _>(token, xs[j], n)
                        .unwrap()
                        .to_bits()
                );
                assert_eq!(
                    backend[j].to_bits(),
                    p::try_powf_with_token::<BACKEND, _>(token, xs[j], n)
                        .unwrap()
                        .to_bits()
                );
            }
        }
        for x in xs {
            logs += 1;
            let log = l::try_log2_with_token::<LOG, _>(token, x).unwrap();
            log_diffs += usize::from(legacy::fast_log2f(x).to_bits() != log.to_bits());
            let arg = x * 16.0 - 8.0;
            let exp = e::try_exp2_with_token::<EXP, _>(token, arg).unwrap();
            exp_diffs += usize::from(legacy::fast_pow2f(arg).to_bits() != exp.to_bits());
        }
    }
    println!(
        "token={} powers={powers} legacy_vs_portable={power_diffs} backend_vs_portable={policy_diffs} log_inputs={logs} legacy_vs_lowp_log={log_diffs} exp_inputs={logs} legacy_vs_lowp_exp={exp_diffs}",
        Token::NAME
    );
}

fn main() {
    #[cfg(target_arch = "x86_64")]
    probe_v3_t(archmage::X64V3Token::summon().expect("native review requires V3"));
    #[cfg(target_arch = "aarch64")]
    probe_neon_t(archmage::NeonToken::summon().expect("ARM review requires NEON"));
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    probe_wasm128_t(archmage::Wasm128Token::summon().expect("WASM review requires SIMD"));
    #[cfg(all(
        target_arch = "wasm32",
        target_feature = "simd128",
        target_feature = "relaxed-simd"
    ))]
    {
        let fma = archmage::Wasm128RelaxedFmaToken::summon();
        let expected = std::env::var("ARCHMAGE_EXPECT_RELAXED_FUSION")
            .expect("declare engine fusion expectation");
        assert_eq!(fma.is_some(), expected == "yes");
        println!("wasm_fma_token={}", fma.is_some());
        if let Some(token) = fma {
            probe_wasm128_relaxed_fma_t(token);
        }
    }
    probe_scalar_t(archmage::ScalarToken);
    for (name, x, n) in [
        ("tiny", f32::from_bits(1), 2.4),
        ("pq_inverse_m1", 0.5, 1.0 / (2610.0 / 16384.0)),
        ("pq_m2", 0.5, 78.84375),
    ] {
        assert!(p::try_powf_with::<{ p::PowfOptions::PORTABLE_GAMMA_MIDP.bits() }>(x, n).is_err());
        let full = p::try_powf_with::<{ p::PowfOptions::PORTABLE_HIGHP.bits() }>(x, n).unwrap();
        println!(
            "domain={name} gamma=rejected full_highp={:08x}",
            full.to_bits()
        );
    }
}
