#[path = "../benches/support/portable_math.rs"]
mod math;
use math::Op;

#[test]
fn fused_scalar_matches_native_lanes_on_normal_domains() {
    use archmage::SimdToken;
    let mut hardware = false;
    #[cfg(target_arch = "x86_64")]
    {
        hardware |= archmage::X64V3Token::summon().is_some();
    }
    #[cfg(target_arch = "aarch64")]
    {
        hardware |= archmage::NeonToken::summon().is_some();
    }
    if std::env::var_os("PORTABLE_MATH_REQUIRE_NATIVE").is_some() {
        assert!(hardware, "caller required native FMA hardware");
    }
    for op in [Op::Log2, Op::Exp2, Op::Pow] {
        let input = math::samples(op, 262144);
        let mut native = vec![0.0; input.len()];
        let mut portable = native.clone();
        math::native(&input, &mut native, op);
        math::portable(&input, &mut portable, op);
        let mut in_context = native.clone();
        math::portable_hardware(&input, &mut in_context, op);
        #[cfg(target_arch = "x86_64")]
        if let Some(token) = archmage::X64V3Token::summon() {
            let mut v3 = native.clone();
            math::native_v3(token, &input, &mut v3, op);
            for (&p, &v) in portable.iter().zip(&v3) {
                assert_eq!(p.to_bits(), v.to_bits(), "{op:?} V3 lane");
            }
        }
        for (i, ((&p, &n), &h)) in portable.iter().zip(&native).zip(&in_context).enumerate() {
            assert_eq!(
                p.to_bits(),
                h.to_bits(),
                "{op:?} context input={}",
                input[i]
            );
            if hardware {
                assert_eq!(p.to_bits(), n.to_bits(), "{op:?} lane input={}", input[i]);
            }
        }
    }
}

#[test]
fn scalar_domain_edges() {
    use linear_srgb::portable as p;
    assert_eq!(p::log2_midp(0.0), f32::NEG_INFINITY);
    assert_eq!(p::log2_midp(f32::INFINITY), f32::INFINITY);
    assert!(p::log2_midp(-1.0).is_nan());
    assert_eq!(p::exp2_midp(-127.0), 0.0);
    assert_eq!(p::exp2_midp(-126.0), f32::MIN_POSITIVE);
    assert_eq!(p::exp2_midp(128.0), f32::INFINITY);
    for n in [-2.4, 0.0, 1.0, 2.4] {
        assert_eq!(p::pow_midp(1.0, n), 1.0);
    }
}
