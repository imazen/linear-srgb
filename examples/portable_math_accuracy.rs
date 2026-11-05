//! Dense-grid cost/accuracy evidence for the proposed portable API.
#[path = "../benches/support/portable_math.rs"]
mod math;
use math::Op;
fn main() {
    for op in [Op::Log2, Op::Exp2, Op::Pow] {
        let input = math::samples(op, 1048576);
        let mut native = vec![0.0; input.len()];
        let mut relaxed = native.clone();
        math::native(&input, &mut native, op);
        math::relaxed_scalar(&input, &mut relaxed, op);
        let (mut native_diff, mut relaxed_diff, mut max_ulp, mut high_exp_ulp) = (0, 0, 0, 0);
        let mut max_relative = 0.0f64;
        for (i, &x) in input.iter().enumerate() {
            let y = math::one(x, op);
            native_diff += usize::from(y.to_bits() != native[i].to_bits());
            relaxed_diff += usize::from(y.to_bits() != relaxed[i].to_bits());
            let expected = match op {
                Op::Log2 => (x as f64).log2(),
                Op::Exp2 => (x as f64).exp2(),
                Op::Pow => (x as f64).powf(2.4f32 as f64),
            };
            let error = y.to_bits().abs_diff((expected as f32).to_bits());
            if matches!(op, Op::Exp2) && x >= 127.5 {
                high_exp_ulp = high_exp_ulp.max(error);
            } else {
                max_ulp = max_ulp.max(error);
            }
            if expected != 0.0 {
                max_relative = max_relative.max(((y as f64 - expected) / expected).abs());
            }
        }
        println!(
            "{op:?}: n={} native_diff={native_diff} relaxed_diff={relaxed_diff} max_ulp={max_ulp} high_exp_ulp={high_exp_ulp} max_relative={max_relative:e}",
            input.len()
        );
    }
}
