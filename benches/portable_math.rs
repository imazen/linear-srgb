#[path = "support/portable_math.rs"]
mod math;
use math::Op;
use std::{hint::black_box, time::Duration};
use zenbench::prelude::*;

fn bench(suite: &mut Suite) {
    for op in [Op::Log2, Op::Exp2, Op::Pow] {
        for n in [1, 16, 16384] {
            let input: &'static [f32] = Box::leak(math::samples(op, n).into_boxed_slice());
            suite.compare(format!("{op:?}/{n}"), |g| {
                g.config()
                    .min_rounds(30)
                    .max_rounds(60)
                    .warmup_time(Duration::from_millis(50))
                    .max_time(Duration::from_secs(1));
                g.throughput(Throughput::Elements(n as u64));
                for (name, fun) in [
                    ("native_vector", math::native as fn(&[f32], &mut [f32], Op)),
                    ("relaxed_scalar", math::relaxed_scalar),
                    ("portable_scalar", math::portable),
                    ("portable_fma_context", math::portable_hardware),
                ] {
                    let mut output = vec![0.0; n];
                    g.bench(name, move |b| {
                        b.iter(|| {
                            fun(black_box(input), &mut output, op);
                            black_box(&output);
                        })
                    });
                }
            });
        }
    }
}
zenbench::main!(bench);
