//! Standalone size probe. Source tables are emitted by u8_encode_accuracy.
#![allow(unexpected_cfgs)]
#[cfg(encode_size)]
#[allow(dead_code)]
#[path = "../benches/support/u8_options.rs"]
mod options;
#[cfg(encode_size)]
mod implementation {
    include!(env!("ENCODE_SIZE_SOURCE"));
}
fn main() {
    #[cfg(encode_size)]
    {
        let n = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "16384".into())
            .parse::<usize>()
            .unwrap();
        let input: Vec<f32> = (0..n)
            .map(|i| ((i * 7349 + 31) % 65536) as f32 / 65535.0)
            .collect();
        let mut out = vec![0u8; n];
        implementation::encode(std::hint::black_box(&input), &mut out);
        println!(
            "{}",
            std::hint::black_box(out.iter().map(|&x| u64::from(x)).sum::<u64>())
        );
    }
    #[cfg(not(encode_size))]
    println!("Use u8_encode_accuracy --emit-size DIR, then scripts/measure-encode-size.sh DIR");
}
