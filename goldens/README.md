# Numerical goldens from main

These fixtures freeze **main before PR #30 merges**, at numerical source
[`c5b83e3e`](https://github.com/imazen/linear-srgb/commit/c5b83e3e950e7e5955eb64174eddecc7a90d4c08).
They must not be regenerated from the PR to accept its output. This standalone,
unpublished package calls the real library through public APIs and has its own
committed dependency lock. It is excluded from the shipped crate.

The baseline uses rustc **1.99.1** (`58cf8f9c3`, 2026-10-13), archmage and
magetypes **0.9.31**, and libm **0.2.16**. WASM runs use Wasmtime **49.0.2**
and `+simd128`, with global relaxed-SIMD disabled. A preliminary capture under
the older local beta.2 dependency lock differed in extended-range edge/tail
hashes on unchanged main. The checked-in fixtures use the committed release
resolution, not that preliminary capture.

The [pre-merge PR #30 comparison](PR30.md) records the changed chunks and
measured white-endpoint outputs. Main's frozen expectations remain intact.

## Stored data and coverage

Large output arrays are never stored. Each row records a case name, output-word
count and FNV-1a/64 hashes of successive **4,096-word chunks**. Hashes consume
the explicit little-endian bytes of u32 words, so they are independent of host
endianness and Rust's hash implementation. F32 values use their exact output
bits. Finite values and signed zero remain distinct; NaNs are canonicalized to
`0x7fc00000`, matching the existing cross-tier NaN contract.

The exception is `u8_decode.txt`: it stores all **256 inputs' output bits**,
indexed by input byte. No fixture file exceeds 30 KB.

The main input grid contains every `i / 65535.0` for `i = 0..65535`, followed
by 256 deterministic edge inputs. Edges include signed zero, subnormals,
infinities, NaNs, extended values, neighbors of transfer/alpha junctions and
integer-generated bit patterns. Tail rows independently run lengths 0..65
(pixels for RGBA), rotating the edge inputs; each length is prefixed into its
output stream. A failing hash reports the operation and exact word range.

Fixtures cover default scalar rational conversions, scalar transfer functions,
LUT interpolation, dispatched sRGB/extended/gamma/BT.709/PQ/HLG slices,
RGBA preservation and premultiplication, u8/u16 encode/decode, and direct
token arrays. Five gamma exponents are separate cases. The direct calls cover
x4/x8 V3, x4 NEON/SIMD128, and an explicitly selected x16 V4 mode.

These are drift detectors for the captured behavior, not an assertion that
every output is mathematically correct. Platform `precise`/IEC `powf` APIs,
experimental alternatives, no-std builds and relaxed-SIMD engines are outside
this fixture set. Existing accuracy, parity, monotonicity and row tests remain
necessary. Native profiles were captured on Linux; other OS math backends are
not silently assigned a Linux baseline.

## Running and reviewing changes

```sh
rustup toolchain install 1.99.1 --profile minimal --component clippy,rustfmt
rustup target add --toolchain 1.99.1 wasm32-wasip1
just numerical-goldens-check
just numerical-goldens-wasm
just numerical-goldens-wasm precise-portable
just numerical-goldens-v4 # requires AVX-512; unavailable hardware is an error
```

CI independently checks Linux x86-64, Linux ARM64, strict WASM default and
strict WASM precise-portable. Native checks also run without AVX-512 enabled.
The explicit V4 mode was checked using a prebuilt test binary on AVX-512
hardware; GitHub's ordinary runners are not required to provide AVX-512.
Mutation tests verify detection of a one-bit change in the final partial
chunk, signed-zero changes, and missing/extra/truncated cases. Missing fixture
files fail compilation through `include_str!`; no test blesses or skips them.

For an authorized future baseline change, capture candidates into a **new**
directory and review the changed cases before copying anything into fixtures:

```sh
just numerical-goldens-capture "$HOME/tmp/new-golden-candidates"
```

The capture tool refuses an existing output directory. Dependency updates use
the standalone lock: `cargo update --manifest-path goldens/Cargo.toml`.
Updating a compiler, dependency or expected output requires reviewing the
resulting drift; normal checks never write fixture files.
