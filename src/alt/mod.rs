//! Alternative/experimental implementations for benchmarking.
//!
//! This module contains various pow approximations and conversion methods
//! for performance comparison. Not part of the stable API.
//!
//! Enable with the `alt` feature flag.

#![allow(missing_docs)]

pub mod accuracy;
pub mod fast_math;
pub mod imageflow;
