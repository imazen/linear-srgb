.section .text.stub_gamma_to_linear_premultiply_rgba_slice,"",@
	.globl	stub_gamma_to_linear_premultiply_rgba_slice
.type	stub_gamma_to_linear_premultiply_rgba_slice,@function
stub_gamma_to_linear_premultiply_rgba_slice:
	.functype	stub_gamma_to_linear_premultiply_rgba_slice (i32, i32, f32) -> ()
	local.get 0
	local.get 1
	local.get 2
	call linear_srgb::simd::gamma_to_linear_premultiply_rgba_slice_tier_wasm128
	end_function
