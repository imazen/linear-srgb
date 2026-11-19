.section .text.stub_linear_to_gamma_slice,"",@
	.globl	stub_linear_to_gamma_slice
.type	stub_linear_to_gamma_slice,@function
stub_linear_to_gamma_slice:
	.functype	stub_linear_to_gamma_slice (i32, i32, f32) -> ()
	local.get 0
	local.get 1
	local.get 2
	call linear_srgb::target_features::x16::linear_to_gamma_slice_wasm128
	end_function
