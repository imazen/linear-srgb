.section .text.stub_gamma_to_linear_slice,"",@
	.globl	stub_gamma_to_linear_slice
.type	stub_gamma_to_linear_slice,@function
stub_gamma_to_linear_slice:
	.functype	stub_gamma_to_linear_slice (i32, i32, f32) -> ()
	local.get 0
	local.get 1
	local.get 2
	call linear_srgb::target_features::x16::gamma_to_linear_slice_wasm128
	end_function
