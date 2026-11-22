.section .text.stub_srgb_to_linear_extended_slice,"",@
	.globl	stub_srgb_to_linear_extended_slice
.type	stub_srgb_to_linear_extended_slice,@function
stub_srgb_to_linear_extended_slice:
	.functype	stub_srgb_to_linear_extended_slice (i32, i32) -> ()
	.local  	i32, i32, v128, v128
	local.get 1
	i32.const 3
	i32.and
	local.set 2
	local.get 0
	local.get 1
	i32.const 536870908
	i32.and
	i32.const 2
	i32.shl
	i32.add
	local.set 3
	block
	block
	block
	local.get 1
	i32.const 2
	i32.shr_u
	local.tee 1
	i32.eqz
	br_if 0
	loop
	local.get 0
	v128.load 0:p2align=2
	local.tee 4
	f32x4.abs
	local.tee 5
	v128.const 0x1p3, 0x1p3, 0x1p3, 0x1p3
	f32x4.gt
	i32x4.bitmask
	br_if 2
	local.get 0
	local.get 4
	v128.const -9223372034707292160, -9223372034707292160
	v128.and
	local.get 5
	v128.const 0x1.3d0722p-4, 0x1.3d0722p-4, 0x1.3d0722p-4, 0x1.3d0722p-4
	f32x4.mul
	local.get 5
	local.get 5
	local.get 5
	local.get 5
	local.get 5
	local.get 5
	v128.const 0x1.0aa4c6p14, 0x1.0aa4c6p14, 0x1.0aa4c6p14, 0x1.0aa4c6p14
	f32x4.mul
	v128.const 0x1.056176p17, 0x1.056176p17, 0x1.056176p17, 0x1.056176p17
	f32x4.add
	f32x4.mul
	v128.const 0x1.adac9p17, 0x1.adac9p17, 0x1.adac9p17, 0x1.adac9p17
	f32x4.add
	f32x4.mul
	v128.const 0x1.8eafd2p16, 0x1.8eafd2p16, 0x1.8eafd2p16, 0x1.8eafd2p16
	f32x4.add
	f32x4.mul
	v128.const 0x1.ead02cp13, 0x1.ead02cp13, 0x1.ead02cp13, 0x1.ead02cp13
	f32x4.add
	f32x4.mul
	v128.const 0x1.c78544p9, 0x1.c78544p9, 0x1.c78544p9, 0x1.c78544p9
	f32x4.add
	f32x4.mul
	v128.const 0x1.205782p4, 0x1.205782p4, 0x1.205782p4, 0x1.205782p4
	f32x4.add
	local.get 5
	local.get 5
	local.get 5
	local.get 5
	local.get 5
	local.get 5
	v128.const -0x1.fcb852p5, -0x1.fcb852p5, -0x1.fcb852p5, -0x1.fcb852p5
	f32x4.add
	f32x4.mul
	v128.const 0x1.1793fcp12, 0x1.1793fcp12, 0x1.1793fcp12, 0x1.1793fcp12
	f32x4.add
	f32x4.mul
	v128.const 0x1.41da1cp16, 0x1.41da1cp16, 0x1.41da1cp16, 0x1.41da1cp16
	f32x4.add
	f32x4.mul
	v128.const 0x1.c1dcf4p17, 0x1.c1dcf4p17, 0x1.c1dcf4p17, 0x1.c1dcf4p17
	f32x4.add
	f32x4.mul
	v128.const 0x1.26a3c2p17, 0x1.26a3c2p17, 0x1.26a3c2p17, 0x1.26a3c2p17
	f32x4.add
	f32x4.mul
	v128.const 0x1.516812p14, 0x1.516812p14, 0x1.516812p14, 0x1.516812p14
	f32x4.add
	f32x4.div
	local.get 5
	v128.const 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	v128.store 0:p2align=2
	local.get 0
	i32.const 16
	i32.add
	local.set 0
	local.get 1
	i32.const -1
	i32.add
	local.tee 1
	br_if 0
	end_loop
	end_block
	local.get 2
	i32.eqz
	br_if 1
	i32.const 4
	i32.const 0
	local.get 3
	local.get 2
	i32.const linear_srgb::kernels::float::x4::srgb_to_linear_extended_slice_cfg::slow::<archmage::tokens::generated::wasm::Wasm128Token>
	call linear_srgb::kernels::finish_slice_cold::<archmage::tokens::generated::wasm::Wasm128Token, 4>
	return
	end_block
	local.get 0
	local.get 1
	local.get 3
	local.get 2
	i32.const linear_srgb::kernels::float::x4::srgb_to_linear_extended_slice_cfg::slow::<archmage::tokens::generated::wasm::Wasm128Token>
	call linear_srgb::kernels::finish_slice_cold::<archmage::tokens::generated::wasm::Wasm128Token, 4>
	end_block
	end_function
