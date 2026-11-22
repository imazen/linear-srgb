.section .text.stub_linear_to_srgb_extended_slice,"",@
	.globl	stub_linear_to_srgb_extended_slice
.type	stub_linear_to_srgb_extended_slice,@function
stub_linear_to_srgb_extended_slice:
	.functype	stub_linear_to_srgb_extended_slice (i32, i32) -> ()
	.local  	i32, i32, i32, i32, v128, v128
	global.get __stack_pointer
	i32.const 16
	i32.sub
	local.tee 2
	global.set __stack_pointer
	local.get 1
	i32.const 3
	i32.and
	local.set 3
	block
	local.get 1
	i32.const 2
	i32.shl
	i32.const 2147483632
	i32.and
	local.tee 4
	i32.eqz
	br_if 0
	local.get 0
	local.set 5
	loop
	local.get 5
	v128.load 0:p2align=2
	local.tee 6
	v128.const -9223372034707292160, -9223372034707292160
	v128.and
	local.get 6
	f32x4.abs
	local.tee 7
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	f32x4.mul
	local.get 7
	f32x4.sqrt
	local.tee 6
	local.get 6
	local.get 6
	local.get 6
	local.get 6
	local.get 6
	v128.const 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7
	f32x4.mul
	v128.const 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11
	f32x4.add
	f32x4.mul
	v128.const 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13
	f32x4.add
	f32x4.mul
	v128.const 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12
	f32x4.add
	f32x4.mul
	v128.const 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10
	f32x4.add
	f32x4.mul
	v128.const -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2
	f32x4.add
	f32x4.mul
	v128.const -0x1.068508p0, -0x1.068508p0, -0x1.068508p0, -0x1.068508p0
	f32x4.add
	local.get 6
	local.get 6
	local.get 6
	local.get 6
	local.get 6
	local.get 6
	v128.const 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8
	f32x4.add
	f32x4.mul
	v128.const 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12
	f32x4.add
	f32x4.mul
	v128.const 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13
	f32x4.add
	f32x4.mul
	v128.const 0x1.788caep12, 0x1.788caep12, 0x1.788caep12, 0x1.788caep12
	f32x4.add
	f32x4.mul
	v128.const 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9
	f32x4.add
	f32x4.mul
	v128.const 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4
	f32x4.add
	f32x4.div
	local.get 7
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	local.set 6
	block
	block
	local.get 7
	v128.const 0x1p6, 0x1p6, 0x1p6, 0x1p6
	f32x4.gt
	i32x4.bitmask
	br_if 0
	local.get 5
	local.get 6
	v128.store 0:p2align=2
	br 1
	end_block
	local.get 2
	local.get 6
	v128.store 0
	local.get 5
	local.get 2
	call linear_srgb::kernels::float::linear_to_srgb_extended_patch::<archmage::tokens::generated::wasm::Wasm128Token, 4>
	local.get 5
	local.get 2
	v128.load 0
	v128.store 0:p2align=2
	end_block
	local.get 5
	i32.const 16
	i32.add
	local.set 5
	local.get 4
	i32.const -16
	i32.add
	local.tee 4
	br_if 0
	end_loop
	end_block
	block
	local.get 3
	i32.eqz
	br_if 0
	local.get 0
	local.get 1
	i32.const 536870908
	i32.and
	i32.const 2
	i32.shl
	i32.add
	local.get 3
	i32.const linear_srgb::kernels::float::x4::linear_to_srgb_extended_slice_cfg::tail::<archmage::tokens::generated::wasm::Wasm128Token>
	call linear_srgb::kernels::finish_tail_cold::<archmage::tokens::generated::wasm::Wasm128Token, 4>
	end_block
	local.get 2
	i32.const 16
	i32.add
	global.set __stack_pointer
	end_function
