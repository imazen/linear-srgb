.section .text.stub_linear_to_srgb_extended_slice,"",@
	.globl	stub_linear_to_srgb_extended_slice
.type	stub_linear_to_srgb_extended_slice,@function
stub_linear_to_srgb_extended_slice:
	.functype	stub_linear_to_srgb_extended_slice (i32, i32) -> ()
	.local  	i32, i32, i32, v128, v128, v128, i32, v128, v128, i32, v128, v128, i32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128
	global.get __stack_pointer
	i32.const 128
	i32.sub
	local.tee 2
	global.set __stack_pointer
	local.get 1
	i32.const 15
	i32.and
	local.set 3
	local.get 0
	local.get 1
	i32.const 536870896
	i32.and
	i32.const 2
	i32.shl
	i32.add
	local.set 4
	block
	block
	block
	block
	local.get 1
	i32.const 4
	i32.shr_u
	local.tee 1
	i32.eqz
	br_if 0
	loop
	local.get 0
	v128.load 0:p2align=2
	local.tee 5
	f32x4.abs
	local.tee 6
	v128.const 0x1p6, 0x1p6, 0x1p6, 0x1p6
	local.tee 7
	f32x4.gt
	local.get 0
	i32.const 16
	i32.add
	local.tee 8
	v128.load 0:p2align=2
	local.tee 9
	f32x4.abs
	local.tee 10
	local.get 7
	f32x4.gt
	v128.or
	local.get 0
	i32.const 32
	i32.add
	local.tee 11
	v128.load 0:p2align=2
	local.tee 12
	f32x4.abs
	local.tee 13
	local.get 7
	f32x4.gt
	local.get 0
	i32.const 48
	i32.add
	local.tee 14
	v128.load 0:p2align=2
	local.tee 15
	f32x4.abs
	local.tee 16
	local.get 7
	f32x4.gt
	v128.or
	v128.or
	i32x4.bitmask
	br_if 2
	local.get 14
	local.get 15
	v128.const -9223372034707292160, -9223372034707292160
	local.tee 17
	v128.and
	local.get 16
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	local.tee 15
	f32x4.mul
	local.get 16
	f32x4.sqrt
	local.tee 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	v128.const 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7
	local.tee 18
	f32x4.mul
	v128.const 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11
	local.tee 19
	f32x4.add
	f32x4.mul
	v128.const 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13
	local.tee 20
	f32x4.add
	f32x4.mul
	v128.const 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12
	local.tee 21
	f32x4.add
	f32x4.mul
	v128.const 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10
	local.tee 22
	f32x4.add
	f32x4.mul
	v128.const -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2
	local.tee 23
	f32x4.add
	f32x4.mul
	v128.const -0x1.068508p0, -0x1.068508p0, -0x1.068508p0, -0x1.068508p0
	local.tee 24
	f32x4.add
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	v128.const 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8
	local.tee 25
	f32x4.add
	f32x4.mul
	v128.const 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12
	local.tee 26
	f32x4.add
	f32x4.mul
	v128.const 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13
	local.tee 27
	f32x4.add
	f32x4.mul
	v128.const 0x1.788caep12, 0x1.788caep12, 0x1.788caep12, 0x1.788caep12
	local.tee 28
	f32x4.add
	f32x4.mul
	v128.const 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9
	local.tee 29
	f32x4.add
	f32x4.mul
	v128.const 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4
	local.tee 30
	f32x4.add
	f32x4.div
	local.get 16
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	local.tee 31
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	v128.store 0:p2align=2
	local.get 11
	local.get 12
	local.get 17
	v128.and
	local.get 13
	local.get 15
	f32x4.mul
	local.get 13
	f32x4.sqrt
	local.tee 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 18
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 24
	f32x4.add
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 25
	f32x4.add
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.mul
	local.get 27
	f32x4.add
	f32x4.mul
	local.get 28
	f32x4.add
	f32x4.mul
	local.get 29
	f32x4.add
	f32x4.mul
	local.get 30
	f32x4.add
	f32x4.div
	local.get 13
	local.get 31
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	v128.store 0:p2align=2
	local.get 8
	local.get 9
	local.get 17
	v128.and
	local.get 10
	local.get 15
	f32x4.mul
	local.get 10
	f32x4.sqrt
	local.tee 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 18
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 24
	f32x4.add
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 25
	f32x4.add
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.mul
	local.get 27
	f32x4.add
	f32x4.mul
	local.get 28
	f32x4.add
	f32x4.mul
	local.get 29
	f32x4.add
	f32x4.mul
	local.get 30
	f32x4.add
	f32x4.div
	local.get 10
	local.get 31
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	v128.store 0:p2align=2
	local.get 0
	local.get 5
	local.get 17
	v128.and
	local.get 6
	local.get 15
	f32x4.mul
	local.get 6
	f32x4.sqrt
	local.tee 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 18
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 24
	f32x4.add
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 25
	f32x4.add
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.mul
	local.get 27
	f32x4.add
	f32x4.mul
	local.get 28
	f32x4.add
	f32x4.mul
	local.get 29
	f32x4.add
	f32x4.mul
	local.get 30
	f32x4.add
	f32x4.div
	local.get 6
	local.get 31
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	v128.store 0:p2align=2
	local.get 0
	i32.const 64
	i32.add
	local.set 0
	local.get 1
	i32.const -1
	i32.add
	local.tee 1
	br_if 0
	end_loop
	end_block
	local.get 3
	i32.eqz
	br_if 2
	block
	i32.const 64
	local.get 3
	i32.const 2
	i32.shl
	local.tee 0
	i32.sub
	local.tee 1
	i32.eqz
	br_if 0
	local.get 2
	local.get 0
	i32.add
	i32.const 0
	local.get 1
	memory.fill 0
	end_block
	local.get 0
	i32.eqz
	local.tee 1
	br_if 1
	local.get 2
	local.get 4
	local.get 0
	memory.copy 0, 0
	br 1
	end_block
	local.get 0
	local.get 1
	local.get 4
	local.get 3
	i32.const linear_srgb::kernels::float::x16::linear_to_srgb_extended_slice::slow::<archmage::tokens::generated::wasm::Wasm128Token>
	call linear_srgb::kernels::finish_slice_cold::<archmage::tokens::generated::wasm::Wasm128Token, 16>
	br 1
	end_block
	block
	local.get 2
	v128.load 0
	local.tee 5
	f32x4.abs
	local.tee 6
	v128.const 0x1p6, 0x1p6, 0x1p6, 0x1p6
	local.tee 7
	f32x4.gt
	local.get 2
	v128.load 16
	local.tee 9
	f32x4.abs
	local.tee 10
	local.get 7
	f32x4.gt
	v128.or
	local.get 2
	v128.load 32
	local.tee 12
	f32x4.abs
	local.tee 13
	local.get 7
	f32x4.gt
	local.get 2
	v128.load 48
	local.tee 15
	f32x4.abs
	local.tee 16
	local.get 7
	f32x4.gt
	v128.or
	v128.or
	i32x4.bitmask
	br_if 0
	local.get 2
	local.get 15
	v128.const -9223372034707292160, -9223372034707292160
	local.tee 17
	v128.and
	local.get 16
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	local.tee 15
	f32x4.mul
	local.get 16
	f32x4.sqrt
	local.tee 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	v128.const 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7
	local.tee 18
	f32x4.mul
	v128.const 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11
	local.tee 19
	f32x4.add
	f32x4.mul
	v128.const 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13
	local.tee 20
	f32x4.add
	f32x4.mul
	v128.const 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12
	local.tee 21
	f32x4.add
	f32x4.mul
	v128.const 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10
	local.tee 22
	f32x4.add
	f32x4.mul
	v128.const -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2
	local.tee 23
	f32x4.add
	f32x4.mul
	v128.const -0x1.068508p0, -0x1.068508p0, -0x1.068508p0, -0x1.068508p0
	local.tee 24
	f32x4.add
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	v128.const 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8
	local.tee 25
	f32x4.add
	f32x4.mul
	v128.const 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12
	local.tee 26
	f32x4.add
	f32x4.mul
	v128.const 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13
	local.tee 27
	f32x4.add
	f32x4.mul
	v128.const 0x1.788caep12, 0x1.788caep12, 0x1.788caep12, 0x1.788caep12
	local.tee 28
	f32x4.add
	f32x4.mul
	v128.const 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9
	local.tee 29
	f32x4.add
	f32x4.mul
	v128.const 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4
	local.tee 30
	f32x4.add
	f32x4.div
	local.get 16
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	local.tee 31
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	v128.store 112
	local.get 2
	local.get 12
	local.get 17
	v128.and
	local.get 13
	local.get 15
	f32x4.mul
	local.get 13
	f32x4.sqrt
	local.tee 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 18
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 24
	f32x4.add
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 25
	f32x4.add
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.mul
	local.get 27
	f32x4.add
	f32x4.mul
	local.get 28
	f32x4.add
	f32x4.mul
	local.get 29
	f32x4.add
	f32x4.mul
	local.get 30
	f32x4.add
	f32x4.div
	local.get 13
	local.get 31
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	v128.store 96
	local.get 2
	local.get 9
	local.get 17
	v128.and
	local.get 10
	local.get 15
	f32x4.mul
	local.get 10
	f32x4.sqrt
	local.tee 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 18
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 24
	f32x4.add
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 25
	f32x4.add
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.mul
	local.get 27
	f32x4.add
	f32x4.mul
	local.get 28
	f32x4.add
	f32x4.mul
	local.get 29
	f32x4.add
	f32x4.mul
	local.get 30
	f32x4.add
	f32x4.div
	local.get 10
	local.get 31
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	v128.store 80
	local.get 2
	local.get 5
	local.get 17
	v128.and
	local.get 6
	local.get 15
	f32x4.mul
	local.get 6
	f32x4.sqrt
	local.tee 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 18
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 24
	f32x4.add
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	local.get 25
	f32x4.add
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.mul
	local.get 27
	f32x4.add
	f32x4.mul
	local.get 28
	f32x4.add
	f32x4.mul
	local.get 29
	f32x4.add
	f32x4.mul
	local.get 30
	f32x4.add
	f32x4.div
	local.get 6
	local.get 31
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	v128.store 64
	local.get 1
	br_if 1
	local.get 4
	local.get 2
	i32.const 64
	i32.add
	local.get 0
	memory.copy 0, 0
	br 1
	end_block
	i32.const 4
	i32.const 0
	local.get 4
	local.get 3
	i32.const linear_srgb::kernels::float::x16::linear_to_srgb_extended_slice::slow::<archmage::tokens::generated::wasm::Wasm128Token>
	call linear_srgb::kernels::finish_slice_cold::<archmage::tokens::generated::wasm::Wasm128Token, 16>
	end_block
	local.get 2
	i32.const 128
	i32.add
	global.set __stack_pointer
	end_function
