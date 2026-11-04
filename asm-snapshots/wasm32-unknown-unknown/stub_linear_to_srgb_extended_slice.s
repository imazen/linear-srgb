.section .text.stub_linear_to_srgb_extended_slice,"",@
	.globl	stub_linear_to_srgb_extended_slice
.type	stub_linear_to_srgb_extended_slice,@function
stub_linear_to_srgb_extended_slice:
	.functype	stub_linear_to_srgb_extended_slice (i32, i32) -> ()
	.local  	i32, i32, i32, i32, i32, i32, i32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128
	global.get __stack_pointer
	i32.const 144
	i32.sub 
	local.tee 2
	global.set __stack_pointer
	local.get 1
	i32.const 4
	i32.shr_u
	local.set 3
	local.get 0
	local.get 1
	i32.const 536870896
	i32.and 
	i32.const 2
	i32.shl 
	i32.add 
	local.set 4
	i32.const 0
	local.set 5
	block 
	local.get 1
	i32.const 15
	i32.and 
	local.tee 6
	i32.eqz
	br_if 0
	block 
	i32.const 64
	local.get 6
	i32.const 2
	i32.shl 
	local.tee 1
	i32.sub 
	local.tee 5
	i32.eqz
	br_if 0
	local.get 2
	i32.const 80
	i32.add 
	local.get 1
	i32.add 
	i32.const 0
	local.get 5
	memory.fill 0
	end_block
	block 
	local.get 1
	i32.eqz
	br_if 0
	local.get 2
	i32.const 80
	i32.add 
	local.get 4
	local.get 1
	memory.copy 0, 0
	end_block
	local.get 2
	local.get 2
	v128.load 128:p2align=2
	v128.store 64:p2align=2
	local.get 2
	local.get 2
	v128.load 112:p2align=2
	v128.store 48:p2align=2
	local.get 2
	local.get 2
	v128.load 96:p2align=2
	v128.store 32:p2align=2
	local.get 2
	local.get 2
	v128.load 80:p2align=2
	v128.store 16:p2align=2
	i32.const 1
	local.set 5
	end_block
	local.get 2
	local.get 5
	i32.store 12
	local.get 2
	i32.const 12
	i32.add 
	i32.const 4
	i32.add 
	local.tee 7
	i32.const 0
	local.get 6
	i32.select
	local.set 5
	local.get 0
	local.get 3
	i32.const 6
	i32.shl 
	i32.add 
	local.set 8
	block 
	loop 
	block 
	block 
	local.get 0
	i32.eqz
	br_if 0
	local.get 0
	local.get 8
	i32.eq 
	br_if 0
	local.get 0
	local.set 1
	local.get 0
	i32.const 64
	i32.add 
	local.set 0
	local.get 5
	local.set 3
	br 1
	end_block
	i32.const 0
	local.set 0
	local.get 5
	local.set 1
	i32.const 0
	local.set 3
	local.get 5
	i32.eqz
	br_if 2
	end_block
	local.get 1
	local.get 1
	v128.load 48:p2align=2
	local.tee 9
	f32x4.abs
	local.tee 10
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	local.tee 11
	f32x4.mul
	local.get 10
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	v128.const 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7
	local.tee 13
	f32x4.mul
	v128.const 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11
	local.tee 14
	f32x4.add
	f32x4.mul
	v128.const 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13
	local.tee 15
	f32x4.add
	f32x4.mul
	v128.const 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12
	local.tee 16
	f32x4.add
	f32x4.mul
	v128.const 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10
	local.tee 17
	f32x4.add
	f32x4.mul
	v128.const -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2
	local.tee 18
	f32x4.add
	f32x4.mul
	v128.const -0x1.068508p0, -0x1.068508p0, -0x1.068508p0, -0x1.068508p0
	local.tee 19
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	v128.const 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8
	local.tee 20
	f32x4.add
	f32x4.mul
	v128.const 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12
	local.tee 21
	f32x4.add
	f32x4.mul
	v128.const 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13
	local.tee 22
	f32x4.add
	f32x4.mul
	v128.const 0x1.788caep12, 0x1.788caep12, 0x1.788caep12, 0x1.788caep12
	local.tee 23
	f32x4.add
	f32x4.mul
	v128.const 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9
	local.tee 24
	f32x4.add
	f32x4.mul
	v128.const 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4
	local.tee 25
	f32x4.add
	f32x4.div
	local.get 10
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	local.tee 26
	f32x4.lt
	v128.bitselect
	local.tee 12
	f32x4.neg
	local.get 12
	local.get 9
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 10
	f32x4.lt
	v128.bitselect
	v128.store 48:p2align=2
	local.get 1
	local.get 1
	v128.load 32:p2align=2
	local.tee 27
	f32x4.abs
	local.tee 9
	local.get 11
	f32x4.mul
	local.get 9
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 13
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	f32x4.mul
	local.get 18
	f32x4.add
	f32x4.mul
	local.get 19
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
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
	f32x4.mul
	local.get 25
	f32x4.add
	f32x4.div
	local.get 9
	local.get 26
	f32x4.lt
	v128.bitselect
	local.tee 12
	f32x4.neg
	local.get 12
	local.get 27
	local.get 10
	f32x4.lt
	v128.bitselect
	v128.store 32:p2align=2
	local.get 1
	local.get 1
	v128.load 16:p2align=2
	local.tee 27
	f32x4.abs
	local.tee 9
	local.get 11
	f32x4.mul
	local.get 9
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 13
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	f32x4.mul
	local.get 18
	f32x4.add
	f32x4.mul
	local.get 19
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
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
	f32x4.mul
	local.get 25
	f32x4.add
	f32x4.div
	local.get 9
	local.get 26
	f32x4.lt
	v128.bitselect
	local.tee 12
	f32x4.neg
	local.get 12
	local.get 27
	local.get 10
	f32x4.lt
	v128.bitselect
	v128.store 16:p2align=2
	local.get 1
	local.get 1
	v128.load 0:p2align=2
	local.tee 27
	f32x4.abs
	local.tee 9
	local.get 11
	f32x4.mul
	local.get 9
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 13
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	f32x4.mul
	local.get 18
	f32x4.add
	f32x4.mul
	local.get 19
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 12
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
	f32x4.mul
	local.get 25
	f32x4.add
	f32x4.div
	local.get 9
	local.get 26
	f32x4.lt
	v128.bitselect
	local.tee 12
	f32x4.neg
	local.get 12
	local.get 27
	local.get 10
	f32x4.lt
	v128.bitselect
	v128.store 0:p2align=2
	local.get 3
	local.set 5
	br 0
	end_loop
	end_block
	block 
	local.get 2
	i32.load 12
	i32.const 1
	i32.ne 
	br_if 0
	local.get 6
	i32.const 2
	i32.shl 
	local.tee 1
	i32.eqz
	br_if 0
	local.get 4
	local.get 7
	local.get 1
	memory.copy 0, 0
	end_block
	local.get 2
	i32.const 144
	i32.add 
	global.set __stack_pointer
	end_function
