.section .text.stub_linear_to_srgb_extended_slice,"",@
	.globl	stub_linear_to_srgb_extended_slice
.type	stub_linear_to_srgb_extended_slice,@function
stub_linear_to_srgb_extended_slice:
	.functype	stub_linear_to_srgb_extended_slice (i32, i32) -> ()
	.local  	i32, i32, i32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, i32, v128, f32, f32
	block 
	local.get 1
	i32.const 2
	i32.shl 
	local.tee 2
	i32.const 2147483584
	i32.and 
	local.tee 3
	i32.eqz
	br_if 0
	local.get 0
	local.get 3
	i32.add 
	local.set 4
	local.get 0
	local.set 3
	loop 
	local.get 3
	local.get 3
	v128.load 0:p2align=2
	local.tee 5
	f32x4.abs
	local.tee 6
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	local.tee 7
	f32x4.mul
	local.get 6
	f32x4.sqrt
	local.tee 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	v128.const 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7
	local.tee 9
	f32x4.mul
	v128.const 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11
	local.tee 10
	f32x4.add
	f32x4.mul
	v128.const 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13
	local.tee 11
	f32x4.add
	f32x4.mul
	v128.const 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12
	local.tee 12
	f32x4.add
	f32x4.mul
	v128.const 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10
	local.tee 13
	f32x4.add
	f32x4.mul
	v128.const -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2
	local.tee 14
	f32x4.add
	f32x4.mul
	v128.const -0x1.068508p0, -0x1.068508p0, -0x1.068508p0, -0x1.068508p0
	local.tee 15
	f32x4.add
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	v128.const 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8
	local.tee 16
	f32x4.add
	f32x4.mul
	v128.const 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12
	local.tee 17
	f32x4.add
	f32x4.mul
	v128.const 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13
	local.tee 18
	f32x4.add
	f32x4.mul
	v128.const 0x1.788caep12, 0x1.788caep12, 0x1.788caep12, 0x1.788caep12
	local.tee 19
	f32x4.add
	f32x4.mul
	v128.const 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9
	local.tee 20
	f32x4.add
	f32x4.mul
	v128.const 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4
	local.tee 21
	f32x4.add
	f32x4.div
	local.get 6
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	local.tee 22
	f32x4.lt
	v128.bitselect
	local.tee 8
	f32x4.neg
	local.get 8
	local.get 5
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 6
	f32x4.lt
	v128.bitselect
	v128.store 0:p2align=2
	local.get 3
	i32.const 48
	i32.add 
	local.tee 23
	local.get 23
	v128.load 0:p2align=2
	local.tee 24
	f32x4.abs
	local.tee 5
	local.get 7
	f32x4.mul
	local.get 5
	f32x4.sqrt
	local.tee 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 9
	f32x4.mul
	local.get 10
	f32x4.add
	f32x4.mul
	local.get 11
	f32x4.add
	f32x4.mul
	local.get 12
	f32x4.add
	f32x4.mul
	local.get 13
	f32x4.add
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
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
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.div
	local.get 5
	local.get 22
	f32x4.lt
	v128.bitselect
	local.tee 8
	f32x4.neg
	local.get 8
	local.get 24
	local.get 6
	f32x4.lt
	v128.bitselect
	v128.store 0:p2align=2
	local.get 3
	i32.const 32
	i32.add 
	local.tee 23
	local.get 23
	v128.load 0:p2align=2
	local.tee 24
	f32x4.abs
	local.tee 5
	local.get 7
	f32x4.mul
	local.get 5
	f32x4.sqrt
	local.tee 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 9
	f32x4.mul
	local.get 10
	f32x4.add
	f32x4.mul
	local.get 11
	f32x4.add
	f32x4.mul
	local.get 12
	f32x4.add
	f32x4.mul
	local.get 13
	f32x4.add
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
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
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.div
	local.get 5
	local.get 22
	f32x4.lt
	v128.bitselect
	local.tee 8
	f32x4.neg
	local.get 8
	local.get 24
	local.get 6
	f32x4.lt
	v128.bitselect
	v128.store 0:p2align=2
	local.get 3
	i32.const 16
	i32.add 
	local.tee 23
	local.get 23
	v128.load 0:p2align=2
	local.tee 24
	f32x4.abs
	local.tee 5
	local.get 7
	f32x4.mul
	local.get 5
	f32x4.sqrt
	local.tee 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 9
	f32x4.mul
	local.get 10
	f32x4.add
	f32x4.mul
	local.get 11
	f32x4.add
	f32x4.mul
	local.get 12
	f32x4.add
	f32x4.mul
	local.get 13
	f32x4.add
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
	local.get 8
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
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.div
	local.get 5
	local.get 22
	f32x4.lt
	v128.bitselect
	local.tee 8
	f32x4.neg
	local.get 8
	local.get 24
	local.get 6
	f32x4.lt
	v128.bitselect
	v128.store 0:p2align=2
	local.get 3
	i32.const 64
	i32.add 
	local.tee 3
	local.get 4
	i32.ne 
	br_if 0
	end_loop
	end_block
	block 
	local.get 2
	i32.const 60
	i32.and 
	local.tee 23
	i32.eqz
	br_if 0
	local.get 0
	local.get 1
	i32.const 536870896
	i32.and 
	i32.const 2
	i32.shl 
	i32.add 
	local.set 3
	loop 
	block 
	block 
	local.get 3
	f32.load 0
	local.tee 25
	f32.abs 
	local.tee 26
	f32.const 0x1.8ea082p-9
	f32.lt 
	br_if 0
	f32.const nan
	f32.const 0x1p0
	local.get 25
	f32.copysign
	local.get 25
	local.get 25
	f32.ne 
	f32.select
	local.get 26
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul 
	f32.const -0x1.c2a5d6p-5
	f32.add 
	f32.mul 
	local.set 25
	br 1
	end_block
	local.get 25
	f32.const 0x1.9d70a4p3
	f32.mul 
	local.set 25
	end_block
	local.get 3
	local.get 25
	f32.store 0
	local.get 3
	i32.const 4
	i32.add 
	local.set 3
	local.get 23
	i32.const -4
	i32.add 
	local.tee 23
	br_if 0
	end_loop
	end_block
	end_function
