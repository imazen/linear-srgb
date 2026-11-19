.section .text.stub_linear_to_srgb_extended_slice,"",@
	.globl	stub_linear_to_srgb_extended_slice
.type	stub_linear_to_srgb_extended_slice,@function
stub_linear_to_srgb_extended_slice:
	.functype	stub_linear_to_srgb_extended_slice (i32, i32) -> ()
	.local  	i32, i32, i32, i32, i32, i32, i32, i32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, f32, f32, f64, i64
	global.get __stack_pointer
	i32.const 304
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
	local.set 1
	local.get 0
	local.get 3
	i32.const 6
	i32.shl
	i32.add
	local.set 8
	local.get 2
	i32.const 232
	i32.add
	local.set 5
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
	local.set 3
	local.get 0
	i32.const 64
	i32.add
	local.set 0
	local.get 1
	local.set 9
	br 1
	end_block
	i32.const 0
	local.set 0
	local.get 1
	local.set 3
	i32.const 0
	local.set 9
	local.get 1
	i32.eqz
	br_if 2
	end_block
	local.get 3
	v128.load 48:p2align=2
	local.tee 10
	v128.const -9223372034707292160, -9223372034707292160
	local.tee 11
	v128.and
	local.get 10
	f32x4.abs
	local.tee 12
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	local.tee 13
	f32x4.mul
	local.get 12
	f32x4.sqrt
	local.tee 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	v128.const 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7, 0x1.61d37ap7
	local.tee 15
	f32x4.mul
	v128.const 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11, 0x1.93d6eap11
	local.tee 16
	f32x4.add
	f32x4.mul
	v128.const 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13, 0x1.3aa584p13
	local.tee 17
	f32x4.add
	f32x4.mul
	v128.const 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12
	local.tee 18
	f32x4.add
	f32x4.mul
	v128.const 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10
	local.tee 19
	f32x4.add
	f32x4.mul
	v128.const -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2
	local.tee 20
	f32x4.add
	f32x4.mul
	v128.const -0x1.068508p0, -0x1.068508p0, -0x1.068508p0, -0x1.068508p0
	local.tee 21
	f32x4.add
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	v128.const 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8
	local.tee 22
	f32x4.add
	f32x4.mul
	v128.const 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12
	local.tee 23
	f32x4.add
	f32x4.mul
	v128.const 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13
	local.tee 24
	f32x4.add
	f32x4.mul
	v128.const 0x1.788caep12, 0x1.788caep12, 0x1.788caep12, 0x1.788caep12
	local.tee 25
	f32x4.add
	f32x4.mul
	v128.const 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9
	local.tee 26
	f32x4.add
	f32x4.mul
	v128.const 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4
	local.tee 27
	f32x4.add
	f32x4.div
	local.get 12
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	local.tee 28
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	local.set 29
	local.get 3
	v128.load 32:p2align=2
	local.tee 30
	local.get 11
	v128.and
	local.get 30
	f32x4.abs
	local.tee 31
	local.get 13
	f32x4.mul
	local.get 31
	f32x4.sqrt
	local.tee 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 15
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
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
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
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.mul
	local.get 27
	f32x4.add
	f32x4.div
	local.get 31
	local.get 28
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	local.set 32
	local.get 3
	v128.load 16:p2align=2
	local.tee 33
	local.get 11
	v128.and
	local.get 33
	f32x4.abs
	local.tee 34
	local.get 13
	f32x4.mul
	local.get 34
	f32x4.sqrt
	local.tee 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 15
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
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
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
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.mul
	local.get 27
	f32x4.add
	f32x4.div
	local.get 34
	local.get 28
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	local.set 35
	local.get 3
	v128.load 0:p2align=2
	local.tee 36
	local.get 11
	v128.and
	local.get 36
	f32x4.abs
	local.tee 11
	local.get 13
	f32x4.mul
	local.get 11
	f32x4.sqrt
	local.tee 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 15
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
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
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
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.mul
	local.get 27
	f32x4.add
	f32x4.div
	local.get 11
	local.get 28
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	local.set 13
	block
	local.get 11
	v128.const 0x1p6, 0x1p6, 0x1p6, 0x1p6
	local.tee 14
	f32x4.gt
	local.get 34
	local.get 14
	f32x4.gt
	v128.or
	local.get 31
	local.get 14
	f32x4.gt
	local.get 12
	local.get 14
	f32x4.gt
	v128.or
	v128.or
	i32x4.bitmask
	i32.eqz
	br_if 0
	local.get 2
	local.get 29
	v128.store 192
	local.get 2
	local.get 32
	v128.store 176
	local.get 2
	local.get 35
	v128.store 160
	local.get 2
	local.get 13
	v128.store 144
	local.get 5
	local.get 36
	v128.store 0:p2align=2
	local.get 2
	i64.const 0
	i64.store 296:p2align=2
	local.get 2
	local.get 10
	v128.store 280:p2align=2
	local.get 2
	local.get 30
	v128.store 264:p2align=2
	local.get 2
	local.get 33
	v128.store 248:p2align=2
	i32.const 0
	local.set 1
	loop
	block
	local.get 5
	local.get 1
	i32.add
	f32.load 0
	local.tee 37
	f32.abs
	local.tee 38
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	f32.const 0x1p0
	local.get 37
	f32.copysign
	local.set 37
	block
	block
	local.get 38
	f32.const 0x1.aaaaaap-2
	call powf
	local.tee 38
	f64.promote_f32
	f64.const 0x1.0e152ep0
	f64.mul
	f64.const -0x1.c2a5d6p-5
	f64.add
	local.tee 39
	i64.reinterpret_f64
	local.tee 40
	i64.const 536870911
	i64.and
	i64.const 268435456
	i64.eq
	br_if 0
	local.get 40
	i64.const 32
	i64.shr_u
	i32.wrap_i64
	i32.const 2147483647
	i32.and
	i32.const -940572672
	i32.add
	i32.const -940572671
	i32.ge_u
	br_if 0
	local.get 39
	f32.demote_f64
	local.set 38
	br 1
	end_block
	f32.const 0x1.0e152ep0
	local.get 38
	f32.const -0x1.c2a5d6p-5
	call magetypes::soft_fma::fmaf_exact
	local.set 38
	end_block
	local.get 2
	i32.const 144
	i32.add
	local.get 1
	i32.add
	local.get 37
	local.get 38
	f32.mul
	f32.store 0
	end_block
	local.get 1
	i32.const 4
	i32.add
	local.tee 1
	i32.const 64
	i32.ne
	br_if 0
	end_loop
	local.get 2
	v128.load 144
	local.set 13
	local.get 2
	v128.load 160
	local.set 35
	local.get 2
	v128.load 176
	local.set 32
	local.get 2
	v128.load 192
	local.set 29
	end_block
	local.get 3
	local.get 29
	v128.store 48:p2align=2
	local.get 3
	local.get 32
	v128.store 32:p2align=2
	local.get 3
	local.get 35
	v128.store 16:p2align=2
	local.get 3
	local.get 13
	v128.store 0:p2align=2
	local.get 9
	local.set 1
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
	i32.const 304
	i32.add
	global.set __stack_pointer
	end_function
