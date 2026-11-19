.section .text.stub_linear_to_srgb_extended_slice,"",@
	.globl	stub_linear_to_srgb_extended_slice
.type	stub_linear_to_srgb_extended_slice,@function
stub_linear_to_srgb_extended_slice:
	.functype	stub_linear_to_srgb_extended_slice (i32, i32) -> ()
	.local  	i32, i32, i32, i32, i32, i32, i32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, f32, f32, f32
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
	loop
	block
	block
	block
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
	br_if 1
	end_block
	local.get 1
	v128.load 48:p2align=2
	local.tee 9
	f32x4.abs
	local.tee 10
	f32x4.sqrt
	local.tee 11
	local.get 11
	v128.const 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8, 0x1.3deeb6p8
	local.tee 12
	f32x4.add
	f32x4.mul
	v128.const 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12, 0x1.03d88cp12
	local.tee 13
	f32x4.add
	local.set 14
	local.get 11
	local.get 11
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
	local.set 18
	local.get 1
	v128.load 32:p2align=2
	local.tee 19
	f32x4.abs
	local.tee 20
	f32x4.sqrt
	local.tee 21
	local.get 21
	local.get 21
	local.get 21
	local.get 12
	f32x4.add
	f32x4.mul
	local.get 13
	f32x4.add
	f32x4.mul
	v128.const 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13, 0x1.40209ap13
	local.tee 22
	f32x4.add
	f32x4.mul
	v128.const 0x1.788caep12, 0x1.788caep12, 0x1.788caep12, 0x1.788caep12
	local.tee 23
	f32x4.add
	local.set 24
	local.get 21
	local.get 21
	local.get 21
	local.get 21
	local.get 15
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	f32x4.mul
	v128.const 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12, 0x1.bb5aa4p12
	local.tee 25
	f32x4.add
	f32x4.mul
	v128.const 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10, 0x1.00d24ep10
	local.tee 26
	f32x4.add
	local.set 27
	local.get 1
	v128.load 16:p2align=2
	local.tee 28
	f32x4.abs
	local.tee 29
	f32x4.sqrt
	local.tee 30
	local.get 30
	local.get 30
	local.get 30
	local.get 30
	local.get 30
	local.get 12
	f32x4.add
	f32x4.mul
	local.get 13
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.mul
	v128.const 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9, 0x1.9f69dep9
	local.tee 31
	f32x4.add
	f32x4.mul
	v128.const 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4, 0x1.3c64c8p4
	local.tee 32
	f32x4.add
	local.set 33
	local.get 30
	local.get 30
	local.get 30
	local.get 30
	local.get 30
	local.get 30
	local.get 15
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	f32x4.mul
	local.get 25
	f32x4.add
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.mul
	v128.const -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2, -0x1.3aeac2p-2
	local.tee 34
	f32x4.add
	f32x4.mul
	v128.const -0x1.068508p0, -0x1.068508p0, -0x1.068508p0, -0x1.068508p0
	local.tee 35
	f32x4.add
	local.set 36
	local.get 1
	v128.load 0:p2align=2
	local.tee 37
	v128.const -9223372034707292160, -9223372034707292160
	local.tee 38
	v128.and
	local.get 37
	f32x4.abs
	local.tee 39
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	local.tee 40
	f32x4.mul
	local.get 39
	f32x4.sqrt
	local.tee 30
	local.get 30
	local.get 30
	local.get 30
	local.get 30
	local.get 30
	local.get 15
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	f32x4.mul
	local.get 25
	f32x4.add
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.mul
	local.get 34
	f32x4.add
	f32x4.mul
	local.get 35
	f32x4.add
	local.get 30
	local.get 30
	local.get 30
	local.get 30
	local.get 30
	local.get 30
	local.get 12
	f32x4.add
	f32x4.mul
	local.get 13
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 31
	f32x4.add
	f32x4.mul
	local.get 32
	f32x4.add
	f32x4.div
	local.get 39
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	local.tee 12
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	local.set 30
	local.get 37
	f32x4.extract_lane 1
	local.set 41
	local.get 37
	f32x4.extract_lane 0
	local.tee 42
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	br_if 1
	br 2
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
	return
	end_block
	local.get 30
	f32.const 0x1p0
	local.get 42
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 0
	local.set 30
	end_block
	local.get 11
	local.get 14
	f32x4.mul
	local.set 13
	local.get 11
	local.get 18
	f32x4.mul
	local.set 15
	local.get 21
	local.get 24
	f32x4.mul
	local.set 16
	local.get 21
	local.get 27
	f32x4.mul
	local.set 17
	local.get 36
	local.get 33
	f32x4.div
	local.set 39
	local.get 29
	local.get 12
	f32x4.lt
	local.set 14
	local.get 29
	local.get 40
	f32x4.mul
	local.set 29
	local.get 37
	f32x4.extract_lane 2
	local.set 42
	block
	local.get 41
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 30
	f32.const 0x1p0
	local.get 41
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 1
	local.set 30
	end_block
	local.get 13
	local.get 22
	f32x4.add
	local.set 13
	local.get 15
	local.get 25
	f32x4.add
	local.set 15
	local.get 16
	local.get 31
	f32x4.add
	local.set 16
	local.get 17
	local.get 34
	f32x4.add
	local.set 17
	local.get 29
	local.get 39
	local.get 14
	v128.bitselect
	local.set 22
	local.get 37
	f32x4.extract_lane 3
	local.set 41
	block
	local.get 42
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 30
	f32.const 0x1p0
	local.get 42
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 2
	local.set 30
	end_block
	local.get 11
	local.get 13
	f32x4.mul
	local.set 37
	local.get 11
	local.get 15
	f32x4.mul
	local.set 13
	local.get 21
	local.get 16
	f32x4.mul
	local.set 15
	local.get 21
	local.get 17
	f32x4.mul
	local.set 21
	local.get 22
	f32x4.abs
	local.set 16
	local.get 28
	local.get 38
	v128.and
	local.set 17
	local.get 28
	f32x4.extract_lane 0
	local.set 42
	block
	local.get 41
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 30
	f32.const 0x1p0
	local.get 41
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 3
	local.set 30
	end_block
	local.get 37
	local.get 23
	f32x4.add
	local.set 37
	local.get 13
	local.get 26
	f32x4.add
	local.set 13
	local.get 15
	local.get 32
	f32x4.add
	local.set 15
	local.get 21
	local.get 35
	f32x4.add
	local.set 22
	local.get 17
	local.get 16
	v128.or
	local.set 21
	local.get 28
	f32x4.extract_lane 1
	local.set 41
	block
	local.get 42
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 21
	f32.const 0x1p0
	local.get 42
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 0
	local.set 21
	end_block
	local.get 11
	local.get 37
	f32x4.mul
	local.set 37
	local.get 11
	local.get 13
	f32x4.mul
	local.set 13
	local.get 22
	local.get 15
	f32x4.div
	local.set 15
	local.get 20
	local.get 12
	f32x4.lt
	local.set 16
	local.get 20
	local.get 40
	f32x4.mul
	local.set 17
	local.get 28
	f32x4.extract_lane 2
	local.set 42
	block
	local.get 41
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 21
	f32.const 0x1p0
	local.get 41
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 1
	local.set 21
	end_block
	local.get 37
	local.get 31
	f32x4.add
	local.set 37
	local.get 13
	local.get 34
	f32x4.add
	local.set 13
	local.get 17
	local.get 15
	local.get 16
	v128.bitselect
	local.set 15
	local.get 28
	f32x4.extract_lane 3
	local.set 41
	block
	local.get 42
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 21
	f32.const 0x1p0
	local.get 42
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 2
	local.set 21
	end_block
	local.get 11
	local.get 37
	f32x4.mul
	local.set 28
	local.get 11
	local.get 13
	f32x4.mul
	local.set 11
	local.get 15
	f32x4.abs
	local.set 37
	local.get 19
	local.get 38
	v128.and
	local.set 13
	local.get 19
	f32x4.extract_lane 0
	local.set 42
	block
	local.get 41
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 21
	f32.const 0x1p0
	local.get 41
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 3
	local.set 21
	end_block
	local.get 28
	local.get 32
	f32x4.add
	local.set 28
	local.get 11
	local.get 35
	f32x4.add
	local.set 15
	local.get 13
	local.get 37
	v128.or
	local.set 11
	local.get 19
	f32x4.extract_lane 1
	local.set 41
	block
	local.get 42
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 11
	f32.const 0x1p0
	local.get 42
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 0
	local.set 11
	end_block
	local.get 15
	local.get 28
	f32x4.div
	local.set 28
	local.get 10
	local.get 12
	f32x4.lt
	local.set 37
	local.get 10
	local.get 40
	f32x4.mul
	local.set 12
	local.get 19
	f32x4.extract_lane 2
	local.set 42
	block
	local.get 41
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 11
	f32.const 0x1p0
	local.get 41
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 1
	local.set 11
	end_block
	local.get 12
	local.get 28
	local.get 37
	v128.bitselect
	local.set 28
	local.get 19
	f32x4.extract_lane 3
	local.set 41
	block
	local.get 42
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 11
	f32.const 0x1p0
	local.get 42
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 2
	local.set 11
	end_block
	local.get 28
	f32x4.abs
	local.set 19
	local.get 9
	local.get 38
	v128.and
	local.set 28
	local.get 9
	f32x4.extract_lane 0
	local.set 42
	block
	local.get 41
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 11
	f32.const 0x1p0
	local.get 41
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 3
	local.set 11
	end_block
	local.get 28
	local.get 19
	v128.or
	local.set 19
	local.get 9
	f32x4.extract_lane 1
	local.set 41
	block
	local.get 42
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 19
	f32.const 0x1p0
	local.get 42
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 0
	local.set 19
	end_block
	local.get 9
	f32x4.extract_lane 2
	local.set 42
	block
	local.get 41
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 19
	f32.const 0x1p0
	local.get 41
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 1
	local.set 19
	end_block
	local.get 9
	f32x4.extract_lane 3
	local.set 41
	block
	local.get 42
	f32.abs
	local.tee 43
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 19
	f32.const 0x1p0
	local.get 42
	f32.copysign
	local.get 43
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 2
	local.set 19
	end_block
	block
	local.get 41
	f32.abs
	local.tee 42
	f32.const 0x1p6
	f32.gt
	i32.eqz
	br_if 0
	local.get 19
	f32.const 0x1p0
	local.get 41
	f32.copysign
	local.get 42
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul
	f32.const -0x1.c2a5d6p-5
	f32.add
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 3
	local.set 19
	end_block
	local.get 1
	local.get 19
	v128.store 48:p2align=2
	local.get 1
	local.get 11
	v128.store 32:p2align=2
	local.get 1
	local.get 21
	v128.store 16:p2align=2
	local.get 1
	local.get 30
	v128.store 0:p2align=2
	local.get 3
	local.set 5
	br 0
	end_loop
	end_function
