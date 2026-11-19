.section .text.stub_srgb_to_linear_extended_slice,"",@
	.globl	stub_srgb_to_linear_extended_slice
.type	stub_srgb_to_linear_extended_slice,@function
stub_srgb_to_linear_extended_slice:
	.functype	stub_srgb_to_linear_extended_slice (i32, i32) -> ()
	.local  	i32, i32, i32, i32, i32, i32, i32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, f32, f32
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
	local.get 10
	v128.const -0x1.fcb852p5, -0x1.fcb852p5, -0x1.fcb852p5, -0x1.fcb852p5
	local.tee 11
	f32x4.add
	f32x4.mul
	v128.const 0x1.1793fcp12, 0x1.1793fcp12, 0x1.1793fcp12, 0x1.1793fcp12
	local.tee 12
	f32x4.add
	local.set 13
	local.get 10
	local.get 10
	v128.const 0x1.0aa4c6p14, 0x1.0aa4c6p14, 0x1.0aa4c6p14, 0x1.0aa4c6p14
	local.tee 14
	f32x4.mul
	v128.const 0x1.056176p17, 0x1.056176p17, 0x1.056176p17, 0x1.056176p17
	local.tee 15
	f32x4.add
	f32x4.mul
	v128.const 0x1.adac9p17, 0x1.adac9p17, 0x1.adac9p17, 0x1.adac9p17
	local.tee 16
	f32x4.add
	local.set 17
	local.get 1
	v128.load 32:p2align=2
	local.tee 18
	f32x4.abs
	local.tee 19
	local.get 19
	local.get 19
	local.get 19
	local.get 11
	f32x4.add
	f32x4.mul
	local.get 12
	f32x4.add
	f32x4.mul
	v128.const 0x1.41da1cp16, 0x1.41da1cp16, 0x1.41da1cp16, 0x1.41da1cp16
	local.tee 20
	f32x4.add
	f32x4.mul
	v128.const 0x1.c1dcf4p17, 0x1.c1dcf4p17, 0x1.c1dcf4p17, 0x1.c1dcf4p17
	local.tee 21
	f32x4.add
	local.set 22
	local.get 19
	local.get 19
	local.get 19
	local.get 19
	local.get 14
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	v128.const 0x1.8eafd2p16, 0x1.8eafd2p16, 0x1.8eafd2p16, 0x1.8eafd2p16
	local.tee 23
	f32x4.add
	f32x4.mul
	v128.const 0x1.ead02cp13, 0x1.ead02cp13, 0x1.ead02cp13, 0x1.ead02cp13
	local.tee 24
	f32x4.add
	local.set 25
	local.get 1
	v128.load 16:p2align=2
	local.tee 26
	f32x4.abs
	local.tee 27
	local.get 27
	local.get 27
	local.get 27
	local.get 27
	local.get 27
	local.get 11
	f32x4.add
	f32x4.mul
	local.get 12
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	v128.const 0x1.26a3c2p17, 0x1.26a3c2p17, 0x1.26a3c2p17, 0x1.26a3c2p17
	local.tee 28
	f32x4.add
	f32x4.mul
	v128.const 0x1.516812p14, 0x1.516812p14, 0x1.516812p14, 0x1.516812p14
	local.tee 29
	f32x4.add
	local.set 30
	local.get 27
	local.get 27
	local.get 27
	local.get 27
	local.get 27
	local.get 27
	local.get 14
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 24
	f32x4.add
	f32x4.mul
	v128.const 0x1.c78544p9, 0x1.c78544p9, 0x1.c78544p9, 0x1.c78544p9
	local.tee 31
	f32x4.add
	f32x4.mul
	v128.const 0x1.205782p4, 0x1.205782p4, 0x1.205782p4, 0x1.205782p4
	local.tee 32
	f32x4.add
	local.set 33
	local.get 1
	v128.load 0:p2align=2
	local.tee 34
	v128.const -9223372034707292160, -9223372034707292160
	local.tee 35
	v128.and
	local.get 34
	f32x4.abs
	local.tee 36
	v128.const 0x1.3d0722p-4, 0x1.3d0722p-4, 0x1.3d0722p-4, 0x1.3d0722p-4
	local.tee 37
	f32x4.mul
	local.get 36
	local.get 36
	local.get 36
	local.get 36
	local.get 36
	local.get 36
	local.get 14
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 24
	f32x4.add
	f32x4.mul
	local.get 31
	f32x4.add
	f32x4.mul
	local.get 32
	f32x4.add
	local.get 36
	local.get 36
	local.get 36
	local.get 36
	local.get 36
	local.get 36
	local.get 11
	f32x4.add
	f32x4.mul
	local.get 12
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 28
	f32x4.add
	f32x4.mul
	local.get 29
	f32x4.add
	f32x4.div
	local.get 36
	v128.const 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5
	local.tee 11
	f32x4.lt
	v128.bitselect
	f32x4.abs
	v128.or
	local.set 36
	local.get 34
	f32x4.extract_lane 0
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 1
	local.get 36
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 0
	local.set 36
	br 1
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
	local.get 10
	local.get 13
	f32x4.mul
	local.set 12
	local.get 10
	local.get 17
	f32x4.mul
	local.set 14
	local.get 19
	local.get 22
	f32x4.mul
	local.set 15
	local.get 19
	local.get 25
	f32x4.mul
	local.set 16
	local.get 33
	local.get 30
	f32x4.div
	local.set 13
	local.get 27
	local.get 11
	f32x4.lt
	local.set 17
	local.get 27
	local.get 37
	f32x4.mul
	local.set 27
	block
	local.get 34
	f32x4.extract_lane 1
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 36
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 1
	local.set 36
	end_block
	local.get 12
	local.get 20
	f32x4.add
	local.set 12
	local.get 14
	local.get 23
	f32x4.add
	local.set 14
	local.get 15
	local.get 28
	f32x4.add
	local.set 15
	local.get 16
	local.get 31
	f32x4.add
	local.set 16
	local.get 27
	local.get 13
	local.get 17
	v128.bitselect
	local.set 27
	block
	local.get 34
	f32x4.extract_lane 2
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 36
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 2
	local.set 36
	end_block
	local.get 10
	local.get 12
	f32x4.mul
	local.set 12
	local.get 10
	local.get 14
	f32x4.mul
	local.set 14
	local.get 19
	local.get 15
	f32x4.mul
	local.set 15
	local.get 19
	local.get 16
	f32x4.mul
	local.set 16
	local.get 27
	f32x4.abs
	local.set 27
	local.get 26
	local.get 35
	v128.and
	local.set 20
	block
	local.get 34
	f32x4.extract_lane 3
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 36
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 3
	local.set 36
	end_block
	local.get 12
	local.get 21
	f32x4.add
	local.set 34
	local.get 14
	local.get 24
	f32x4.add
	local.set 12
	local.get 15
	local.get 29
	f32x4.add
	local.set 14
	local.get 16
	local.get 32
	f32x4.add
	local.set 15
	local.get 20
	local.get 27
	v128.or
	local.set 27
	block
	local.get 26
	f32x4.extract_lane 0
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 27
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 0
	local.set 27
	end_block
	local.get 10
	local.get 34
	f32x4.mul
	local.set 34
	local.get 10
	local.get 12
	f32x4.mul
	local.set 12
	local.get 15
	local.get 14
	f32x4.div
	local.set 14
	local.get 19
	local.get 11
	f32x4.lt
	local.set 15
	local.get 19
	local.get 37
	f32x4.mul
	local.set 19
	block
	local.get 26
	f32x4.extract_lane 1
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 27
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 1
	local.set 27
	end_block
	local.get 34
	local.get 28
	f32x4.add
	local.set 34
	local.get 12
	local.get 31
	f32x4.add
	local.set 12
	local.get 19
	local.get 14
	local.get 15
	v128.bitselect
	local.set 19
	block
	local.get 26
	f32x4.extract_lane 2
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 27
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 2
	local.set 27
	end_block
	local.get 10
	local.get 34
	f32x4.mul
	local.set 34
	local.get 10
	local.get 12
	f32x4.mul
	local.set 12
	local.get 19
	f32x4.abs
	local.set 19
	local.get 18
	local.get 35
	v128.and
	local.set 14
	block
	local.get 26
	f32x4.extract_lane 3
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 27
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 3
	local.set 27
	end_block
	local.get 34
	local.get 29
	f32x4.add
	local.set 26
	local.get 12
	local.get 32
	f32x4.add
	local.set 34
	local.get 14
	local.get 19
	v128.or
	local.set 19
	block
	local.get 18
	f32x4.extract_lane 0
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 19
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 0
	local.set 19
	end_block
	local.get 34
	local.get 26
	f32x4.div
	local.set 26
	local.get 10
	local.get 11
	f32x4.lt
	local.set 34
	local.get 10
	local.get 37
	f32x4.mul
	local.set 10
	block
	local.get 18
	f32x4.extract_lane 1
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 19
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 1
	local.set 19
	end_block
	local.get 10
	local.get 26
	local.get 34
	v128.bitselect
	local.set 10
	block
	local.get 18
	f32x4.extract_lane 2
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 19
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 2
	local.set 19
	end_block
	local.get 10
	f32x4.abs
	local.set 10
	local.get 9
	local.get 35
	v128.and
	local.set 26
	block
	local.get 18
	f32x4.extract_lane 3
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 19
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 3
	local.set 19
	end_block
	local.get 26
	local.get 10
	v128.or
	local.set 10
	block
	local.get 9
	f32x4.extract_lane 0
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 10
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 0
	local.set 10
	end_block
	block
	local.get 9
	f32x4.extract_lane 1
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 10
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 1
	local.set 10
	end_block
	block
	local.get 9
	f32x4.extract_lane 2
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 10
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 2
	local.set 10
	end_block
	block
	local.get 9
	f32x4.extract_lane 3
	local.tee 38
	f32.abs
	local.tee 39
	f32.const 0x1p3
	f32.gt
	i32.eqz
	br_if 0
	local.get 10
	f32.const 0x1p0
	local.get 38
	f32.copysign
	local.get 39
	f32.const 0x1.c2a5d6p-5
	f32.add
	f32.const 0x1.0e152ep0
	f32.div
	f32.const 0x1.333334p1
	call powf
	f32.mul
	i32.reinterpret_f32
	i32x4.replace_lane 3
	local.set 10
	end_block
	local.get 1
	local.get 10
	v128.store 48:p2align=2
	local.get 1
	local.get 19
	v128.store 32:p2align=2
	local.get 1
	local.get 27
	v128.store 16:p2align=2
	local.get 1
	local.get 36
	v128.store 0:p2align=2
	local.get 3
	local.set 5
	br 0
	end_loop
	end_function
