.section .text.stub_unpremultiply_linear_to_gamma_rgba_slice,"",@
	.globl	stub_unpremultiply_linear_to_gamma_rgba_slice
.type	stub_unpremultiply_linear_to_gamma_rgba_slice,@function
stub_unpremultiply_linear_to_gamma_rgba_slice:
	.functype	stub_unpremultiply_linear_to_gamma_rgba_slice (i32, i32, f32) -> ()
	.local  	i32, i32, i32, i32, v128, i32, v128, i32, v128, v128, v128, v128, v128, f32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128
	global.get __stack_pointer
	i32.const 192
	i32.sub
	local.tee 3
	global.set __stack_pointer
	block
	block
	block
	f32.const 0x1p0
	local.get 2
	f32.div
	local.tee 2
	f32.const 0x0p0
	f32.ge
	i32.eqz
	br_if 0
	local.get 2
	f32.const 0x1p2
	f32.le
	br_if 1
	end_block
	block
	local.get 1
	i32.const 2
	i32.shl
	i32.const 2147483584
	i32.and
	local.tee 4
	i32.eqz
	br_if 0
	local.get 0
	local.get 4
	i32.add
	local.set 5
	local.get 0
	local.set 4
	loop
	local.get 4
	i32.const 16
	i32.add
	local.tee 6
	v128.load 0:p2align=0
	local.set 7
	local.get 4
	i32.const 32
	i32.add
	local.tee 8
	v128.load 0:p2align=0
	local.set 9
	local.get 4
	i32.const 48
	i32.add
	local.tee 10
	v128.load 0:p2align=0
	local.set 11
	local.get 3
	local.get 4
	v128.load 0:p2align=0
	local.tee 12
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 13
	v128.const 0x1p-10, 0x1p-10, 0x1p-10, 0x1p-10
	local.tee 14
	f32x4.gt
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 15
	local.get 13
	f32x4.div
	v128.and
	local.get 12
	f32x4.mul
	v128.store 128
	local.get 3
	local.get 11
	local.get 11
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 13
	local.get 14
	f32x4.gt
	local.get 15
	local.get 13
	f32x4.div
	v128.and
	f32x4.mul
	v128.store 176
	local.get 3
	local.get 9
	local.get 9
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 13
	local.get 14
	f32x4.gt
	local.get 15
	local.get 13
	f32x4.div
	v128.and
	f32x4.mul
	v128.store 160
	local.get 3
	local.get 7
	local.get 7
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 13
	local.get 14
	f32x4.gt
	local.get 15
	local.get 13
	f32x4.div
	v128.and
	f32x4.mul
	v128.store 144
	local.get 3
	i32.const 64
	i32.add
	local.get 3
	i32.const 128
	i32.add
	local.get 2
	call linear_srgb::tf::math::gamma_outside_fit::<16>
	local.get 10
	local.get 3
	v128.load 112
	local.get 11
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 8
	local.get 3
	v128.load 96
	local.get 9
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 6
	local.get 3
	v128.load 80
	local.get 7
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 4
	local.get 3
	v128.load 64
	local.get 12
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 4
	i32.const 64
	i32.add
	local.tee 4
	local.get 5
	i32.ne
	br_if 0
	end_loop
	end_block
	local.get 1
	i32.const 12
	i32.and
	local.tee 4
	i32.eqz
	br_if 1
	local.get 1
	i32.const 536870896
	i32.and
	i32.const 2
	i32.shl
	local.set 6
	block
	i32.const 64
	local.get 4
	i32.const 2
	i32.shl
	local.tee 4
	i32.sub
	local.tee 1
	i32.eqz
	br_if 0
	local.get 3
	local.get 4
	i32.add
	i32.const 0
	local.get 1
	memory.fill 0
	end_block
	local.get 0
	local.get 6
	i32.add
	local.set 0
	block
	local.get 4
	i32.eqz
	local.tee 6
	br_if 0
	local.get 3
	local.get 0
	local.get 4
	memory.copy 0, 0
	end_block
	local.get 3
	v128.load 0
	local.set 7
	local.get 3
	v128.load 16
	local.set 9
	local.get 3
	v128.load 32
	local.set 11
	local.get 3
	local.get 3
	v128.load 48
	local.tee 12
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 13
	v128.const 0x1p-10, 0x1p-10, 0x1p-10, 0x1p-10
	local.tee 14
	f32x4.gt
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 15
	local.get 13
	f32x4.div
	v128.and
	local.get 12
	f32x4.mul
	v128.store 176
	local.get 3
	local.get 11
	local.get 11
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 13
	local.get 14
	f32x4.gt
	local.get 15
	local.get 13
	f32x4.div
	v128.and
	f32x4.mul
	v128.store 160
	local.get 3
	local.get 9
	local.get 9
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 13
	local.get 14
	f32x4.gt
	local.get 15
	local.get 13
	f32x4.div
	v128.and
	f32x4.mul
	v128.store 144
	local.get 3
	local.get 7
	local.get 7
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 13
	local.get 14
	f32x4.gt
	local.get 15
	local.get 13
	f32x4.div
	v128.and
	f32x4.mul
	v128.store 128
	local.get 3
	i32.const 64
	i32.add
	local.get 3
	i32.const 128
	i32.add
	local.get 2
	call linear_srgb::tf::math::gamma_outside_fit::<16>
	local.get 3
	local.get 3
	v128.load 112
	local.get 12
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 48
	local.get 3
	local.get 3
	v128.load 96
	local.get 11
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 32
	local.get 3
	local.get 3
	v128.load 80
	local.get 9
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 16
	local.get 3
	local.get 3
	v128.load 64
	local.get 7
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0
	local.get 6
	br_if 1
	local.get 0
	local.get 3
	local.get 4
	memory.copy 0, 0
	br 1
	end_block
	local.get 0
	local.get 1
	i32.const 2
	i32.shl
	i32.const 2147483584
	i32.and
	local.tee 4
	i32.add
	local.set 6
	local.get 0
	local.get 1
	i32.const 536870896
	i32.and
	i32.const 2
	i32.shl
	i32.add
	local.set 8
	block
	local.get 2
	f32.const 0x0p0
	f32.eq
	br_if 0
	block
	local.get 4
	i32.eqz
	br_if 0
	f32.const 0x0p0
	f32.const 0x1p2
	local.get 2
	local.get 2
	f32.const 0x1p2
	f32.gt
	f32.select
	local.tee 16
	local.get 16
	i32.reinterpret_f32
	i32.const -1
	i32.add
	i32.const 8388607
	i32.lt_u
	local.tee 4
	f32.select
	f32x4.splat
	local.set 17
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	local.tee 18
	v128.const 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	local.tee 12
	local.get 4
	v128.select
	local.set 19
	loop
	local.get 0
	local.get 0
	v128.load 0:p2align=0
	local.tee 20
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 7
	v128.const 0x1p-10, 0x1p-10, 0x1p-10, 0x1p-10
	local.tee 21
	f32x4.gt
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 22
	local.get 7
	f32x4.div
	v128.and
	local.get 20
	f32x4.mul
	local.tee 23
	local.get 19
	v128.const 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127
	local.tee 24
	local.get 12
	v128.const 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
	local.tee 9
	v128.const -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
	local.tee 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 17
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 18
	local.get 23
	local.get 23
	v128.const 0x1p-126, 0x1p-126, 0x1p-126, 0x1p-126
	local.tee 25
	f32x4.ge
	v128.and
	local.tee 7
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	local.tee 26
	local.get 7
	local.get 22
	f32x4.lt
	v128.bitselect
	local.tee 7
	local.get 7
	v128.const 0, 0, 0, 0
	local.tee 27
	i32x4.eq
	local.tee 28
	v128.bitselect
	v128.const 4913933, 4913933, 4913933, 4913933
	local.tee 29
	i32x4.add
	local.tee 30
	v128.const 8388607, 8388607, 8388607, 8388607
	local.tee 31
	v128.and
	v128.const 1060439283, 1060439283, 1060439283, 1060439283
	local.tee 32
	i32x4.add
	local.tee 7
	v128.const -0x1p0, -0x1p0, -0x1p0, -0x1p0
	local.tee 33
	f32x4.add
	local.get 7
	local.get 22
	f32x4.add
	f32x4.div
	local.tee 34
	local.get 34
	f32x4.mul
	local.tee 35
	f64x2.promote_low_f32x4
	local.tee 36
	v128.const 0x1.bcd67ep-2, 0x1.bcd67ep-2
	local.tee 37
	f64x2.mul
	local.tee 14
	v128.const 0x1.276932p-1, 0x1.276932p-1
	local.tee 13
	f64x2.add
	local.tee 38
	v128.const 0x0p0, 0x0p0
	local.tee 7
	f64x2.gt
	local.get 13
	local.get 38
	local.get 14
	f64x2.sub
	local.tee 15
	f64x2.sub
	local.get 14
	local.get 15
	local.get 38
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 39
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 38
	i64x2.add
	local.get 38
	i64.const -1
	i64.const 0
	local.get 38
	i32.const 1
	i64x2.shl
	local.tee 14
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 14
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 38
	v128.const 1, 1
	local.tee 14
	v128.and
	v128.const 0, 0
	local.tee 15
	i64x2.eq
	local.get 39
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 38
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 38
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 36
	f64x2.mul
	local.tee 40
	v128.const 0x1.ec7126p-1, 0x1.ec7126p-1
	local.tee 38
	f64x2.add
	local.tee 39
	local.get 7
	f64x2.gt
	local.get 38
	local.get 39
	local.get 40
	f64x2.sub
	local.tee 41
	f64x2.sub
	local.get 40
	local.get 39
	local.get 41
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 39
	i64x2.add
	local.get 39
	i64.const -1
	i64.const 0
	local.get 39
	i32.const 1
	i64x2.shl
	local.tee 41
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 41
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 39
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 40
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 39
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 39
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 36
	f64x2.mul
	local.tee 40
	v128.const 0x1.715476p1, 0x1.715476p1
	local.tee 39
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 39
	local.get 36
	local.get 40
	f64x2.sub
	local.tee 41
	f64x2.sub
	local.get 40
	local.get 36
	local.get 41
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 41
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 41
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 40
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 34
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 40
	local.get 30
	i32.const 23
	i32x4.shr_s
	v128.const -127, -127, -127, -127
	local.tee 42
	i32x4.add
	f32x4.convert_i32x4_s
	local.tee 41
	f64x2.promote_low_f32x4
	local.tee 30
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 30
	local.get 36
	local.get 40
	f64x2.sub
	local.tee 43
	f64x2.sub
	local.get 40
	local.get 36
	local.get 43
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 30
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 30
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 40
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 35
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 40
	local.get 37
	f64x2.mul
	local.tee 30
	local.get 13
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 13
	local.get 36
	local.get 30
	f64x2.sub
	local.tee 35
	f64x2.sub
	local.get 30
	local.get 35
	local.get 36
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 30
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 35
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 35
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 30
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 40
	f64x2.mul
	local.tee 30
	local.get 38
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 38
	local.get 36
	local.get 30
	f64x2.sub
	local.tee 35
	f64x2.sub
	local.get 30
	local.get 36
	local.get 35
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 30
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 35
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 35
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 30
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 40
	f64x2.mul
	local.tee 40
	local.get 39
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 39
	local.get 36
	local.get 40
	f64x2.sub
	local.tee 30
	f64x2.sub
	local.get 40
	local.get 36
	local.get 30
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 30
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 30
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 40
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 34
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 34
	local.get 41
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 40
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 40
	local.get 36
	local.get 34
	f64x2.sub
	local.tee 30
	f64x2.sub
	local.get 34
	local.get 36
	local.get 30
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 34
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 40
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 40
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 34
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	f32x4.mul
	local.tee 44
	v128.const -0x1.f8p6, -0x1.f8p6, -0x1.f8p6, -0x1.f8p6
	local.tee 43
	local.get 44
	f32x4.pmax
	local.get 12
	v128.bitselect
	local.tee 36
	v128.const 0x1p7, 0x1p7, 0x1p7, 0x1p7
	local.tee 45
	local.get 36
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 36
	local.get 36
	f32x4.nearest
	local.tee 36
	v128.const 0x1.fcp6, 0x1.fcp6, 0x1.fcp6, 0x1.fcp6
	local.tee 46
	local.get 36
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 47
	f32x4.sub
	local.tee 48
	f64x2.promote_low_f32x4
	local.tee 49
	v128.const 0x1.43f274p-13, 0x1.43f274p-13
	local.tee 50
	f64x2.mul
	local.tee 40
	v128.const 0x1.5d88f2p-10, 0x1.5d88f2p-10
	local.tee 36
	f64x2.add
	local.tee 34
	local.get 7
	f64x2.gt
	local.get 36
	local.get 34
	local.get 40
	f64x2.sub
	local.tee 30
	f64x2.sub
	local.get 40
	local.get 34
	local.get 30
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 34
	i64x2.add
	local.get 34
	i64.const -1
	i64.const 0
	local.get 34
	i32.const 1
	i64x2.shl
	local.tee 30
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 30
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 34
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 40
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 34
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 34
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 49
	f64x2.mul
	local.tee 30
	v128.const 0x1.3b2a18p-7, 0x1.3b2a18p-7
	local.tee 34
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	local.get 34
	local.get 40
	local.get 30
	f64x2.sub
	local.tee 35
	f64x2.sub
	local.get 30
	local.get 40
	local.get 35
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 30
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 40
	i64x2.add
	local.get 40
	i64.const -1
	i64.const 0
	local.get 40
	i32.const 1
	i64x2.shl
	local.tee 35
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 35
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 40
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 30
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 40
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 40
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 49
	f64x2.mul
	local.tee 35
	v128.const 0x1.c6b178p-5, 0x1.c6b178p-5
	local.tee 40
	f64x2.add
	local.tee 30
	local.get 7
	f64x2.gt
	local.get 40
	local.get 30
	local.get 35
	f64x2.sub
	local.tee 41
	f64x2.sub
	local.get 35
	local.get 30
	local.get 41
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 35
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 30
	i64x2.add
	local.get 30
	i64.const -1
	i64.const 0
	local.get 30
	i32.const 1
	i64x2.shl
	local.tee 41
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 41
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 30
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 35
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 30
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 30
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 49
	f64x2.mul
	local.tee 41
	v128.const 0x1.ebfbdap-3, 0x1.ebfbdap-3
	local.tee 30
	f64x2.add
	local.tee 35
	local.get 7
	f64x2.gt
	local.get 30
	local.get 35
	local.get 41
	f64x2.sub
	local.tee 51
	f64x2.sub
	local.get 41
	local.get 35
	local.get 51
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 41
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 35
	i64x2.add
	local.get 35
	i64.const -1
	i64.const 0
	local.get 35
	i32.const 1
	i64x2.shl
	local.tee 51
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 51
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 35
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 41
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 35
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 35
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 49
	f64x2.mul
	local.tee 51
	v128.const 0x1.62e43p-1, 0x1.62e43p-1
	local.tee 35
	f64x2.add
	local.tee 41
	local.get 7
	f64x2.gt
	local.get 35
	local.get 41
	local.get 51
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 51
	local.get 41
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 41
	i64x2.add
	local.get 41
	i64.const -1
	i64.const 0
	local.get 41
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 41
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 41
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 41
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 49
	f64x2.mul
	local.tee 51
	v128.const 0x1p0, 0x1p0
	local.tee 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 51
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 48
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 51
	local.get 50
	f64x2.mul
	local.tee 48
	local.get 36
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 36
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 34
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 34
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 40
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 40
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 30
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 30
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 35
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 35
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 51
	local.get 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	local.get 47
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 26
	i32x4.add
	f32x4.mul
	local.get 44
	local.get 43
	f32x4.lt
	v128.bitselect
	local.get 44
	local.get 45
	f32x4.ge
	v128.bitselect
	local.get 28
	v128.bitselect
	local.get 23
	local.get 23
	f32x4.ne
	v128.bitselect
	local.get 20
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 0
	i32.const 48
	i32.add
	local.tee 4
	local.get 4
	v128.load 0:p2align=0
	local.tee 20
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 49
	local.get 21
	f32x4.gt
	local.get 22
	local.get 49
	f32x4.div
	v128.and
	local.get 20
	f32x4.mul
	local.tee 23
	local.get 19
	local.get 24
	local.get 12
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 17
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 18
	local.get 23
	local.get 23
	local.get 25
	f32x4.ge
	v128.and
	local.tee 49
	local.get 26
	local.get 49
	local.get 22
	f32x4.lt
	v128.bitselect
	local.tee 49
	local.get 49
	local.get 27
	i32x4.eq
	local.tee 28
	v128.bitselect
	local.get 29
	i32x4.add
	local.tee 47
	local.get 31
	v128.and
	local.get 32
	i32x4.add
	local.tee 49
	local.get 33
	f32x4.add
	local.get 49
	local.get 22
	f32x4.add
	f32x4.div
	local.tee 51
	local.get 51
	f32x4.mul
	local.tee 52
	f64x2.promote_low_f32x4
	local.tee 44
	local.get 37
	f64x2.mul
	local.tee 48
	local.get 13
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 13
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 53
	local.get 49
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 44
	f64x2.mul
	local.tee 48
	local.get 38
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 38
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 44
	f64x2.mul
	local.tee 44
	local.get 39
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 39
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 44
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 44
	local.get 47
	i32.const 23
	i32x4.shr_s
	local.get 42
	i32x4.add
	f32x4.convert_i32x4_s
	local.tee 47
	f64x2.promote_low_f32x4
	local.tee 48
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 48
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 44
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 52
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 44
	local.get 37
	f64x2.mul
	local.tee 48
	local.get 13
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 13
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 52
	local.get 49
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 44
	f64x2.mul
	local.tee 48
	local.get 38
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 38
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 44
	f64x2.mul
	local.tee 44
	local.get 39
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 39
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 44
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 51
	local.get 47
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 44
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 44
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 44
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 44
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	f32x4.mul
	local.tee 44
	local.get 43
	local.get 44
	f32x4.pmax
	local.get 12
	v128.bitselect
	local.tee 49
	local.get 45
	local.get 49
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 49
	local.get 49
	f32x4.nearest
	local.tee 49
	local.get 46
	local.get 49
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 47
	f32x4.sub
	local.tee 52
	f64x2.promote_low_f32x4
	local.tee 51
	local.get 50
	f64x2.mul
	local.tee 48
	local.get 36
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 36
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 34
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 34
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 40
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 40
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 30
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 30
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 35
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 35
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 51
	local.get 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 52
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 51
	local.get 50
	f64x2.mul
	local.tee 48
	local.get 36
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 36
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 34
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 34
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 40
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 40
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 30
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 30
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 35
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 35
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 51
	local.get 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	local.get 47
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 26
	i32x4.add
	f32x4.mul
	local.get 44
	local.get 43
	f32x4.lt
	v128.bitselect
	local.get 44
	local.get 45
	f32x4.ge
	v128.bitselect
	local.get 28
	v128.bitselect
	local.get 23
	local.get 23
	f32x4.ne
	v128.bitselect
	local.get 20
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 0
	i32.const 32
	i32.add
	local.tee 4
	local.get 4
	v128.load 0:p2align=0
	local.tee 20
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 49
	local.get 21
	f32x4.gt
	local.get 22
	local.get 49
	f32x4.div
	v128.and
	local.get 20
	f32x4.mul
	local.tee 23
	local.get 19
	local.get 24
	local.get 12
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 17
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 18
	local.get 23
	local.get 23
	local.get 25
	f32x4.ge
	v128.and
	local.tee 49
	local.get 26
	local.get 49
	local.get 22
	f32x4.lt
	v128.bitselect
	local.tee 49
	local.get 49
	local.get 27
	i32x4.eq
	local.tee 28
	v128.bitselect
	local.get 29
	i32x4.add
	local.tee 47
	local.get 31
	v128.and
	local.get 32
	i32x4.add
	local.tee 49
	local.get 33
	f32x4.add
	local.get 49
	local.get 22
	f32x4.add
	f32x4.div
	local.tee 51
	local.get 51
	f32x4.mul
	local.tee 52
	f64x2.promote_low_f32x4
	local.tee 44
	local.get 37
	f64x2.mul
	local.tee 48
	local.get 13
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 13
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 53
	local.get 49
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 44
	f64x2.mul
	local.tee 48
	local.get 38
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 38
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 44
	f64x2.mul
	local.tee 44
	local.get 39
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 39
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 44
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 44
	local.get 47
	i32.const 23
	i32x4.shr_s
	local.get 42
	i32x4.add
	f32x4.convert_i32x4_s
	local.tee 47
	f64x2.promote_low_f32x4
	local.tee 48
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 48
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 44
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 52
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 44
	local.get 37
	f64x2.mul
	local.tee 48
	local.get 13
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 13
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 52
	local.get 49
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 44
	f64x2.mul
	local.tee 48
	local.get 38
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 38
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 44
	f64x2.mul
	local.tee 44
	local.get 39
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 39
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 44
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 51
	local.get 47
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 44
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 44
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 44
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 44
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	f32x4.mul
	local.tee 44
	local.get 43
	local.get 44
	f32x4.pmax
	local.get 12
	v128.bitselect
	local.tee 49
	local.get 45
	local.get 49
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 49
	local.get 49
	f32x4.nearest
	local.tee 49
	local.get 46
	local.get 49
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 47
	f32x4.sub
	local.tee 52
	f64x2.promote_low_f32x4
	local.tee 51
	local.get 50
	f64x2.mul
	local.tee 48
	local.get 36
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 36
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 34
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 34
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 40
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 40
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 30
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 30
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 35
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 35
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 51
	local.get 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 52
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 51
	local.get 50
	f64x2.mul
	local.tee 48
	local.get 36
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 36
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 34
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 34
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 40
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 40
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 30
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 30
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 35
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 35
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 51
	local.get 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	local.get 47
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 26
	i32x4.add
	f32x4.mul
	local.get 44
	local.get 43
	f32x4.lt
	v128.bitselect
	local.get 44
	local.get 45
	f32x4.ge
	v128.bitselect
	local.get 28
	v128.bitselect
	local.get 23
	local.get 23
	f32x4.ne
	v128.bitselect
	local.get 20
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 0
	i32.const 16
	i32.add
	local.tee 4
	local.get 4
	v128.load 0:p2align=0
	local.tee 23
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 49
	local.get 21
	f32x4.gt
	local.get 22
	local.get 49
	f32x4.div
	v128.and
	local.get 23
	f32x4.mul
	local.tee 49
	local.get 19
	local.get 24
	local.get 12
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 17
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 18
	local.get 49
	local.get 49
	local.get 25
	f32x4.ge
	v128.and
	local.tee 51
	local.get 26
	local.get 51
	local.get 22
	f32x4.lt
	v128.bitselect
	local.tee 51
	local.get 51
	local.get 27
	i32x4.eq
	local.tee 27
	v128.bitselect
	local.get 29
	i32x4.add
	local.tee 29
	local.get 31
	v128.and
	local.get 32
	i32x4.add
	local.tee 51
	local.get 33
	f32x4.add
	local.get 51
	local.get 22
	f32x4.add
	f32x4.div
	local.tee 51
	local.get 51
	f32x4.mul
	local.tee 31
	f64x2.promote_low_f32x4
	local.tee 21
	local.get 37
	f64x2.mul
	local.tee 25
	local.get 13
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	local.get 13
	local.get 22
	local.get 25
	f64x2.sub
	local.tee 32
	f64x2.sub
	local.get 25
	local.get 32
	local.get 22
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 25
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 22
	i64x2.add
	local.get 22
	i64.const -1
	i64.const 0
	local.get 22
	i32.const 1
	i64x2.shl
	local.tee 32
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 32
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 22
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 25
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 22
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 22
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 21
	f64x2.mul
	local.tee 25
	local.get 38
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	local.get 38
	local.get 22
	local.get 25
	f64x2.sub
	local.tee 32
	f64x2.sub
	local.get 25
	local.get 22
	local.get 32
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 25
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 22
	i64x2.add
	local.get 22
	i64.const -1
	i64.const 0
	local.get 22
	i32.const 1
	i64x2.shl
	local.tee 32
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 32
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 22
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 25
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 22
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 22
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 21
	f64x2.mul
	local.tee 21
	local.get 39
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	local.get 39
	local.get 22
	local.get 21
	f64x2.sub
	local.tee 25
	f64x2.sub
	local.get 21
	local.get 22
	local.get 25
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 21
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 22
	i64x2.add
	local.get 22
	i64.const -1
	i64.const 0
	local.get 22
	i32.const 1
	i64x2.shl
	local.tee 25
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 25
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 22
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 21
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 22
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 22
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 21
	local.get 29
	i32.const 23
	i32x4.shr_s
	local.get 42
	i32x4.add
	f32x4.convert_i32x4_s
	local.tee 25
	f64x2.promote_low_f32x4
	local.tee 29
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	local.get 29
	local.get 22
	local.get 21
	f64x2.sub
	local.tee 32
	f64x2.sub
	local.get 21
	local.get 22
	local.get 32
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 21
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 22
	i64x2.add
	local.get 22
	i64.const -1
	i64.const 0
	local.get 22
	i32.const 1
	i64x2.shl
	local.tee 29
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 29
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 22
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 21
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 22
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 22
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 31
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 21
	local.get 37
	f64x2.mul
	local.tee 37
	local.get 13
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	local.get 13
	local.get 22
	local.get 37
	f64x2.sub
	local.tee 29
	f64x2.sub
	local.get 37
	local.get 29
	local.get 22
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 22
	i64x2.add
	local.get 22
	i64.const -1
	i64.const 0
	local.get 22
	i32.const 1
	i64x2.shl
	local.tee 37
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 37
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 22
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 13
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 21
	f64x2.mul
	local.tee 22
	local.get 38
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 38
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 37
	f64x2.sub
	local.get 22
	local.get 13
	local.get 37
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 38
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 22
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 22
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 38
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 21
	f64x2.mul
	local.tee 38
	local.get 39
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 39
	local.get 13
	local.get 38
	f64x2.sub
	local.tee 22
	f64x2.sub
	local.get 38
	local.get 13
	local.get 22
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 38
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 39
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 39
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 38
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 38
	local.get 25
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 39
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 39
	local.get 13
	local.get 38
	f64x2.sub
	local.tee 22
	f64x2.sub
	local.get 38
	local.get 13
	local.get 22
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 38
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 39
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 39
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 38
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	f32x4.mul
	local.tee 39
	local.get 43
	local.get 39
	f32x4.pmax
	local.get 12
	v128.bitselect
	local.tee 13
	local.get 45
	local.get 13
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 13
	local.get 13
	f32x4.nearest
	local.tee 13
	local.get 46
	local.get 13
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 37
	f32x4.sub
	local.tee 51
	f64x2.promote_low_f32x4
	local.tee 38
	local.get 50
	f64x2.mul
	local.tee 22
	local.get 36
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 36
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 21
	f64x2.sub
	local.get 22
	local.get 13
	local.get 21
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 21
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 21
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 22
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 38
	f64x2.mul
	local.tee 22
	local.get 34
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 34
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 21
	f64x2.sub
	local.get 22
	local.get 13
	local.get 21
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 21
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 21
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 22
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 38
	f64x2.mul
	local.tee 22
	local.get 40
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 40
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 21
	f64x2.sub
	local.get 22
	local.get 13
	local.get 21
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 21
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 21
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 22
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 38
	f64x2.mul
	local.tee 22
	local.get 30
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 30
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 21
	f64x2.sub
	local.get 22
	local.get 13
	local.get 21
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 21
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 21
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 22
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 38
	f64x2.mul
	local.tee 22
	local.get 35
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 35
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 21
	f64x2.sub
	local.get 22
	local.get 13
	local.get 21
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 21
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 21
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 22
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 38
	f64x2.mul
	local.tee 38
	local.get 41
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 41
	local.get 13
	local.get 38
	f64x2.sub
	local.tee 22
	f64x2.sub
	local.get 38
	local.get 13
	local.get 22
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 38
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 22
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 22
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 38
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 51
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 38
	local.get 50
	f64x2.mul
	local.tee 22
	local.get 36
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 36
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 50
	f64x2.sub
	local.get 22
	local.get 13
	local.get 50
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 22
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 22
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 36
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 38
	f64x2.mul
	local.tee 36
	local.get 34
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 34
	local.get 13
	local.get 36
	f64x2.sub
	local.tee 22
	f64x2.sub
	local.get 36
	local.get 13
	local.get 22
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 34
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 34
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 36
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 38
	f64x2.mul
	local.tee 36
	local.get 40
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 40
	local.get 13
	local.get 36
	f64x2.sub
	local.tee 34
	f64x2.sub
	local.get 36
	local.get 13
	local.get 34
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 34
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 34
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 36
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 38
	f64x2.mul
	local.tee 36
	local.get 30
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 30
	local.get 13
	local.get 36
	f64x2.sub
	local.tee 34
	f64x2.sub
	local.get 36
	local.get 13
	local.get 34
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 34
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 34
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 36
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 38
	f64x2.mul
	local.tee 36
	local.get 35
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 35
	local.get 13
	local.get 36
	f64x2.sub
	local.tee 34
	f64x2.sub
	local.get 36
	local.get 13
	local.get 34
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 34
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 34
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 36
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 38
	f64x2.mul
	local.tee 38
	local.get 41
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 41
	local.get 13
	local.get 38
	f64x2.sub
	local.tee 36
	f64x2.sub
	local.get 38
	local.get 13
	local.get 36
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 38
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 9
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 9
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 38
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 7
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 7
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	local.get 37
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 26
	i32x4.add
	f32x4.mul
	local.get 39
	local.get 43
	f32x4.lt
	v128.bitselect
	local.get 39
	local.get 45
	f32x4.ge
	v128.bitselect
	local.get 27
	v128.bitselect
	local.get 49
	local.get 49
	f32x4.ne
	v128.bitselect
	local.get 23
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 0
	i32.const 64
	i32.add
	local.tee 0
	local.get 6
	i32.ne
	br_if 0
	end_loop
	end_block
	local.get 1
	i32.const 12
	i32.and
	local.tee 0
	i32.eqz
	br_if 1
	block
	i32.const 64
	local.get 0
	i32.const 2
	i32.shl
	local.tee 0
	i32.sub
	local.tee 4
	i32.eqz
	br_if 0
	local.get 3
	i32.const 128
	i32.add
	local.get 0
	i32.add
	i32.const 0
	local.get 4
	memory.fill 0
	end_block
	block
	local.get 0
	i32.eqz
	local.tee 4
	br_if 0
	local.get 3
	i32.const 128
	i32.add
	local.get 8
	local.get 0
	memory.copy 0, 0
	end_block
	local.get 3
	local.get 3
	v128.load 176
	local.tee 20
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 7
	v128.const 0x1p-10, 0x1p-10, 0x1p-10, 0x1p-10
	local.tee 18
	f32x4.gt
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 22
	local.get 7
	f32x4.div
	v128.and
	local.get 20
	f32x4.mul
	local.tee 23
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	local.tee 17
	v128.const 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	local.tee 12
	f32.const 0x1p2
	local.get 2
	local.get 2
	f32.const 0x1p2
	f32.gt
	f32.select
	local.tee 2
	i32.reinterpret_f32
	i32.const -1
	i32.add
	i32.const 8388607
	i32.lt_u
	local.tee 6
	v128.select
	local.tee 19
	v128.const 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127
	local.tee 21
	local.get 12
	v128.const 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
	local.tee 9
	v128.const -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
	local.tee 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	f32.const 0x0p0
	local.get 2
	local.get 6
	f32.select
	f32x4.splat
	local.tee 24
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 17
	local.get 23
	local.get 23
	v128.const 0x1p-126, 0x1p-126, 0x1p-126, 0x1p-126
	local.tee 25
	f32x4.ge
	v128.and
	local.tee 7
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	local.tee 26
	local.get 7
	local.get 22
	f32x4.lt
	v128.bitselect
	local.tee 7
	local.get 7
	v128.const 0, 0, 0, 0
	local.tee 27
	i32x4.eq
	local.tee 28
	v128.bitselect
	v128.const 4913933, 4913933, 4913933, 4913933
	local.tee 29
	i32x4.add
	local.tee 30
	v128.const 8388607, 8388607, 8388607, 8388607
	local.tee 31
	v128.and
	v128.const 1060439283, 1060439283, 1060439283, 1060439283
	local.tee 32
	i32x4.add
	local.tee 7
	v128.const -0x1p0, -0x1p0, -0x1p0, -0x1p0
	local.tee 33
	f32x4.add
	local.get 7
	local.get 22
	f32x4.add
	f32x4.div
	local.tee 34
	local.get 34
	f32x4.mul
	local.tee 35
	f64x2.promote_low_f32x4
	local.tee 36
	v128.const 0x1.bcd67ep-2, 0x1.bcd67ep-2
	local.tee 37
	f64x2.mul
	local.tee 14
	v128.const 0x1.276932p-1, 0x1.276932p-1
	local.tee 13
	f64x2.add
	local.tee 38
	v128.const 0x0p0, 0x0p0
	local.tee 7
	f64x2.gt
	local.get 13
	local.get 38
	local.get 14
	f64x2.sub
	local.tee 15
	f64x2.sub
	local.get 14
	local.get 15
	local.get 38
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 39
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 38
	i64x2.add
	local.get 38
	i64.const -1
	i64.const 0
	local.get 38
	i32.const 1
	i64x2.shl
	local.tee 14
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 14
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 38
	v128.const 1, 1
	local.tee 14
	v128.and
	v128.const 0, 0
	local.tee 15
	i64x2.eq
	local.get 39
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 38
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 38
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 36
	f64x2.mul
	local.tee 40
	v128.const 0x1.ec7126p-1, 0x1.ec7126p-1
	local.tee 38
	f64x2.add
	local.tee 39
	local.get 7
	f64x2.gt
	local.get 38
	local.get 39
	local.get 40
	f64x2.sub
	local.tee 41
	f64x2.sub
	local.get 40
	local.get 39
	local.get 41
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 39
	i64x2.add
	local.get 39
	i64.const -1
	i64.const 0
	local.get 39
	i32.const 1
	i64x2.shl
	local.tee 41
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 41
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 39
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 40
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 39
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 39
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 36
	f64x2.mul
	local.tee 40
	v128.const 0x1.715476p1, 0x1.715476p1
	local.tee 39
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 39
	local.get 36
	local.get 40
	f64x2.sub
	local.tee 41
	f64x2.sub
	local.get 40
	local.get 36
	local.get 41
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 41
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 41
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 40
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 34
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 40
	local.get 30
	i32.const 23
	i32x4.shr_s
	v128.const -127, -127, -127, -127
	local.tee 42
	i32x4.add
	f32x4.convert_i32x4_s
	local.tee 41
	f64x2.promote_low_f32x4
	local.tee 30
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 30
	local.get 36
	local.get 40
	f64x2.sub
	local.tee 43
	f64x2.sub
	local.get 40
	local.get 36
	local.get 43
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 30
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 30
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 40
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 35
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 40
	local.get 37
	f64x2.mul
	local.tee 30
	local.get 13
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 13
	local.get 36
	local.get 30
	f64x2.sub
	local.tee 35
	f64x2.sub
	local.get 30
	local.get 35
	local.get 36
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 30
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 35
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 35
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 30
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 40
	f64x2.mul
	local.tee 30
	local.get 38
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 38
	local.get 36
	local.get 30
	f64x2.sub
	local.tee 35
	f64x2.sub
	local.get 30
	local.get 36
	local.get 35
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 30
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 35
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 35
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 30
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 40
	f64x2.mul
	local.tee 40
	local.get 39
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 39
	local.get 36
	local.get 40
	f64x2.sub
	local.tee 30
	f64x2.sub
	local.get 40
	local.get 36
	local.get 30
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 30
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 30
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 40
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 34
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 34
	local.get 41
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 40
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	local.get 40
	local.get 36
	local.get 34
	f64x2.sub
	local.tee 30
	f64x2.sub
	local.get 34
	local.get 36
	local.get 30
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 34
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 36
	i64x2.add
	local.get 36
	i64.const -1
	i64.const 0
	local.get 36
	i32.const 1
	i64x2.shl
	local.tee 40
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 40
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 36
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 34
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 36
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 36
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	f32x4.mul
	local.tee 44
	v128.const -0x1.f8p6, -0x1.f8p6, -0x1.f8p6, -0x1.f8p6
	local.tee 43
	local.get 44
	f32x4.pmax
	local.get 12
	v128.bitselect
	local.tee 36
	v128.const 0x1p7, 0x1p7, 0x1p7, 0x1p7
	local.tee 45
	local.get 36
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 36
	local.get 36
	f32x4.nearest
	local.tee 36
	v128.const 0x1.fcp6, 0x1.fcp6, 0x1.fcp6, 0x1.fcp6
	local.tee 46
	local.get 36
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 47
	f32x4.sub
	local.tee 48
	f64x2.promote_low_f32x4
	local.tee 49
	v128.const 0x1.43f274p-13, 0x1.43f274p-13
	local.tee 50
	f64x2.mul
	local.tee 40
	v128.const 0x1.5d88f2p-10, 0x1.5d88f2p-10
	local.tee 36
	f64x2.add
	local.tee 34
	local.get 7
	f64x2.gt
	local.get 36
	local.get 34
	local.get 40
	f64x2.sub
	local.tee 30
	f64x2.sub
	local.get 40
	local.get 34
	local.get 30
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 34
	i64x2.add
	local.get 34
	i64.const -1
	i64.const 0
	local.get 34
	i32.const 1
	i64x2.shl
	local.tee 30
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 30
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 34
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 40
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 34
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 34
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 49
	f64x2.mul
	local.tee 30
	v128.const 0x1.3b2a18p-7, 0x1.3b2a18p-7
	local.tee 34
	f64x2.add
	local.tee 40
	local.get 7
	f64x2.gt
	local.get 34
	local.get 40
	local.get 30
	f64x2.sub
	local.tee 35
	f64x2.sub
	local.get 30
	local.get 40
	local.get 35
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 30
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 40
	i64x2.add
	local.get 40
	i64.const -1
	i64.const 0
	local.get 40
	i32.const 1
	i64x2.shl
	local.tee 35
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 35
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 40
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 30
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 40
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 40
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 49
	f64x2.mul
	local.tee 35
	v128.const 0x1.c6b178p-5, 0x1.c6b178p-5
	local.tee 40
	f64x2.add
	local.tee 30
	local.get 7
	f64x2.gt
	local.get 40
	local.get 30
	local.get 35
	f64x2.sub
	local.tee 41
	f64x2.sub
	local.get 35
	local.get 30
	local.get 41
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 35
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 30
	i64x2.add
	local.get 30
	i64.const -1
	i64.const 0
	local.get 30
	i32.const 1
	i64x2.shl
	local.tee 41
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 41
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 30
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 35
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 30
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 30
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 49
	f64x2.mul
	local.tee 41
	v128.const 0x1.ebfbdap-3, 0x1.ebfbdap-3
	local.tee 30
	f64x2.add
	local.tee 35
	local.get 7
	f64x2.gt
	local.get 30
	local.get 35
	local.get 41
	f64x2.sub
	local.tee 51
	f64x2.sub
	local.get 41
	local.get 35
	local.get 51
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 41
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 35
	i64x2.add
	local.get 35
	i64.const -1
	i64.const 0
	local.get 35
	i32.const 1
	i64x2.shl
	local.tee 51
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 51
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 35
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 41
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 35
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 35
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 49
	f64x2.mul
	local.tee 51
	v128.const 0x1.62e43p-1, 0x1.62e43p-1
	local.tee 35
	f64x2.add
	local.tee 41
	local.get 7
	f64x2.gt
	local.get 35
	local.get 41
	local.get 51
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 51
	local.get 41
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 41
	i64x2.add
	local.get 41
	i64.const -1
	i64.const 0
	local.get 41
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 41
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 41
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 41
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 49
	f64x2.mul
	local.tee 51
	v128.const 0x1p0, 0x1p0
	local.tee 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 51
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 48
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 51
	local.get 50
	f64x2.mul
	local.tee 48
	local.get 36
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 36
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 34
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 34
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 40
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 40
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 30
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 30
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 35
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 35
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 51
	local.get 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	local.get 47
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 26
	i32x4.add
	f32x4.mul
	local.get 44
	local.get 43
	f32x4.lt
	v128.bitselect
	local.get 44
	local.get 45
	f32x4.ge
	v128.bitselect
	local.get 28
	v128.bitselect
	local.get 23
	local.get 23
	f32x4.ne
	v128.bitselect
	local.get 20
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 176
	local.get 3
	local.get 3
	v128.load 160
	local.tee 20
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 49
	local.get 18
	f32x4.gt
	local.get 22
	local.get 49
	f32x4.div
	v128.and
	local.get 20
	f32x4.mul
	local.tee 23
	local.get 19
	local.get 21
	local.get 12
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 24
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 17
	local.get 23
	local.get 23
	local.get 25
	f32x4.ge
	v128.and
	local.tee 49
	local.get 26
	local.get 49
	local.get 22
	f32x4.lt
	v128.bitselect
	local.tee 49
	local.get 49
	local.get 27
	i32x4.eq
	local.tee 28
	v128.bitselect
	local.get 29
	i32x4.add
	local.tee 47
	local.get 31
	v128.and
	local.get 32
	i32x4.add
	local.tee 49
	local.get 33
	f32x4.add
	local.get 49
	local.get 22
	f32x4.add
	f32x4.div
	local.tee 51
	local.get 51
	f32x4.mul
	local.tee 52
	f64x2.promote_low_f32x4
	local.tee 44
	local.get 37
	f64x2.mul
	local.tee 48
	local.get 13
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 13
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 53
	local.get 49
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 44
	f64x2.mul
	local.tee 48
	local.get 38
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 38
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 44
	f64x2.mul
	local.tee 44
	local.get 39
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 39
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 44
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 44
	local.get 47
	i32.const 23
	i32x4.shr_s
	local.get 42
	i32x4.add
	f32x4.convert_i32x4_s
	local.tee 47
	f64x2.promote_low_f32x4
	local.tee 48
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 48
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 44
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 52
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 44
	local.get 37
	f64x2.mul
	local.tee 48
	local.get 13
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 13
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 52
	local.get 49
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 44
	f64x2.mul
	local.tee 48
	local.get 38
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 38
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 44
	f64x2.mul
	local.tee 44
	local.get 39
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 39
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 44
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 51
	local.get 47
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 44
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 44
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 44
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 44
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	f32x4.mul
	local.tee 44
	local.get 43
	local.get 44
	f32x4.pmax
	local.get 12
	v128.bitselect
	local.tee 49
	local.get 45
	local.get 49
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 49
	local.get 49
	f32x4.nearest
	local.tee 49
	local.get 46
	local.get 49
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 47
	f32x4.sub
	local.tee 52
	f64x2.promote_low_f32x4
	local.tee 51
	local.get 50
	f64x2.mul
	local.tee 48
	local.get 36
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 36
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 34
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 34
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 40
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 40
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 30
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 30
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 35
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 35
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 51
	local.get 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 52
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 51
	local.get 50
	f64x2.mul
	local.tee 48
	local.get 36
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 36
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 34
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 34
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 40
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 40
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 30
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 30
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 35
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 35
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 51
	local.get 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	local.get 47
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 26
	i32x4.add
	f32x4.mul
	local.get 44
	local.get 43
	f32x4.lt
	v128.bitselect
	local.get 44
	local.get 45
	f32x4.ge
	v128.bitselect
	local.get 28
	v128.bitselect
	local.get 23
	local.get 23
	f32x4.ne
	v128.bitselect
	local.get 20
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 160
	local.get 3
	local.get 3
	v128.load 144
	local.tee 20
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 49
	local.get 18
	f32x4.gt
	local.get 22
	local.get 49
	f32x4.div
	v128.and
	local.get 20
	f32x4.mul
	local.tee 23
	local.get 19
	local.get 21
	local.get 12
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 24
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 17
	local.get 23
	local.get 23
	local.get 25
	f32x4.ge
	v128.and
	local.tee 49
	local.get 26
	local.get 49
	local.get 22
	f32x4.lt
	v128.bitselect
	local.tee 49
	local.get 49
	local.get 27
	i32x4.eq
	local.tee 28
	v128.bitselect
	local.get 29
	i32x4.add
	local.tee 47
	local.get 31
	v128.and
	local.get 32
	i32x4.add
	local.tee 49
	local.get 33
	f32x4.add
	local.get 49
	local.get 22
	f32x4.add
	f32x4.div
	local.tee 51
	local.get 51
	f32x4.mul
	local.tee 52
	f64x2.promote_low_f32x4
	local.tee 44
	local.get 37
	f64x2.mul
	local.tee 48
	local.get 13
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 13
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 53
	local.get 49
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 44
	f64x2.mul
	local.tee 48
	local.get 38
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 38
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 44
	f64x2.mul
	local.tee 44
	local.get 39
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 39
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 44
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 44
	local.get 47
	i32.const 23
	i32x4.shr_s
	local.get 42
	i32x4.add
	f32x4.convert_i32x4_s
	local.tee 47
	f64x2.promote_low_f32x4
	local.tee 48
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 48
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 44
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 52
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 44
	local.get 37
	f64x2.mul
	local.tee 48
	local.get 13
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 13
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 52
	local.get 49
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 44
	f64x2.mul
	local.tee 48
	local.get 38
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 38
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 44
	f64x2.mul
	local.tee 44
	local.get 39
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 39
	local.get 49
	local.get 44
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 44
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 44
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 44
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 51
	local.get 47
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 44
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 44
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 44
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 44
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	f32x4.mul
	local.tee 44
	local.get 43
	local.get 44
	f32x4.pmax
	local.get 12
	v128.bitselect
	local.tee 49
	local.get 45
	local.get 49
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 49
	local.get 49
	f32x4.nearest
	local.tee 49
	local.get 46
	local.get 49
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 47
	f32x4.sub
	local.tee 52
	f64x2.promote_low_f32x4
	local.tee 51
	local.get 50
	f64x2.mul
	local.tee 48
	local.get 36
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 36
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 34
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 34
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 40
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 40
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 30
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 30
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 35
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 35
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 53
	f64x2.sub
	local.get 48
	local.get 49
	local.get 53
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 53
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 53
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.mul
	local.tee 51
	local.get 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 52
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 51
	local.get 50
	f64x2.mul
	local.tee 48
	local.get 36
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 36
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 34
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 34
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 40
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 40
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 30
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 30
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 48
	local.get 35
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 35
	local.get 49
	local.get 48
	f64x2.sub
	local.tee 52
	f64x2.sub
	local.get 48
	local.get 49
	local.get 52
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 48
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 52
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 52
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 48
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	f64x2.mul
	local.tee 51
	local.get 41
	f64x2.add
	local.tee 49
	local.get 7
	f64x2.gt
	local.get 41
	local.get 49
	local.get 51
	f64x2.sub
	local.tee 48
	f64x2.sub
	local.get 51
	local.get 49
	local.get 48
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 51
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 49
	i64x2.add
	local.get 49
	i64.const -1
	i64.const 0
	local.get 49
	i32.const 1
	i64x2.shl
	local.tee 48
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 48
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 49
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 51
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 49
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 49
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	local.get 47
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 26
	i32x4.add
	f32x4.mul
	local.get 44
	local.get 43
	f32x4.lt
	v128.bitselect
	local.get 44
	local.get 45
	f32x4.ge
	v128.bitselect
	local.get 28
	v128.bitselect
	local.get 23
	local.get 23
	f32x4.ne
	v128.bitselect
	local.get 20
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 144
	local.get 3
	local.get 3
	v128.load 128
	local.tee 23
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 49
	local.get 18
	f32x4.gt
	local.get 22
	local.get 49
	f32x4.div
	v128.and
	local.get 23
	f32x4.mul
	local.tee 49
	local.get 19
	local.get 21
	local.get 12
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 24
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 17
	local.get 49
	local.get 49
	local.get 25
	f32x4.ge
	v128.and
	local.tee 51
	local.get 26
	local.get 51
	local.get 22
	f32x4.lt
	v128.bitselect
	local.tee 51
	local.get 51
	local.get 27
	i32x4.eq
	local.tee 25
	v128.bitselect
	local.get 29
	i32x4.add
	local.tee 27
	local.get 31
	v128.and
	local.get 32
	i32x4.add
	local.tee 51
	local.get 33
	f32x4.add
	local.get 51
	local.get 22
	f32x4.add
	f32x4.div
	local.tee 51
	local.get 51
	f32x4.mul
	local.tee 29
	f64x2.promote_low_f32x4
	local.tee 17
	local.get 37
	f64x2.mul
	local.tee 18
	local.get 13
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	local.get 13
	local.get 22
	local.get 18
	f64x2.sub
	local.tee 31
	f64x2.sub
	local.get 18
	local.get 31
	local.get 22
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 18
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 22
	i64x2.add
	local.get 22
	i64.const -1
	i64.const 0
	local.get 22
	i32.const 1
	i64x2.shl
	local.tee 31
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 31
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 22
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 18
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 22
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 22
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 17
	f64x2.mul
	local.tee 18
	local.get 38
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	local.get 38
	local.get 22
	local.get 18
	f64x2.sub
	local.tee 31
	f64x2.sub
	local.get 18
	local.get 22
	local.get 31
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 18
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 22
	i64x2.add
	local.get 22
	i64.const -1
	i64.const 0
	local.get 22
	i32.const 1
	i64x2.shl
	local.tee 31
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 31
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 22
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 18
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 22
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 22
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 17
	f64x2.mul
	local.tee 17
	local.get 39
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	local.get 39
	local.get 22
	local.get 17
	f64x2.sub
	local.tee 18
	f64x2.sub
	local.get 17
	local.get 22
	local.get 18
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 17
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 22
	i64x2.add
	local.get 22
	i64.const -1
	i64.const 0
	local.get 22
	i32.const 1
	i64x2.shl
	local.tee 18
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 18
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 22
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 17
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 22
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 22
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 51
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 17
	local.get 27
	i32.const 23
	i32x4.shr_s
	local.get 42
	i32x4.add
	f32x4.convert_i32x4_s
	local.tee 18
	f64x2.promote_low_f32x4
	local.tee 27
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	local.get 27
	local.get 22
	local.get 17
	f64x2.sub
	local.tee 31
	f64x2.sub
	local.get 17
	local.get 22
	local.get 31
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 17
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 22
	i64x2.add
	local.get 22
	i64.const -1
	i64.const 0
	local.get 22
	i32.const 1
	i64x2.shl
	local.tee 27
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 27
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 22
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 17
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 22
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 22
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 29
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 17
	local.get 37
	f64x2.mul
	local.tee 37
	local.get 13
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	local.get 13
	local.get 22
	local.get 37
	f64x2.sub
	local.tee 27
	f64x2.sub
	local.get 37
	local.get 27
	local.get 22
	f64x2.sub
	f64x2.add
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 22
	i64x2.add
	local.get 22
	i64.const -1
	i64.const 0
	local.get 22
	i32.const 1
	i64x2.shl
	local.tee 37
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 37
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 22
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 13
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 17
	f64x2.mul
	local.tee 22
	local.get 38
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 38
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 37
	f64x2.sub
	local.get 22
	local.get 13
	local.get 37
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 38
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 22
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 22
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 38
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 17
	f64x2.mul
	local.tee 38
	local.get 39
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 39
	local.get 13
	local.get 38
	f64x2.sub
	local.tee 22
	f64x2.sub
	local.get 38
	local.get 13
	local.get 22
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 38
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 39
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 39
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 38
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 51
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	f64x2.mul
	local.tee 38
	local.get 18
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 39
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 39
	local.get 13
	local.get 38
	f64x2.sub
	local.tee 22
	f64x2.sub
	local.get 38
	local.get 13
	local.get 22
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 38
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 39
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 39
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 38
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	f32x4.mul
	local.tee 39
	local.get 43
	local.get 39
	f32x4.pmax
	local.get 12
	v128.bitselect
	local.tee 13
	local.get 45
	local.get 13
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 13
	local.get 13
	f32x4.nearest
	local.tee 13
	local.get 46
	local.get 13
	f32x4.pmin
	local.get 12
	v128.bitselect
	local.tee 37
	f32x4.sub
	local.tee 51
	f64x2.promote_low_f32x4
	local.tee 38
	local.get 50
	f64x2.mul
	local.tee 22
	local.get 36
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 36
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 17
	f64x2.sub
	local.get 22
	local.get 13
	local.get 17
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 17
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 17
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 22
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 38
	f64x2.mul
	local.tee 22
	local.get 34
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 34
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 17
	f64x2.sub
	local.get 22
	local.get 13
	local.get 17
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 17
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 17
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 22
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 38
	f64x2.mul
	local.tee 22
	local.get 40
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 40
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 17
	f64x2.sub
	local.get 22
	local.get 13
	local.get 17
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 17
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 17
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 22
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 38
	f64x2.mul
	local.tee 22
	local.get 30
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 30
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 17
	f64x2.sub
	local.get 22
	local.get 13
	local.get 17
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 17
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 17
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 22
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 38
	f64x2.mul
	local.tee 22
	local.get 35
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 35
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 17
	f64x2.sub
	local.get 22
	local.get 13
	local.get 17
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 22
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 17
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 17
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 22
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f64.promote_f32
	f64x2.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f64.promote_f32
	f64x2.replace_lane 1
	local.get 38
	f64x2.mul
	local.tee 38
	local.get 41
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 41
	local.get 13
	local.get 38
	f64x2.sub
	local.tee 22
	f64x2.sub
	local.get 38
	local.get 13
	local.get 22
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 38
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 22
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 22
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 38
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 9
	local.get 11
	local.get 51
	local.get 7
	i8x16.shuffle 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 0, 1, 2, 3
	f64x2.promote_low_f32x4
	local.tee 38
	local.get 50
	f64x2.mul
	local.tee 22
	local.get 36
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 36
	local.get 13
	local.get 22
	f64x2.sub
	local.tee 50
	f64x2.sub
	local.get 22
	local.get 13
	local.get 50
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 22
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 22
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 36
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 38
	f64x2.mul
	local.tee 36
	local.get 34
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 34
	local.get 13
	local.get 36
	f64x2.sub
	local.tee 22
	f64x2.sub
	local.get 36
	local.get 13
	local.get 22
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 34
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 34
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 36
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 38
	f64x2.mul
	local.tee 36
	local.get 40
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 40
	local.get 13
	local.get 36
	f64x2.sub
	local.tee 34
	f64x2.sub
	local.get 36
	local.get 13
	local.get 34
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 34
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 34
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 36
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 38
	f64x2.mul
	local.tee 36
	local.get 30
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 30
	local.get 13
	local.get 36
	f64x2.sub
	local.tee 34
	f64x2.sub
	local.get 36
	local.get 13
	local.get 34
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 34
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 34
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 36
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 38
	f64x2.mul
	local.tee 36
	local.get 35
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 35
	local.get 13
	local.get 36
	f64x2.sub
	local.tee 34
	f64x2.sub
	local.get 36
	local.get 13
	local.get 34
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 36
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 34
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 34
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 36
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 13
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.splat
	local.get 13
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 1
	f64x2.promote_low_f32x4
	local.get 38
	f64x2.mul
	local.tee 38
	local.get 41
	f64x2.add
	local.tee 13
	local.get 7
	f64x2.gt
	local.get 41
	local.get 13
	local.get 38
	f64x2.sub
	local.tee 36
	f64x2.sub
	local.get 38
	local.get 13
	local.get 36
	f64x2.sub
	f64x2.sub
	f64x2.add
	local.tee 38
	local.get 7
	f64x2.gt
	v128.xor
	v128.not
	v128.bitselect
	local.get 13
	i64x2.add
	local.get 13
	i64.const -1
	i64.const 0
	local.get 13
	i32.const 1
	i64x2.shl
	local.tee 9
	i64x2.extract_lane 0
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.splat
	i64.const -1
	i64.const 0
	local.get 9
	i64x2.extract_lane 1
	i64.const -9007199254740992
	i64.lt_u
	i64.select
	i64x2.replace_lane 1
	local.get 13
	local.get 14
	v128.and
	local.get 15
	i64x2.eq
	local.get 38
	local.get 7
	f64x2.ne
	v128.and
	v128.and
	v128.bitselect
	local.tee 7
	f64x2.extract_lane 0
	f32.demote_f64
	f32x4.replace_lane 2
	local.get 7
	f64x2.extract_lane 1
	f32.demote_f64
	f32x4.replace_lane 3
	local.get 37
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 26
	i32x4.add
	f32x4.mul
	local.get 39
	local.get 43
	f32x4.lt
	v128.bitselect
	local.get 39
	local.get 45
	f32x4.ge
	v128.bitselect
	local.get 25
	v128.bitselect
	local.get 49
	local.get 49
	f32x4.ne
	v128.bitselect
	local.get 23
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 128
	local.get 4
	br_if 1
	local.get 8
	local.get 3
	i32.const 128
	i32.add
	local.get 0
	memory.copy 0, 0
	br 1
	end_block
	block
	local.get 4
	i32.eqz
	br_if 0
	loop
	local.get 0
	local.get 0
	v128.load 0:p2align=0
	local.tee 12
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 11
	v128.const 0x1p-10, 0x1p-10, 0x1p-10, 0x1p-10
	local.tee 7
	f32x4.gt
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 9
	local.get 11
	f32x4.div
	v128.and
	local.get 12
	f32x4.mul
	local.tee 11
	local.get 11
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 14
	f32x4.gt
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	local.tee 15
	v128.and
	local.get 11
	local.get 11
	f32x4.ne
	v128.bitselect
	local.get 12
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 0
	i32.const 48
	i32.add
	local.tee 4
	local.get 4
	v128.load 0:p2align=0
	local.tee 12
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 11
	local.get 7
	f32x4.gt
	local.get 9
	local.get 11
	f32x4.div
	v128.and
	local.get 12
	f32x4.mul
	local.tee 11
	local.get 11
	local.get 14
	f32x4.gt
	local.get 15
	v128.and
	local.get 11
	local.get 11
	f32x4.ne
	v128.bitselect
	local.get 12
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 0
	i32.const 32
	i32.add
	local.tee 4
	local.get 4
	v128.load 0:p2align=0
	local.tee 12
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 11
	local.get 7
	f32x4.gt
	local.get 9
	local.get 11
	f32x4.div
	v128.and
	local.get 12
	f32x4.mul
	local.tee 11
	local.get 11
	local.get 14
	f32x4.gt
	local.get 15
	v128.and
	local.get 11
	local.get 11
	f32x4.ne
	v128.bitselect
	local.get 12
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 0
	i32.const 16
	i32.add
	local.tee 4
	local.get 4
	v128.load 0:p2align=0
	local.tee 11
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 12
	local.get 7
	f32x4.gt
	local.get 9
	local.get 12
	f32x4.div
	v128.and
	local.get 11
	f32x4.mul
	local.tee 7
	local.get 7
	local.get 14
	f32x4.gt
	local.get 15
	v128.and
	local.get 7
	local.get 7
	f32x4.ne
	v128.bitselect
	local.get 11
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=0
	local.get 0
	i32.const 64
	i32.add
	local.tee 0
	local.get 6
	i32.ne
	br_if 0
	end_loop
	end_block
	local.get 1
	i32.const 12
	i32.and
	local.tee 0
	i32.eqz
	br_if 0
	block
	i32.const 64
	local.get 0
	i32.const 2
	i32.shl
	local.tee 0
	i32.sub
	local.tee 4
	i32.eqz
	br_if 0
	local.get 3
	i32.const 128
	i32.add
	local.get 0
	i32.add
	i32.const 0
	local.get 4
	memory.fill 0
	end_block
	block
	local.get 0
	i32.eqz
	local.tee 4
	br_if 0
	local.get 3
	i32.const 128
	i32.add
	local.get 8
	local.get 0
	memory.copy 0, 0
	end_block
	local.get 3
	local.get 3
	v128.load 176
	local.tee 12
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 11
	v128.const 0x1p-10, 0x1p-10, 0x1p-10, 0x1p-10
	local.tee 7
	f32x4.gt
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 9
	local.get 11
	f32x4.div
	v128.and
	local.get 12
	f32x4.mul
	local.tee 11
	local.get 11
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 14
	f32x4.gt
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	local.tee 15
	v128.and
	local.get 11
	local.get 11
	f32x4.ne
	v128.bitselect
	local.get 12
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 176
	local.get 3
	local.get 3
	v128.load 160
	local.tee 12
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 11
	local.get 7
	f32x4.gt
	local.get 9
	local.get 11
	f32x4.div
	v128.and
	local.get 12
	f32x4.mul
	local.tee 11
	local.get 11
	local.get 14
	f32x4.gt
	local.get 15
	v128.and
	local.get 11
	local.get 11
	f32x4.ne
	v128.bitselect
	local.get 12
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 160
	local.get 3
	local.get 3
	v128.load 144
	local.tee 12
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 11
	local.get 7
	f32x4.gt
	local.get 9
	local.get 11
	f32x4.div
	v128.and
	local.get 12
	f32x4.mul
	local.tee 11
	local.get 11
	local.get 14
	f32x4.gt
	local.get 15
	v128.and
	local.get 11
	local.get 11
	f32x4.ne
	v128.bitselect
	local.get 12
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 144
	local.get 3
	local.get 3
	v128.load 128
	local.tee 11
	local.get 7
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 12
	local.get 7
	f32x4.gt
	local.get 9
	local.get 12
	f32x4.div
	v128.and
	local.get 11
	f32x4.mul
	local.tee 7
	local.get 7
	local.get 14
	f32x4.gt
	local.get 15
	v128.and
	local.get 7
	local.get 7
	f32x4.ne
	v128.bitselect
	local.get 11
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 128
	local.get 4
	br_if 0
	local.get 8
	local.get 3
	i32.const 128
	i32.add
	local.get 0
	memory.copy 0, 0
	end_block
	local.get 3
	i32.const 192
	i32.add
	global.set __stack_pointer
	end_function
