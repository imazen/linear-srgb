.section .text.stub_unpremultiply_linear_to_gamma_rgba_slice,"",@
	.globl	stub_unpremultiply_linear_to_gamma_rgba_slice
.type	stub_unpremultiply_linear_to_gamma_rgba_slice,@function
stub_unpremultiply_linear_to_gamma_rgba_slice:
	.functype	stub_unpremultiply_linear_to_gamma_rgba_slice (i32, i32, f32) -> ()
	.local  	i32, v128, v128, i32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128
	global.get __stack_pointer
	i32.const 32
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
	local.get 1
	i32.const 2
	i32.shl
	i32.const 2147483632
	i32.and
	local.tee 1
	i32.eqz
	br_if 1
	loop
	local.get 3
	local.get 0
	v128.load 0:p2align=2
	local.tee 4
	local.get 4
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 5
	v128.const 0x1p-10, 0x1p-10, 0x1p-10, 0x1p-10
	f32x4.gt
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.get 5
	f32x4.div
	v128.and
	local.get 4
	f32x4.mul
	v128.store 16
	local.get 3
	local.get 3
	i32.const 16
	i32.add
	local.get 2
	call linear_srgb::tf::math::gamma_outside_fit::<4>
	local.get 0
	local.get 3
	v128.load 0
	local.get 4
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=2
	local.get 0
	i32.const 16
	i32.add
	local.set 0
	local.get 1
	i32.const -16
	i32.add
	local.tee 1
	br_if 0
	br 2
	end_loop
	end_block
	local.get 1
	i32.const 2
	i32.shl
	i32.const 2147483632
	i32.and
	local.set 1
	block
	local.get 2
	f32.const 0x0p0
	f32.eq
	br_if 0
	local.get 1
	i32.eqz
	br_if 1
	f32.const 0x0p0
	f32.const 0x1p2
	local.get 2
	local.get 2
	f32.const 0x1p2
	f32.gt
	f32.select
	local.tee 2
	local.get 2
	i32.reinterpret_f32
	i32.const -1
	i32.add
	i32.const 8388607
	i32.lt_u
	local.tee 6
	f32.select
	f32x4.splat
	local.set 7
	f32.const 0x1p0
	f32.const 0x0p0
	local.get 6
	f32.select
	f32x4.splat
	local.set 8
	loop
	local.get 0
	local.get 0
	v128.load 0:p2align=2
	local.tee 9
	local.get 4
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 4
	v128.const 0x1p-10, 0x1p-10, 0x1p-10, 0x1p-10
	f32x4.gt
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 10
	local.get 4
	f32x4.div
	v128.and
	local.get 9
	f32x4.mul
	local.tee 5
	local.get 8
	v128.const 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127
	v128.const 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	local.tee 11
	v128.const 0x1p7, 0x1p7, 0x1p7, 0x1p7
	local.tee 12
	local.get 7
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	local.get 5
	local.get 5
	v128.const 0x1p-126, 0x1p-126, 0x1p-126, 0x1p-126
	f32x4.ge
	v128.and
	local.tee 4
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	local.tee 13
	local.get 4
	local.get 10
	f32x4.lt
	v128.bitselect
	local.tee 4
	local.get 4
	v128.const 0, 0, 0, 0
	i32x4.eq
	local.tee 14
	v128.bitselect
	v128.const 4913933, 4913933, 4913933, 4913933
	i32x4.add
	local.tee 15
	v128.const 8388607, 8388607, 8388607, 8388607
	v128.and
	v128.const 1060439283, 1060439283, 1060439283, 1060439283
	i32x4.add
	local.tee 4
	v128.const -0x1p0, -0x1p0, -0x1p0, -0x1p0
	f32x4.add
	local.get 4
	local.get 10
	f32x4.add
	f32x4.div
	local.tee 4
	local.get 4
	local.get 4
	f32x4.mul
	local.tee 4
	local.get 4
	local.get 4
	v128.const 0x1.bcd67ep-2, 0x1.bcd67ep-2, 0x1.bcd67ep-2, 0x1.bcd67ep-2
	f32x4.mul
	v128.const 0x1.276932p-1, 0x1.276932p-1, 0x1.276932p-1, 0x1.276932p-1
	f32x4.add
	f32x4.mul
	v128.const 0x1.ec7126p-1, 0x1.ec7126p-1, 0x1.ec7126p-1, 0x1.ec7126p-1
	f32x4.add
	f32x4.mul
	v128.const 0x1.715476p1, 0x1.715476p1, 0x1.715476p1, 0x1.715476p1
	f32x4.add
	f32x4.mul
	local.get 15
	i32.const 23
	i32x4.shr_s
	v128.const -127, -127, -127, -127
	i32x4.add
	f32x4.convert_i32x4_s
	f32x4.add
	f32x4.mul
	local.tee 15
	local.get 15
	v128.const -1023672320, -1023672320, -1023672320, -1023672320
	local.get 15
	v128.const -0x1.f8p6, -0x1.f8p6, -0x1.f8p6, -0x1.f8p6
	local.tee 16
	f32x4.gt
	v128.bitselect
	local.get 11
	v128.bitselect
	f32x4.pmin
	local.tee 4
	v128.const 0x1.fcp6, 0x1.fcp6, 0x1.fcp6, 0x1.fcp6
	local.get 4
	f32x4.nearest
	f32x4.pmin
	local.tee 11
	f32x4.sub
	local.tee 4
	local.get 4
	local.get 4
	local.get 4
	local.get 4
	local.get 4
	v128.const 0x1.43f274p-13, 0x1.43f274p-13, 0x1.43f274p-13, 0x1.43f274p-13
	f32x4.mul
	v128.const 0x1.5d88f2p-10, 0x1.5d88f2p-10, 0x1.5d88f2p-10, 0x1.5d88f2p-10
	f32x4.add
	f32x4.mul
	v128.const 0x1.3b2a18p-7, 0x1.3b2a18p-7, 0x1.3b2a18p-7, 0x1.3b2a18p-7
	f32x4.add
	f32x4.mul
	v128.const 0x1.c6b178p-5, 0x1.c6b178p-5, 0x1.c6b178p-5, 0x1.c6b178p-5
	f32x4.add
	f32x4.mul
	v128.const 0x1.ebfbdap-3, 0x1.ebfbdap-3, 0x1.ebfbdap-3, 0x1.ebfbdap-3
	f32x4.add
	f32x4.mul
	v128.const 0x1.62e43p-1, 0x1.62e43p-1, 0x1.62e43p-1, 0x1.62e43p-1
	f32x4.add
	f32x4.mul
	local.get 10
	f32x4.add
	local.get 11
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 13
	i32x4.add
	f32x4.mul
	local.get 15
	local.get 16
	f32x4.lt
	v128.bitselect
	local.get 15
	local.get 12
	f32x4.ge
	v128.bitselect
	local.get 14
	v128.bitselect
	local.get 5
	local.get 5
	f32x4.ne
	v128.bitselect
	local.get 9
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=2
	local.get 0
	i32.const 16
	i32.add
	local.set 0
	local.get 1
	i32.const -16
	i32.add
	local.tee 1
	br_if 0
	br 2
	end_loop
	end_block
	local.get 1
	i32.eqz
	br_if 0
	loop
	local.get 0
	local.get 0
	v128.load 0:p2align=2
	local.tee 5
	local.get 4
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 4
	v128.const 0x1p-10, 0x1p-10, 0x1p-10, 0x1p-10
	f32x4.gt
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.get 4
	f32x4.div
	v128.and
	local.get 5
	f32x4.mul
	local.tee 4
	local.get 4
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	f32x4.gt
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	v128.and
	local.get 4
	local.get 4
	f32x4.ne
	v128.bitselect
	local.get 5
	i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31
	v128.store 0:p2align=2
	local.get 0
	i32.const 16
	i32.add
	local.set 0
	local.get 1
	i32.const -16
	i32.add
	local.tee 1
	br_if 0
	end_loop
	end_block
	local.get 3
	i32.const 32
	i32.add
	global.set __stack_pointer
	end_function
