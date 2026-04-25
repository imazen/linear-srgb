.section .text.stub_unpremultiply_linear_to_gamma_rgba_slice,"",@
	.globl	stub_unpremultiply_linear_to_gamma_rgba_slice
.type	stub_unpremultiply_linear_to_gamma_rgba_slice,@function
stub_unpremultiply_linear_to_gamma_rgba_slice:
	.functype	stub_unpremultiply_linear_to_gamma_rgba_slice (i32, i32, f32) -> ()
	.local  	v128, i32, v128, f32, v128, v128, v128, v128, v128
	block 
	local.get 1
	i32.const 2
	i32.shl 
	i32.const 2147483632
	i32.and 
	local.tee 1
	i32.eqz
	br_if 0
	f32.const 0x1p0
	local.get 2
	f32.div 
	f32x4.splat
	local.set 3
	loop 
	block 
	block 
	local.get 0
	i32.const 12
	i32.add 
	local.tee 4
	f32.load 0
	local.tee 2
	f32.const 0x1p-10
	f32.gt 
	br_if 0
	local.get 0
	i64.const 0
	i64.store 0:p2align=2
	local.get 0
	i32.const 8
	i32.add 
	i32.const 0
	i32.store 0
	br 1
	end_block
	local.get 0
	v128.const 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127
	v128.const 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	local.tee 5
	local.get 3
	v128.const 0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 192, 127
	v128.const 0, 0, 128, 255, 0, 0, 128, 255, 0, 0, 128, 255, 0, 0, 128, 255
	f32.const 0x1p0
	local.get 2
	f32.div 
	local.tee 6
	local.get 0
	f32.load 0
	f32.mul 
	f32x4.splat
	local.get 6
	local.get 0
	i32.const 4
	i32.add 
	f32.load 0
	f32.mul 
	f32x4.replace_lane 1
	local.get 6
	local.get 0
	i32.const 8
	i32.add 
	f32.load 0
	f32.mul 
	f32x4.replace_lane 2
	i32.const 1065353216
	i32x4.replace_lane 3
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 7
	f32x4.max
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 8
	f32x4.min
	local.tee 9
	v128.const 4913933, 4913933, 4913933, 4913933
	i32x4.add
	local.tee 10
	v128.const 8388607, 8388607, 8388607, 8388607
	v128.and
	v128.const 1060439283, 1060439283, 1060439283, 1060439283
	i32x4.add
	local.tee 11
	v128.const -0x1p0, -0x1p0, -0x1p0, -0x1p0
	f32x4.add
	local.get 11
	local.get 8
	f32x4.add
	f32x4.div
	local.tee 11
	local.get 11
	local.get 11
	f32x4.mul
	local.tee 11
	local.get 11
	local.get 11
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
	local.get 10
	i32.const 23
	i32x4.shr_s
	v128.const -127, -127, -127, -127
	i32x4.add
	f32x4.convert_i32x4_s
	f32x4.add
	local.get 9
	local.get 7
	f32x4.eq
	v128.bitselect
	local.get 5
	v128.bitselect
	f32x4.mul
	local.tee 5
	v128.const -0x1.f8p6, -0x1.f8p6, -0x1.f8p6, -0x1.f8p6
	local.tee 7
	f32x4.max
	v128.const 0x1p7, 0x1p7, 0x1p7, 0x1p7
	local.tee 9
	f32x4.min
	local.tee 11
	local.get 11
	f32x4.nearest
	v128.const 0x1.fcp6, 0x1.fcp6, 0x1.fcp6, 0x1.fcp6
	f32x4.min
	local.tee 10
	f32x4.sub
	local.tee 11
	local.get 11
	local.get 11
	local.get 11
	local.get 11
	local.get 11
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
	local.get 8
	f32x4.add
	local.get 10
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	i32x4.add
	f32x4.mul
	local.get 5
	local.get 7
	f32x4.lt
	v128.bitselect
	local.get 5
	local.get 9
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	local.get 2
	f32.store 0
	end_block
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
	end_function
