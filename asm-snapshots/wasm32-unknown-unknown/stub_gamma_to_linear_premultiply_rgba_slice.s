.section .text.stub_gamma_to_linear_premultiply_rgba_slice,"",@
	.globl	stub_gamma_to_linear_premultiply_rgba_slice
.type	stub_gamma_to_linear_premultiply_rgba_slice,@function
stub_gamma_to_linear_premultiply_rgba_slice:
	.functype	stub_gamma_to_linear_premultiply_rgba_slice (i32, i32, f32) -> ()
	.local  	i32, v128, i32, i32, v128, i32, f32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, i32, i32, f32, v128, i32, f32, f32
	block 
	local.get 1
	i32.const 2
	i32.shl 
	i32.const 2147483584
	i32.and 
	local.tee 3
	i32.eqz
	br_if 0
	local.get 2
	f32x4.splat
	local.set 4
	i32.const 0
	local.set 5
	loop 
	local.get 0
	local.get 5
	i32.add 
	local.tee 6
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 7
	local.get 6
	i32.const 12
	i32.add 
	local.tee 8
	f32.load 0
	local.tee 9
	f32x4.replace_lane 0
	local.get 9
	f32x4.replace_lane 1
	local.get 9
	f32x4.replace_lane 2
	v128.const 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127
	local.tee 10
	v128.const 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	local.tee 11
	local.get 4
	v128.const 0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 192, 127
	local.tee 12
	v128.const 0, 0, 128, 255, 0, 0, 128, 255, 0, 0, 128, 255, 0, 0, 128, 255
	local.tee 13
	local.get 6
	v128.load 0:p2align=2
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 14
	f32x4.max
	local.get 7
	f32x4.min
	local.tee 15
	v128.const 4913933, 4913933, 4913933, 4913933
	local.tee 16
	i32x4.add
	local.tee 17
	v128.const 8388607, 8388607, 8388607, 8388607
	local.tee 18
	v128.and
	v128.const 1060439283, 1060439283, 1060439283, 1060439283
	local.tee 19
	i32x4.add
	local.tee 20
	v128.const -0x1p0, -0x1p0, -0x1p0, -0x1p0
	local.tee 21
	f32x4.add
	local.get 20
	local.get 7
	f32x4.add
	f32x4.div
	local.tee 20
	local.get 20
	local.get 20
	f32x4.mul
	local.tee 20
	local.get 20
	local.get 20
	v128.const 0x1.bcd67ep-2, 0x1.bcd67ep-2, 0x1.bcd67ep-2, 0x1.bcd67ep-2
	local.tee 22
	f32x4.mul
	v128.const 0x1.276932p-1, 0x1.276932p-1, 0x1.276932p-1, 0x1.276932p-1
	local.tee 23
	f32x4.add
	f32x4.mul
	v128.const 0x1.ec7126p-1, 0x1.ec7126p-1, 0x1.ec7126p-1, 0x1.ec7126p-1
	local.tee 24
	f32x4.add
	f32x4.mul
	v128.const 0x1.715476p1, 0x1.715476p1, 0x1.715476p1, 0x1.715476p1
	local.tee 25
	f32x4.add
	f32x4.mul
	local.get 17
	i32.const 23
	i32x4.shr_s
	v128.const -127, -127, -127, -127
	local.tee 26
	i32x4.add
	f32x4.convert_i32x4_s
	f32x4.add
	local.get 15
	local.get 14
	f32x4.eq
	v128.bitselect
	local.get 11
	v128.bitselect
	f32x4.mul
	local.tee 27
	v128.const -0x1.f8p6, -0x1.f8p6, -0x1.f8p6, -0x1.f8p6
	local.tee 20
	f32x4.max
	v128.const 0x1p7, 0x1p7, 0x1p7, 0x1p7
	local.tee 15
	f32x4.min
	local.tee 17
	local.get 17
	f32x4.nearest
	v128.const 0x1.fcp6, 0x1.fcp6, 0x1.fcp6, 0x1.fcp6
	local.tee 28
	f32x4.min
	local.tee 29
	f32x4.sub
	local.tee 17
	local.get 17
	local.get 17
	local.get 17
	local.get 17
	local.get 17
	v128.const 0x1.43f274p-13, 0x1.43f274p-13, 0x1.43f274p-13, 0x1.43f274p-13
	local.tee 30
	f32x4.mul
	v128.const 0x1.5d88f2p-10, 0x1.5d88f2p-10, 0x1.5d88f2p-10, 0x1.5d88f2p-10
	local.tee 31
	f32x4.add
	f32x4.mul
	v128.const 0x1.3b2a18p-7, 0x1.3b2a18p-7, 0x1.3b2a18p-7, 0x1.3b2a18p-7
	local.tee 32
	f32x4.add
	f32x4.mul
	v128.const 0x1.c6b178p-5, 0x1.c6b178p-5, 0x1.c6b178p-5, 0x1.c6b178p-5
	local.tee 33
	f32x4.add
	f32x4.mul
	v128.const 0x1.ebfbdap-3, 0x1.ebfbdap-3, 0x1.ebfbdap-3, 0x1.ebfbdap-3
	local.tee 34
	f32x4.add
	f32x4.mul
	v128.const 0x1.62e43p-1, 0x1.62e43p-1, 0x1.62e43p-1, 0x1.62e43p-1
	local.tee 35
	f32x4.add
	f32x4.mul
	local.get 7
	f32x4.add
	local.get 29
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	local.tee 29
	i32x4.add
	f32x4.mul
	local.get 27
	local.get 20
	f32x4.lt
	v128.bitselect
	local.get 27
	local.get 15
	f32x4.ge
	v128.bitselect
	f32x4.mul
	v128.store 0:p2align=2
	local.get 6
	i32.const 48
	i32.add 
	local.tee 36
	local.get 7
	local.get 6
	i32.const 60
	i32.add 
	local.tee 37
	f32.load 0
	local.tee 38
	f32x4.replace_lane 0
	local.get 38
	f32x4.replace_lane 1
	local.get 38
	f32x4.replace_lane 2
	local.get 10
	local.get 11
	local.get 4
	local.get 12
	local.get 13
	local.get 36
	v128.load 0:p2align=2
	local.get 14
	f32x4.max
	local.get 7
	f32x4.min
	local.tee 27
	local.get 16
	i32x4.add
	local.tee 39
	local.get 18
	v128.and
	local.get 19
	i32x4.add
	local.tee 17
	local.get 21
	f32x4.add
	local.get 17
	local.get 7
	f32x4.add
	f32x4.div
	local.tee 17
	local.get 17
	local.get 17
	f32x4.mul
	local.tee 17
	local.get 17
	local.get 17
	local.get 22
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
	local.get 39
	i32.const 23
	i32x4.shr_s
	local.get 26
	i32x4.add
	f32x4.convert_i32x4_s
	f32x4.add
	local.get 27
	local.get 14
	f32x4.eq
	v128.bitselect
	local.get 11
	v128.bitselect
	f32x4.mul
	local.tee 27
	local.get 20
	f32x4.max
	local.get 15
	f32x4.min
	local.tee 17
	local.get 17
	f32x4.nearest
	local.get 28
	f32x4.min
	local.tee 39
	f32x4.sub
	local.tee 17
	local.get 17
	local.get 17
	local.get 17
	local.get 17
	local.get 17
	local.get 30
	f32x4.mul
	local.get 31
	f32x4.add
	f32x4.mul
	local.get 32
	f32x4.add
	f32x4.mul
	local.get 33
	f32x4.add
	f32x4.mul
	local.get 34
	f32x4.add
	f32x4.mul
	local.get 35
	f32x4.add
	f32x4.mul
	local.get 7
	f32x4.add
	local.get 39
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 29
	i32x4.add
	f32x4.mul
	local.get 27
	local.get 20
	f32x4.lt
	v128.bitselect
	local.get 27
	local.get 15
	f32x4.ge
	v128.bitselect
	f32x4.mul
	v128.store 0:p2align=2
	local.get 6
	i32.const 32
	i32.add 
	local.tee 36
	local.get 7
	local.get 6
	i32.const 44
	i32.add 
	local.tee 40
	f32.load 0
	local.tee 41
	f32x4.replace_lane 0
	local.get 41
	f32x4.replace_lane 1
	local.get 41
	f32x4.replace_lane 2
	local.get 10
	local.get 11
	local.get 4
	local.get 12
	local.get 13
	local.get 36
	v128.load 0:p2align=2
	local.get 14
	f32x4.max
	local.get 7
	f32x4.min
	local.tee 27
	local.get 16
	i32x4.add
	local.tee 39
	local.get 18
	v128.and
	local.get 19
	i32x4.add
	local.tee 17
	local.get 21
	f32x4.add
	local.get 17
	local.get 7
	f32x4.add
	f32x4.div
	local.tee 17
	local.get 17
	local.get 17
	f32x4.mul
	local.tee 17
	local.get 17
	local.get 17
	local.get 22
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
	local.get 39
	i32.const 23
	i32x4.shr_s
	local.get 26
	i32x4.add
	f32x4.convert_i32x4_s
	f32x4.add
	local.get 27
	local.get 14
	f32x4.eq
	v128.bitselect
	local.get 11
	v128.bitselect
	f32x4.mul
	local.tee 27
	local.get 20
	f32x4.max
	local.get 15
	f32x4.min
	local.tee 17
	local.get 17
	f32x4.nearest
	local.get 28
	f32x4.min
	local.tee 39
	f32x4.sub
	local.tee 17
	local.get 17
	local.get 17
	local.get 17
	local.get 17
	local.get 17
	local.get 30
	f32x4.mul
	local.get 31
	f32x4.add
	f32x4.mul
	local.get 32
	f32x4.add
	f32x4.mul
	local.get 33
	f32x4.add
	f32x4.mul
	local.get 34
	f32x4.add
	f32x4.mul
	local.get 35
	f32x4.add
	f32x4.mul
	local.get 7
	f32x4.add
	local.get 39
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 29
	i32x4.add
	f32x4.mul
	local.get 27
	local.get 20
	f32x4.lt
	v128.bitselect
	local.get 27
	local.get 15
	f32x4.ge
	v128.bitselect
	f32x4.mul
	v128.store 0:p2align=2
	local.get 6
	i32.const 16
	i32.add 
	local.tee 36
	local.get 7
	local.get 6
	i32.const 28
	i32.add 
	local.tee 6
	f32.load 0
	local.tee 42
	f32x4.replace_lane 0
	local.get 42
	f32x4.replace_lane 1
	local.get 42
	f32x4.replace_lane 2
	local.get 10
	local.get 11
	local.get 4
	local.get 12
	local.get 13
	local.get 36
	v128.load 0:p2align=2
	local.get 14
	f32x4.max
	local.get 7
	f32x4.min
	local.tee 27
	local.get 16
	i32x4.add
	local.tee 16
	local.get 18
	v128.and
	local.get 19
	i32x4.add
	local.tee 17
	local.get 21
	f32x4.add
	local.get 17
	local.get 7
	f32x4.add
	f32x4.div
	local.tee 17
	local.get 17
	local.get 17
	f32x4.mul
	local.tee 17
	local.get 17
	local.get 17
	local.get 22
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
	local.get 16
	i32.const 23
	i32x4.shr_s
	local.get 26
	i32x4.add
	f32x4.convert_i32x4_s
	f32x4.add
	local.get 27
	local.get 14
	f32x4.eq
	v128.bitselect
	local.get 11
	v128.bitselect
	f32x4.mul
	local.tee 17
	local.get 20
	f32x4.max
	local.get 15
	f32x4.min
	local.tee 14
	local.get 14
	f32x4.nearest
	local.get 28
	f32x4.min
	local.tee 12
	f32x4.sub
	local.tee 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 30
	f32x4.mul
	local.get 31
	f32x4.add
	f32x4.mul
	local.get 32
	f32x4.add
	f32x4.mul
	local.get 33
	f32x4.add
	f32x4.mul
	local.get 34
	f32x4.add
	f32x4.mul
	local.get 35
	f32x4.add
	f32x4.mul
	local.get 7
	f32x4.add
	local.get 12
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 29
	i32x4.add
	f32x4.mul
	local.get 17
	local.get 20
	f32x4.lt
	v128.bitselect
	local.get 17
	local.get 15
	f32x4.ge
	v128.bitselect
	f32x4.mul
	v128.store 0:p2align=2
	local.get 8
	local.get 9
	f32.store 0
	local.get 37
	local.get 38
	f32.store 0
	local.get 40
	local.get 41
	f32.store 0
	local.get 6
	local.get 42
	f32.store 0
	local.get 3
	local.get 5
	i32.const 64
	i32.add 
	local.tee 5
	i32.ne 
	br_if 0
	end_loop
	end_block
	block 
	local.get 1
	i32.const 12
	i32.and 
	local.tee 5
	i32.eqz
	br_if 0
	local.get 0
	local.get 1
	i32.const 536870896
	i32.and 
	i32.const 2
	i32.shl 
	i32.add 
	local.set 6
	i32.const 0
	local.get 5
	i32.sub 
	local.set 5
	loop 
	f32.const 0x0p0
	local.set 38
	local.get 6
	i32.const 12
	i32.add 
	f32.load 0
	local.set 9
	f32.const 0x0p0
	local.set 41
	block 
	local.get 6
	f32.load 0
	local.tee 42
	f32.const 0x0p0
	f32.le 
	br_if 0
	f32.const 0x1p0
	local.set 41
	local.get 42
	f32.const 0x1p0
	f32.ge 
	br_if 0
	local.get 42
	local.get 2
	call powf
	local.set 41
	end_block
	local.get 6
	local.get 9
	local.get 41
	f32.mul 
	f32.store 0
	block 
	local.get 6
	i32.const 4
	i32.add 
	local.tee 8
	f32.load 0
	local.tee 41
	f32.const 0x0p0
	f32.le 
	br_if 0
	f32.const 0x1p0
	local.set 38
	local.get 41
	f32.const 0x1p0
	f32.ge 
	br_if 0
	local.get 41
	local.get 2
	call powf
	local.set 38
	end_block
	local.get 8
	local.get 9
	local.get 38
	f32.mul 
	f32.store 0
	f32.const 0x0p0
	local.set 38
	block 
	local.get 6
	i32.const 8
	i32.add 
	local.tee 8
	f32.load 0
	local.tee 41
	f32.const 0x0p0
	f32.le 
	br_if 0
	f32.const 0x1p0
	local.set 38
	local.get 41
	f32.const 0x1p0
	f32.ge 
	br_if 0
	local.get 41
	local.get 2
	call powf
	local.set 38
	end_block
	local.get 6
	i32.const 16
	i32.add 
	local.set 6
	local.get 8
	local.get 9
	local.get 38
	f32.mul 
	f32.store 0
	local.get 5
	i32.const 4
	i32.add 
	local.tee 5
	br_if 0
	end_loop
	end_block
	end_function
