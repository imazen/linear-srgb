.section .text.stub_gamma_to_linear_slice,"",@
	.globl	stub_gamma_to_linear_slice
.type	stub_gamma_to_linear_slice,@function
stub_gamma_to_linear_slice:
	.functype	stub_gamma_to_linear_slice (i32, i32, f32) -> ()
	.local  	i32, i32, i32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, i32, v128, f32, f32, f32
	block 
	local.get 1
	i32.const 2
	i32.shl 
	local.tee 3
	i32.const 2147483584
	i32.and 
	local.tee 4
	i32.eqz
	br_if 0
	local.get 0
	local.get 4
	i32.add 
	local.set 5
	local.get 2
	f32x4.splat
	local.set 6
	local.get 0
	local.set 4
	loop 
	local.get 4
	v128.const 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127
	local.tee 7
	v128.const 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	local.tee 8
	local.get 6
	v128.const 0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 192, 127
	local.tee 9
	v128.const 0, 0, 128, 255, 0, 0, 128, 255, 0, 0, 128, 255, 0, 0, 128, 255
	local.tee 10
	local.get 4
	v128.load 0:p2align=2
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 11
	f32x4.max
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 12
	f32x4.min
	local.tee 13
	v128.const 4913933, 4913933, 4913933, 4913933
	local.tee 14
	i32x4.add
	local.tee 15
	v128.const 8388607, 8388607, 8388607, 8388607
	local.tee 16
	v128.and
	v128.const 1060439283, 1060439283, 1060439283, 1060439283
	local.tee 17
	i32x4.add
	local.tee 18
	v128.const -0x1p0, -0x1p0, -0x1p0, -0x1p0
	local.tee 19
	f32x4.add
	local.get 18
	local.get 12
	f32x4.add
	f32x4.div
	local.tee 18
	local.get 18
	local.get 18
	f32x4.mul
	local.tee 18
	local.get 18
	local.get 18
	v128.const 0x1.bcd67ep-2, 0x1.bcd67ep-2, 0x1.bcd67ep-2, 0x1.bcd67ep-2
	local.tee 20
	f32x4.mul
	v128.const 0x1.276932p-1, 0x1.276932p-1, 0x1.276932p-1, 0x1.276932p-1
	local.tee 21
	f32x4.add
	f32x4.mul
	v128.const 0x1.ec7126p-1, 0x1.ec7126p-1, 0x1.ec7126p-1, 0x1.ec7126p-1
	local.tee 22
	f32x4.add
	f32x4.mul
	v128.const 0x1.715476p1, 0x1.715476p1, 0x1.715476p1, 0x1.715476p1
	local.tee 23
	f32x4.add
	f32x4.mul
	local.get 15
	i32.const 23
	i32x4.shr_s
	v128.const -127, -127, -127, -127
	local.tee 24
	i32x4.add
	f32x4.convert_i32x4_s
	f32x4.add
	local.get 13
	local.get 11
	f32x4.eq
	v128.bitselect
	local.get 8
	v128.bitselect
	f32x4.mul
	local.tee 25
	v128.const -0x1.f8p6, -0x1.f8p6, -0x1.f8p6, -0x1.f8p6
	local.tee 18
	f32x4.max
	v128.const 0x1p7, 0x1p7, 0x1p7, 0x1p7
	local.tee 13
	f32x4.min
	local.tee 15
	local.get 15
	f32x4.nearest
	v128.const 0x1.fcp6, 0x1.fcp6, 0x1.fcp6, 0x1.fcp6
	local.tee 26
	f32x4.min
	local.tee 27
	f32x4.sub
	local.tee 15
	local.get 15
	local.get 15
	local.get 15
	local.get 15
	local.get 15
	v128.const 0x1.43f274p-13, 0x1.43f274p-13, 0x1.43f274p-13, 0x1.43f274p-13
	local.tee 28
	f32x4.mul
	v128.const 0x1.5d88f2p-10, 0x1.5d88f2p-10, 0x1.5d88f2p-10, 0x1.5d88f2p-10
	local.tee 29
	f32x4.add
	f32x4.mul
	v128.const 0x1.3b2a18p-7, 0x1.3b2a18p-7, 0x1.3b2a18p-7, 0x1.3b2a18p-7
	local.tee 30
	f32x4.add
	f32x4.mul
	v128.const 0x1.c6b178p-5, 0x1.c6b178p-5, 0x1.c6b178p-5, 0x1.c6b178p-5
	local.tee 31
	f32x4.add
	f32x4.mul
	v128.const 0x1.ebfbdap-3, 0x1.ebfbdap-3, 0x1.ebfbdap-3, 0x1.ebfbdap-3
	local.tee 32
	f32x4.add
	f32x4.mul
	v128.const 0x1.62e43p-1, 0x1.62e43p-1, 0x1.62e43p-1, 0x1.62e43p-1
	local.tee 33
	f32x4.add
	f32x4.mul
	local.get 12
	f32x4.add
	local.get 27
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	local.tee 27
	i32x4.add
	f32x4.mul
	local.get 25
	local.get 18
	f32x4.lt
	v128.bitselect
	local.get 25
	local.get 13
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	i32.const 48
	i32.add 
	local.tee 34
	local.get 7
	local.get 8
	local.get 6
	local.get 9
	local.get 10
	local.get 34
	v128.load 0:p2align=2
	local.get 11
	f32x4.max
	local.get 12
	f32x4.min
	local.tee 25
	local.get 14
	i32x4.add
	local.tee 35
	local.get 16
	v128.and
	local.get 17
	i32x4.add
	local.tee 15
	local.get 19
	f32x4.add
	local.get 15
	local.get 12
	f32x4.add
	f32x4.div
	local.tee 15
	local.get 15
	local.get 15
	f32x4.mul
	local.tee 15
	local.get 15
	local.get 15
	local.get 20
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
	local.get 35
	i32.const 23
	i32x4.shr_s
	local.get 24
	i32x4.add
	f32x4.convert_i32x4_s
	f32x4.add
	local.get 25
	local.get 11
	f32x4.eq
	v128.bitselect
	local.get 8
	v128.bitselect
	f32x4.mul
	local.tee 25
	local.get 18
	f32x4.max
	local.get 13
	f32x4.min
	local.tee 15
	local.get 15
	f32x4.nearest
	local.get 26
	f32x4.min
	local.tee 35
	f32x4.sub
	local.tee 15
	local.get 15
	local.get 15
	local.get 15
	local.get 15
	local.get 15
	local.get 28
	f32x4.mul
	local.get 29
	f32x4.add
	f32x4.mul
	local.get 30
	f32x4.add
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
	local.get 12
	f32x4.add
	local.get 35
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 27
	i32x4.add
	f32x4.mul
	local.get 25
	local.get 18
	f32x4.lt
	v128.bitselect
	local.get 25
	local.get 13
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	i32.const 32
	i32.add 
	local.tee 34
	local.get 7
	local.get 8
	local.get 6
	local.get 9
	local.get 10
	local.get 34
	v128.load 0:p2align=2
	local.get 11
	f32x4.max
	local.get 12
	f32x4.min
	local.tee 25
	local.get 14
	i32x4.add
	local.tee 35
	local.get 16
	v128.and
	local.get 17
	i32x4.add
	local.tee 15
	local.get 19
	f32x4.add
	local.get 15
	local.get 12
	f32x4.add
	f32x4.div
	local.tee 15
	local.get 15
	local.get 15
	f32x4.mul
	local.tee 15
	local.get 15
	local.get 15
	local.get 20
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
	local.get 35
	i32.const 23
	i32x4.shr_s
	local.get 24
	i32x4.add
	f32x4.convert_i32x4_s
	f32x4.add
	local.get 25
	local.get 11
	f32x4.eq
	v128.bitselect
	local.get 8
	v128.bitselect
	f32x4.mul
	local.tee 25
	local.get 18
	f32x4.max
	local.get 13
	f32x4.min
	local.tee 15
	local.get 15
	f32x4.nearest
	local.get 26
	f32x4.min
	local.tee 35
	f32x4.sub
	local.tee 15
	local.get 15
	local.get 15
	local.get 15
	local.get 15
	local.get 15
	local.get 28
	f32x4.mul
	local.get 29
	f32x4.add
	f32x4.mul
	local.get 30
	f32x4.add
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
	local.get 12
	f32x4.add
	local.get 35
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 27
	i32x4.add
	f32x4.mul
	local.get 25
	local.get 18
	f32x4.lt
	v128.bitselect
	local.get 25
	local.get 13
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	i32.const 16
	i32.add 
	local.tee 34
	local.get 7
	local.get 8
	local.get 6
	local.get 9
	local.get 10
	local.get 34
	v128.load 0:p2align=2
	local.get 11
	f32x4.max
	local.get 12
	f32x4.min
	local.tee 25
	local.get 14
	i32x4.add
	local.tee 14
	local.get 16
	v128.and
	local.get 17
	i32x4.add
	local.tee 15
	local.get 19
	f32x4.add
	local.get 15
	local.get 12
	f32x4.add
	f32x4.div
	local.tee 15
	local.get 15
	local.get 15
	f32x4.mul
	local.tee 15
	local.get 15
	local.get 15
	local.get 20
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
	local.get 14
	i32.const 23
	i32x4.shr_s
	local.get 24
	i32x4.add
	f32x4.convert_i32x4_s
	f32x4.add
	local.get 25
	local.get 11
	f32x4.eq
	v128.bitselect
	local.get 8
	v128.bitselect
	f32x4.mul
	local.tee 15
	local.get 18
	f32x4.max
	local.get 13
	f32x4.min
	local.tee 11
	local.get 11
	f32x4.nearest
	local.get 26
	f32x4.min
	local.tee 9
	f32x4.sub
	local.tee 11
	local.get 11
	local.get 11
	local.get 11
	local.get 11
	local.get 11
	local.get 28
	f32x4.mul
	local.get 29
	f32x4.add
	f32x4.mul
	local.get 30
	f32x4.add
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
	local.get 12
	f32x4.add
	local.get 9
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 27
	i32x4.add
	f32x4.mul
	local.get 15
	local.get 18
	f32x4.lt
	v128.bitselect
	local.get 15
	local.get 13
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	i32.const 64
	i32.add 
	local.tee 4
	local.get 5
	i32.ne 
	br_if 0
	end_loop
	end_block
	block 
	local.get 3
	i32.const 60
	i32.and 
	local.tee 34
	i32.eqz
	br_if 0
	local.get 0
	local.get 1
	i32.const 536870896
	i32.and 
	i32.const 2
	i32.shl 
	i32.add 
	local.tee 5
	local.set 4
	block 
	local.get 34
	i32.const -4
	i32.add 
	local.tee 0
	i32.const 4
	i32.and 
	br_if 0
	f32.const 0x0p0
	local.set 36
	block 
	local.get 5
	f32.load 0
	local.tee 37
	f32.const 0x0p0
	f32.le 
	br_if 0
	f32.const 0x1p0
	local.set 36
	local.get 37
	f32.const 0x1p0
	f32.ge 
	br_if 0
	local.get 37
	local.get 2
	call powf
	local.set 36
	end_block
	local.get 5
	i32.const 4
	i32.add 
	local.set 4
	local.get 5
	local.get 36
	f32.store 0
	end_block
	local.get 0
	i32.eqz
	br_if 0
	local.get 5
	local.get 34
	i32.add 
	local.set 5
	loop 
	f32.const 0x0p0
	local.set 36
	f32.const 0x0p0
	local.set 37
	block 
	local.get 4
	f32.load 0
	local.tee 38
	f32.const 0x0p0
	f32.le 
	br_if 0
	f32.const 0x1p0
	local.set 37
	local.get 38
	f32.const 0x1p0
	f32.ge 
	br_if 0
	local.get 38
	local.get 2
	call powf
	local.set 37
	end_block
	local.get 4
	local.get 37
	f32.store 0
	block 
	local.get 4
	i32.const 4
	i32.add 
	local.tee 34
	f32.load 0
	local.tee 37
	f32.const 0x0p0
	f32.le 
	br_if 0
	f32.const 0x1p0
	local.set 36
	local.get 37
	f32.const 0x1p0
	f32.ge 
	br_if 0
	local.get 37
	local.get 2
	call powf
	local.set 36
	end_block
	local.get 34
	local.get 36
	f32.store 0
	local.get 4
	i32.const 8
	i32.add 
	local.tee 4
	local.get 5
	i32.ne 
	br_if 0
	end_loop
	end_block
	end_function
