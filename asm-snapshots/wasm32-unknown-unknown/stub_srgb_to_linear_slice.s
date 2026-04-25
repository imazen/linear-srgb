.section .text.stub_srgb_to_linear_slice,"",@
	.globl	stub_srgb_to_linear_slice
.type	stub_srgb_to_linear_slice,@function
stub_srgb_to_linear_slice:
	.functype	stub_srgb_to_linear_slice (i32, i32) -> ()
	.local  	i32, i32, i32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, i32, f32, f32
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
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	local.tee 5
	local.get 3
	v128.load 0:p2align=2
	local.tee 6
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 7
	f32x4.max
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 8
	f32x4.min
	local.tee 9
	v128.const 0x1.3d0722p-4, 0x1.3d0722p-4, 0x1.3d0722p-4, 0x1.3d0722p-4
	local.tee 10
	f32x4.mul
	local.get 9
	local.get 9
	local.get 9
	local.get 9
	v128.const 0x1.48fc2ap6, 0x1.48fc2ap6, 0x1.48fc2ap6, 0x1.48fc2ap6
	local.tee 11
	f32x4.mul
	v128.const 0x1.122cccp6, 0x1.122cccp6, 0x1.122cccp6, 0x1.122cccp6
	local.tee 12
	f32x4.add
	f32x4.mul
	v128.const 0x1.9c4cfap3, 0x1.9c4cfap3, 0x1.9c4cfap3, 0x1.9c4cfap3
	local.tee 13
	f32x4.add
	f32x4.mul
	v128.const 0x1.9e2f2cp-1, 0x1.9e2f2cp-1, 0x1.9e2f2cp-1, 0x1.9e2f2cp-1
	local.tee 14
	f32x4.add
	f32x4.mul
	v128.const 0x1.1212eep-6, 0x1.1212eep-6, 0x1.1212eep-6, 0x1.1212eep-6
	local.tee 15
	f32x4.add
	local.get 9
	local.get 9
	local.get 9
	local.get 9
	v128.const -0x1.c80a74p2, -0x1.c80a74p2, -0x1.c80a74p2, -0x1.c80a74p2
	local.tee 16
	f32x4.add
	f32x4.mul
	v128.const 0x1.ae2f2cp5, 0x1.ae2f2cp5, 0x1.ae2f2cp5, 0x1.ae2f2cp5
	local.tee 17
	f32x4.add
	f32x4.mul
	v128.const 0x1.83423cp6, 0x1.83423cp6, 0x1.83423cp6, 0x1.83423cp6
	local.tee 18
	f32x4.add
	f32x4.mul
	v128.const 0x1.409bf8p4, 0x1.409bf8p4, 0x1.409bf8p4, 0x1.409bf8p4
	local.tee 19
	f32x4.add
	f32x4.div
	local.get 8
	f32x4.min
	local.get 9
	v128.const 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5
	local.tee 20
	f32x4.lt
	v128.bitselect
	local.get 6
	local.get 8
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 3
	i32.const 48
	i32.add 
	local.tee 21
	local.get 5
	local.get 21
	v128.load 0:p2align=2
	local.tee 6
	local.get 7
	f32x4.max
	local.get 8
	f32x4.min
	local.tee 9
	local.get 10
	f32x4.mul
	local.get 9
	local.get 9
	local.get 9
	local.get 9
	local.get 11
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
	local.get 9
	local.get 9
	local.get 9
	local.get 9
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
	f32x4.div
	local.get 8
	f32x4.min
	local.get 9
	local.get 20
	f32x4.lt
	v128.bitselect
	local.get 6
	local.get 8
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 3
	i32.const 32
	i32.add 
	local.tee 21
	local.get 5
	local.get 21
	v128.load 0:p2align=2
	local.tee 6
	local.get 7
	f32x4.max
	local.get 8
	f32x4.min
	local.tee 9
	local.get 10
	f32x4.mul
	local.get 9
	local.get 9
	local.get 9
	local.get 9
	local.get 11
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
	local.get 9
	local.get 9
	local.get 9
	local.get 9
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
	f32x4.div
	local.get 8
	f32x4.min
	local.get 9
	local.get 20
	f32x4.lt
	v128.bitselect
	local.get 6
	local.get 8
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 3
	i32.const 16
	i32.add 
	local.tee 21
	local.get 5
	local.get 21
	v128.load 0:p2align=2
	local.tee 6
	local.get 7
	f32x4.max
	local.get 8
	f32x4.min
	local.tee 9
	local.get 10
	f32x4.mul
	local.get 9
	local.get 9
	local.get 9
	local.get 9
	local.get 11
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
	local.get 9
	local.get 9
	local.get 9
	local.get 9
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
	f32x4.div
	local.get 8
	f32x4.min
	local.get 9
	local.get 20
	f32x4.lt
	v128.bitselect
	local.get 6
	local.get 8
	f32x4.ge
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
	local.tee 21
	i32.eqz
	br_if 0
	local.get 0
	local.get 1
	i32.const 536870896
	i32.and 
	i32.const 2
	i32.shl 
	i32.add 
	local.set 4
	loop 
	f32.const 0x0p0
	local.set 22
	block 
	local.get 4
	local.tee 3
	f32.load 0
	local.tee 23
	f32.const 0x0p0
	f32.lt 
	br_if 0
	block 
	local.get 23
	f32.const 0x1.41e42cp-5
	f32.lt 
	br_if 0
	f32.const 0x1p0
	local.set 22
	local.get 23
	f32.const 0x1p0
	f32.lt 
	i32.eqz
	br_if 1
	local.get 23
	f32.const 0x1.c2a5d6p-5
	f32.add 
	f32.const 0x1.0e152ep0
	f32.div 
	f32.const 0x1.333334p1
	call powf
	local.set 22
	br 1
	end_block
	local.get 23
	f32.const 0x1.3d0722p-4
	f32.mul 
	local.set 22
	end_block
	local.get 3
	i32.const 4
	i32.add 
	local.set 4
	local.get 3
	local.get 22
	f32.store 0
	local.get 21
	i32.const -4
	i32.add 
	local.tee 21
	br_if 0
	end_loop
	end_block
	end_function
