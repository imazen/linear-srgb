.section .text.stub_srgb_to_linear_slice,"",@
	.globl	stub_srgb_to_linear_slice
.type	stub_srgb_to_linear_slice,@function
stub_srgb_to_linear_slice:
	.functype	stub_srgb_to_linear_slice (i32, i32) -> ()
	.local  	i32, i32, i32, v128, v128, v128, f32, f32
	block 
	local.get 1
	i32.const 2
	i32.shl 
	local.tee 2
	i32.const 2147483632
	i32.and 
	local.tee 3
	i32.eqz
	br_if 0
	local.get 0
	local.set 4
	loop 
	local.get 4
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	local.get 4
	v128.load 0:p2align=2
	local.tee 5
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	f32x4.max
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 6
	f32x4.min
	local.tee 7
	v128.const 0x1.3d0722p-4, 0x1.3d0722p-4, 0x1.3d0722p-4, 0x1.3d0722p-4
	f32x4.mul
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	v128.const 0x1.48fc2ap6, 0x1.48fc2ap6, 0x1.48fc2ap6, 0x1.48fc2ap6
	f32x4.mul
	v128.const 0x1.122cccp6, 0x1.122cccp6, 0x1.122cccp6, 0x1.122cccp6
	f32x4.add
	f32x4.mul
	v128.const 0x1.9c4cfap3, 0x1.9c4cfap3, 0x1.9c4cfap3, 0x1.9c4cfap3
	f32x4.add
	f32x4.mul
	v128.const 0x1.9e2f2cp-1, 0x1.9e2f2cp-1, 0x1.9e2f2cp-1, 0x1.9e2f2cp-1
	f32x4.add
	f32x4.mul
	v128.const 0x1.1212eep-6, 0x1.1212eep-6, 0x1.1212eep-6, 0x1.1212eep-6
	f32x4.add
	local.get 7
	local.get 7
	local.get 7
	local.get 7
	v128.const -0x1.c80a74p2, -0x1.c80a74p2, -0x1.c80a74p2, -0x1.c80a74p2
	f32x4.add
	f32x4.mul
	v128.const 0x1.ae2f2cp5, 0x1.ae2f2cp5, 0x1.ae2f2cp5, 0x1.ae2f2cp5
	f32x4.add
	f32x4.mul
	v128.const 0x1.83423cp6, 0x1.83423cp6, 0x1.83423cp6, 0x1.83423cp6
	f32x4.add
	f32x4.mul
	v128.const 0x1.409bf8p4, 0x1.409bf8p4, 0x1.409bf8p4, 0x1.409bf8p4
	f32x4.add
	f32x4.div
	local.get 6
	f32x4.min
	local.get 7
	v128.const 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5
	f32x4.lt
	v128.bitselect
	local.get 5
	local.get 6
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	i32.const 16
	i32.add 
	local.set 4
	local.get 3
	i32.const -16
	i32.add 
	local.tee 3
	br_if 0
	end_loop
	end_block
	block 
	local.get 2
	i32.const 12
	i32.and 
	local.tee 3
	i32.eqz
	br_if 0
	local.get 0
	local.get 1
	i32.const 536870908
	i32.and 
	i32.const 2
	i32.shl 
	i32.add 
	local.set 1
	loop 
	f32.const 0x0p0
	local.set 8
	block 
	local.get 1
	local.tee 4
	f32.load 0
	local.tee 9
	f32.const 0x0p0
	f32.lt 
	br_if 0
	block 
	local.get 9
	f32.const 0x1.41e42cp-5
	f32.lt 
	br_if 0
	f32.const 0x1p0
	local.set 8
	local.get 9
	f32.const 0x1p0
	f32.lt 
	i32.eqz
	br_if 1
	local.get 9
	f32.const 0x1.c2a5d6p-5
	f32.add 
	f32.const 0x1.0e152ep0
	f32.div 
	f32.const 0x1.333334p1
	call powf
	local.set 8
	br 1
	end_block
	local.get 9
	f32.const 0x1.3d0722p-4
	f32.mul 
	local.set 8
	end_block
	local.get 4
	i32.const 4
	i32.add 
	local.set 1
	local.get 4
	local.get 8
	f32.store 0
	local.get 3
	i32.const -4
	i32.add 
	local.tee 3
	br_if 0
	end_loop
	end_block
	end_function
