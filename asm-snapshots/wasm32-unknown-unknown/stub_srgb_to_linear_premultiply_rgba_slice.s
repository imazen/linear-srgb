.section .text.stub_srgb_to_linear_premultiply_rgba_slice,"",@
	.globl	stub_srgb_to_linear_premultiply_rgba_slice
.type	stub_srgb_to_linear_premultiply_rgba_slice,@function
stub_srgb_to_linear_premultiply_rgba_slice:
	.functype	stub_srgb_to_linear_premultiply_rgba_slice (i32, i32) -> ()
	.local  	f32, v128, v128, v128
	block 
	local.get 1
	i32.const 2
	i32.shl 
	i32.const 2147483632
	i32.and 
	local.tee 1
	i32.eqz
	br_if 0
	loop 
	local.get 0
	i32.const 8
	i32.add 
	local.get 0
	i32.const 12
	i32.add 
	f32.load 0
	local.tee 2
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	local.get 0
	v128.load 0:p2align=2
	local.tee 3
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	f32x4.max
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 4
	f32x4.min
	local.tee 5
	v128.const 0x1.3d0722p-4, 0x1.3d0722p-4, 0x1.3d0722p-4, 0x1.3d0722p-4
	f32x4.mul
	local.get 5
	local.get 5
	local.get 5
	local.get 5
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
	local.get 5
	local.get 5
	local.get 5
	local.get 5
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
	local.get 4
	f32x4.min
	local.get 5
	v128.const 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5
	f32x4.lt
	v128.bitselect
	local.get 3
	local.get 4
	f32x4.ge
	v128.bitselect
	local.tee 5
	f32x4.extract_lane 2
	f32.mul 
	f32.store 0
	local.get 0
	i32.const 4
	i32.add 
	local.get 2
	local.get 5
	f32x4.extract_lane 1
	f32.mul 
	f32.store 0
	local.get 0
	local.get 2
	local.get 5
	f32x4.extract_lane 0
	f32.mul 
	f32.store 0
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
