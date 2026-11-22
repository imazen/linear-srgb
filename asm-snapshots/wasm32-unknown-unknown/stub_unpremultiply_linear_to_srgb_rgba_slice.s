.section .text.stub_unpremultiply_linear_to_srgb_rgba_slice,"",@
	.globl	stub_unpremultiply_linear_to_srgb_rgba_slice
.type	stub_unpremultiply_linear_to_srgb_rgba_slice,@function
stub_unpremultiply_linear_to_srgb_rgba_slice:
	.functype	stub_unpremultiply_linear_to_srgb_rgba_slice (i32, i32) -> ()
	.local  	v128, v128, v128, v128, v128
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
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 2
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.get 0
	v128.load 0:p2align=2
	local.tee 3
	local.get 4
	i8x16.shuffle 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15
	local.tee 4
	v128.const 0x1p-10, 0x1p-10, 0x1p-10, 0x1p-10
	f32x4.gt
	local.get 2
	local.get 4
	f32x4.div
	v128.and
	local.get 3
	f32x4.mul
	local.tee 5
	f32x4.pmax
	f32x4.pmin
	local.tee 6
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	f32x4.mul
	local.get 6
	f32x4.sqrt
	local.tee 4
	local.get 4
	local.get 4
	local.get 4
	v128.const 0x1.bec99p4, 0x1.bec99p4, 0x1.bec99p4, 0x1.bec99p4
	f32x4.mul
	v128.const 0x1.902b84p5, 0x1.902b84p5, 0x1.902b84p5, 0x1.902b84p5
	f32x4.add
	f32x4.mul
	v128.const 0x1.72539ap3, 0x1.72539ap3, 0x1.72539ap3, 0x1.72539ap3
	f32x4.add
	f32x4.mul
	v128.const 0x1.7e7074p-4, 0x1.7e7074p-4, 0x1.7e7074p-4, 0x1.7e7074p-4
	f32x4.add
	f32x4.mul
	v128.const -0x1.bc7a84p-7, -0x1.bc7a84p-7, -0x1.bc7a84p-7, -0x1.bc7a84p-7
	f32x4.add
	local.get 4
	local.get 4
	local.get 4
	local.get 4
	v128.const 0x1.14776ap5, 0x1.14776ap5, 0x1.14776ap5, 0x1.14776ap5
	f32x4.add
	f32x4.mul
	v128.const 0x1.66381cp5, 0x1.66381cp5, 0x1.66381cp5, 0x1.66381cp5
	f32x4.add
	f32x4.mul
	v128.const 0x1.1ff722p3, 0x1.1ff722p3, 0x1.1ff722p3, 0x1.1ff722p3
	f32x4.add
	f32x4.mul
	v128.const 0x1.0db3d2p-2, 0x1.0db3d2p-2, 0x1.0db3d2p-2, 0x1.0db3d2p-2
	f32x4.add
	f32x4.div
	local.get 6
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	f32x4.le
	v128.bitselect
	local.get 5
	local.get 2
	f32x4.ge
	v128.bitselect
	local.get 3
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
	end_function
