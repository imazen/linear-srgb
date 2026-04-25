.section .text.stub_unpremultiply_linear_to_srgb_rgba_slice,"",@
	.globl	stub_unpremultiply_linear_to_srgb_rgba_slice
.type	stub_unpremultiply_linear_to_srgb_rgba_slice,@function
stub_unpremultiply_linear_to_srgb_rgba_slice:
	.functype	stub_unpremultiply_linear_to_srgb_rgba_slice (i32, i32) -> ()
	.local  	i32, i32, i32, i32, v128, v128, f32, f32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, i32, f32, i32, f32, i32, f32
	block 
	local.get 1
	i32.const 2
	i32.shl 
	i32.const 2147483584
	i32.and 
	local.tee 2
	i32.eqz
	br_if 0
	i32.const 0
	local.set 3
	loop 
	local.get 0
	local.get 3
	i32.add 
	local.tee 4
	i32.const 48
	i32.add 
	local.tee 5
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	local.tee 6
	local.get 5
	v128.load 0:p2align=2
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 7
	f32.const 0x1p0
	local.get 4
	i32.const 60
	i32.add 
	local.tee 5
	f32.load 0
	local.tee 8
	f32.div 
	f32.const 0x0p0
	local.get 8
	f32.const 0x1p-10
	f32.gt 
	f32.select
	local.tee 9
	f32x4.replace_lane 0
	local.get 9
	f32x4.replace_lane 1
	local.get 9
	f32x4.replace_lane 2
	f32x4.mul
	local.tee 10
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 11
	f32x4.max
	local.get 7
	f32x4.min
	local.tee 12
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	local.tee 13
	f32x4.mul
	local.get 12
	f32x4.sqrt
	local.tee 14
	local.get 14
	local.get 14
	local.get 14
	v128.const 0x1.bec99p4, 0x1.bec99p4, 0x1.bec99p4, 0x1.bec99p4
	local.tee 15
	f32x4.mul
	v128.const 0x1.902b84p5, 0x1.902b84p5, 0x1.902b84p5, 0x1.902b84p5
	local.tee 16
	f32x4.add
	f32x4.mul
	v128.const 0x1.72539ap3, 0x1.72539ap3, 0x1.72539ap3, 0x1.72539ap3
	local.tee 17
	f32x4.add
	f32x4.mul
	v128.const 0x1.7e7074p-4, 0x1.7e7074p-4, 0x1.7e7074p-4, 0x1.7e7074p-4
	local.tee 18
	f32x4.add
	f32x4.mul
	v128.const -0x1.bc7a84p-7, -0x1.bc7a84p-7, -0x1.bc7a84p-7, -0x1.bc7a84p-7
	local.tee 19
	f32x4.add
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	v128.const 0x1.14776ap5, 0x1.14776ap5, 0x1.14776ap5, 0x1.14776ap5
	local.tee 20
	f32x4.add
	f32x4.mul
	v128.const 0x1.66381cp5, 0x1.66381cp5, 0x1.66381cp5, 0x1.66381cp5
	local.tee 21
	f32x4.add
	f32x4.mul
	v128.const 0x1.1ff722p3, 0x1.1ff722p3, 0x1.1ff722p3, 0x1.1ff722p3
	local.tee 22
	f32x4.add
	f32x4.mul
	v128.const 0x1.0db3d2p-2, 0x1.0db3d2p-2, 0x1.0db3d2p-2, 0x1.0db3d2p-2
	local.tee 23
	f32x4.add
	f32x4.div
	local.get 7
	f32x4.min
	local.get 12
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	local.tee 24
	f32x4.lt
	v128.bitselect
	local.get 10
	local.get 7
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	i32.const 32
	i32.add 
	local.tee 25
	local.get 6
	local.get 25
	v128.load 0:p2align=2
	local.get 7
	f32.const 0x1p0
	local.get 4
	i32.const 44
	i32.add 
	local.tee 25
	f32.load 0
	local.tee 9
	f32.div 
	f32.const 0x0p0
	local.get 9
	f32.const 0x1p-10
	f32.gt 
	f32.select
	local.tee 26
	f32x4.replace_lane 0
	local.get 26
	f32x4.replace_lane 1
	local.get 26
	f32x4.replace_lane 2
	f32x4.mul
	local.tee 10
	local.get 11
	f32x4.max
	local.get 7
	f32x4.min
	local.tee 12
	local.get 13
	f32x4.mul
	local.get 12
	f32x4.sqrt
	local.tee 14
	local.get 14
	local.get 14
	local.get 14
	local.get 15
	f32x4.mul
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
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.div
	local.get 7
	f32x4.min
	local.get 12
	local.get 24
	f32x4.lt
	v128.bitselect
	local.get 10
	local.get 7
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	i32.const 16
	i32.add 
	local.tee 27
	local.get 6
	local.get 27
	v128.load 0:p2align=2
	local.get 7
	f32.const 0x1p0
	local.get 4
	i32.const 28
	i32.add 
	local.tee 27
	f32.load 0
	local.tee 26
	f32.div 
	f32.const 0x0p0
	local.get 26
	f32.const 0x1p-10
	f32.gt 
	f32.select
	local.tee 28
	f32x4.replace_lane 0
	local.get 28
	f32x4.replace_lane 1
	local.get 28
	f32x4.replace_lane 2
	f32x4.mul
	local.tee 10
	local.get 11
	f32x4.max
	local.get 7
	f32x4.min
	local.tee 12
	local.get 13
	f32x4.mul
	local.get 12
	f32x4.sqrt
	local.tee 14
	local.get 14
	local.get 14
	local.get 14
	local.get 15
	f32x4.mul
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
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.div
	local.get 7
	f32x4.min
	local.get 12
	local.get 24
	f32x4.lt
	v128.bitselect
	local.get 10
	local.get 7
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	local.get 6
	local.get 4
	v128.load 0:p2align=2
	local.get 7
	f32.const 0x1p0
	local.get 4
	i32.const 12
	i32.add 
	local.tee 29
	f32.load 0
	local.tee 28
	f32.div 
	f32.const 0x0p0
	local.get 28
	f32.const 0x1p-10
	f32.gt 
	f32.select
	local.tee 30
	f32x4.replace_lane 0
	local.get 30
	f32x4.replace_lane 1
	local.get 30
	f32x4.replace_lane 2
	f32x4.mul
	local.tee 12
	local.get 11
	f32x4.max
	local.get 7
	f32x4.min
	local.tee 11
	local.get 13
	f32x4.mul
	local.get 11
	f32x4.sqrt
	local.tee 14
	local.get 14
	local.get 14
	local.get 14
	local.get 15
	f32x4.mul
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
	local.get 14
	local.get 14
	local.get 14
	local.get 14
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	f32x4.mul
	local.get 23
	f32x4.add
	f32x4.div
	local.get 7
	f32x4.min
	local.get 11
	local.get 24
	f32x4.lt
	v128.bitselect
	local.get 12
	local.get 7
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 5
	local.get 8
	f32.store 0
	local.get 25
	local.get 9
	f32.store 0
	local.get 27
	local.get 26
	f32.store 0
	local.get 29
	local.get 28
	f32.store 0
	local.get 2
	local.get 3
	i32.const 64
	i32.add 
	local.tee 3
	i32.ne 
	br_if 0
	end_loop
	end_block
	block 
	local.get 1
	i32.const 12
	i32.and 
	local.tee 3
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
	i32.const 0
	local.get 3
	i32.sub 
	local.set 3
	loop 
	block 
	block 
	block 
	block 
	local.get 4
	i32.const 12
	i32.add 
	f32.load 0
	local.tee 8
	f32.const 0x1p-10
	f32.gt 
	i32.eqz
	br_if 0
	f32.const 0x0p0
	local.set 9
	f32.const 0x0p0
	local.set 26
	f32.const 0x1p0
	local.get 8
	f32.div 
	local.tee 8
	local.get 4
	f32.load 0
	f32.mul 
	local.tee 28
	f32.const 0x0p0
	f32.lt 
	br_if 2
	local.get 28
	f32.const 0x1.8ea082p-9
	f32.lt 
	br_if 1
	f32.const 0x1p0
	local.set 26
	local.get 28
	f32.const 0x1p0
	f32.lt 
	i32.eqz
	br_if 2
	local.get 28
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul 
	f32.const -0x1.c2a5d6p-5
	f32.add 
	local.set 26
	br 2
	end_block
	local.get 4
	i64.const 0
	i64.store 0:p2align=2
	local.get 4
	i32.const 8
	i32.add 
	i32.const 0
	i32.store 0
	br 2
	end_block
	local.get 28
	f32.const 0x1.9d70a4p3
	f32.mul 
	local.set 26
	end_block
	local.get 4
	local.get 26
	f32.store 0
	block 
	local.get 8
	local.get 4
	i32.const 4
	i32.add 
	local.tee 5
	f32.load 0
	f32.mul 
	local.tee 26
	f32.const 0x0p0
	f32.lt 
	br_if 0
	block 
	local.get 26
	f32.const 0x1.8ea082p-9
	f32.lt 
	br_if 0
	f32.const 0x1p0
	local.set 9
	local.get 26
	f32.const 0x1p0
	f32.lt 
	i32.eqz
	br_if 1
	local.get 26
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul 
	f32.const -0x1.c2a5d6p-5
	f32.add 
	local.set 9
	br 1
	end_block
	local.get 26
	f32.const 0x1.9d70a4p3
	f32.mul 
	local.set 9
	end_block
	local.get 5
	local.get 9
	f32.store 0
	f32.const 0x0p0
	local.set 9
	block 
	block 
	local.get 8
	local.get 4
	i32.const 8
	i32.add 
	local.tee 5
	f32.load 0
	f32.mul 
	local.tee 8
	f32.const 0x0p0
	f32.lt 
	br_if 0
	local.get 8
	f32.const 0x1.8ea082p-9
	f32.lt 
	br_if 1
	f32.const 0x1p0
	local.set 9
	local.get 8
	f32.const 0x1p0
	f32.lt 
	i32.eqz
	br_if 0
	local.get 8
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul 
	f32.const -0x1.c2a5d6p-5
	f32.add 
	local.set 9
	end_block
	local.get 5
	local.get 9
	f32.store 0
	br 1
	end_block
	local.get 5
	local.get 8
	f32.const 0x1.9d70a4p3
	f32.mul 
	f32.store 0
	end_block
	local.get 4
	i32.const 16
	i32.add 
	local.set 4
	local.get 3
	i32.const 4
	i32.add 
	local.tee 3
	br_if 0
	end_loop
	end_block
	end_function
