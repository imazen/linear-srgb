.section .text.stub_unpremultiply_linear_to_srgb_rgba_slice,"",@
	.globl	stub_unpremultiply_linear_to_srgb_rgba_slice
.type	stub_unpremultiply_linear_to_srgb_rgba_slice,@function
stub_unpremultiply_linear_to_srgb_rgba_slice:
	.functype	stub_unpremultiply_linear_to_srgb_rgba_slice (i32, i32) -> ()
	.local  	i32, i32, i32, i32, i32, i32, i32, v128, v128, f32, f32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, f32, f32, f32
	global.get __stack_pointer
	i32.const 144
	i32.sub 
	local.tee 2
	global.set __stack_pointer
	local.get 1
	i32.const 4
	i32.shr_u
	local.set 3
	local.get 0
	local.get 1
	i32.const 536870896
	i32.and 
	i32.const 2
	i32.shl 
	i32.add 
	local.set 4
	i32.const 0
	local.set 5
	block 
	local.get 1
	i32.const 12
	i32.and 
	local.tee 6
	i32.eqz
	br_if 0
	block 
	i32.const 64
	local.get 6
	i32.const 2
	i32.shl 
	local.tee 1
	i32.sub 
	local.tee 5
	i32.eqz
	br_if 0
	local.get 2
	i32.const 80
	i32.add 
	local.get 1
	i32.add 
	i32.const 0
	local.get 5
	memory.fill 0
	end_block
	block 
	local.get 1
	i32.eqz
	br_if 0
	local.get 2
	i32.const 80
	i32.add 
	local.get 4
	local.get 1
	memory.copy 0, 0
	end_block
	local.get 2
	local.get 2
	v128.load 128:p2align=2
	v128.store 64:p2align=2
	local.get 2
	local.get 2
	v128.load 112:p2align=2
	v128.store 48:p2align=2
	local.get 2
	local.get 2
	v128.load 96:p2align=2
	v128.store 32:p2align=2
	local.get 2
	local.get 2
	v128.load 80:p2align=2
	v128.store 16:p2align=2
	i32.const 1
	local.set 5
	end_block
	local.get 2
	local.get 5
	i32.store 12
	local.get 2
	i32.const 12
	i32.add 
	i32.const 4
	i32.add 
	local.tee 7
	i32.const 0
	local.get 6
	i32.select
	local.set 5
	local.get 0
	local.get 3
	i32.const 6
	i32.shl 
	i32.add 
	local.set 8
	block 
	loop 
	block 
	block 
	local.get 0
	i32.eqz
	br_if 0
	local.get 0
	local.get 8
	i32.eq 
	br_if 0
	local.get 0
	local.set 1
	local.get 0
	i32.const 64
	i32.add 
	local.set 0
	local.get 5
	local.set 3
	br 1
	end_block
	i32.const 0
	local.set 0
	local.get 5
	local.set 1
	i32.const 0
	local.set 3
	local.get 5
	i32.eqz
	br_if 2
	end_block
	local.get 1
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	local.tee 9
	local.get 1
	v128.load 48:p2align=2
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 10
	f32.const 0x1p0
	local.get 1
	f32.load 60
	local.tee 11
	f32.div 
	f32.const 0x0p0
	local.get 11
	f32.const 0x1p-10
	f32.gt 
	f32.select
	local.tee 12
	f32x4.replace_lane 0
	local.get 12
	f32x4.replace_lane 1
	local.get 12
	f32x4.replace_lane 2
	f32x4.mul
	local.tee 13
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 14
	f32x4.max
	local.get 10
	f32x4.min
	local.tee 15
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	local.tee 16
	f32x4.mul
	local.get 15
	f32x4.sqrt
	local.tee 17
	local.get 17
	local.get 17
	local.get 17
	v128.const 0x1.bec99p4, 0x1.bec99p4, 0x1.bec99p4, 0x1.bec99p4
	local.tee 18
	f32x4.mul
	v128.const 0x1.902b84p5, 0x1.902b84p5, 0x1.902b84p5, 0x1.902b84p5
	local.tee 19
	f32x4.add
	f32x4.mul
	v128.const 0x1.72539ap3, 0x1.72539ap3, 0x1.72539ap3, 0x1.72539ap3
	local.tee 20
	f32x4.add
	f32x4.mul
	v128.const 0x1.7e7074p-4, 0x1.7e7074p-4, 0x1.7e7074p-4, 0x1.7e7074p-4
	local.tee 21
	f32x4.add
	f32x4.mul
	v128.const -0x1.bc7a84p-7, -0x1.bc7a84p-7, -0x1.bc7a84p-7, -0x1.bc7a84p-7
	local.tee 22
	f32x4.add
	local.get 17
	local.get 17
	local.get 17
	local.get 17
	v128.const 0x1.14776ap5, 0x1.14776ap5, 0x1.14776ap5, 0x1.14776ap5
	local.tee 23
	f32x4.add
	f32x4.mul
	v128.const 0x1.66381cp5, 0x1.66381cp5, 0x1.66381cp5, 0x1.66381cp5
	local.tee 24
	f32x4.add
	f32x4.mul
	v128.const 0x1.1ff722p3, 0x1.1ff722p3, 0x1.1ff722p3, 0x1.1ff722p3
	local.tee 25
	f32x4.add
	f32x4.mul
	v128.const 0x1.0db3d2p-2, 0x1.0db3d2p-2, 0x1.0db3d2p-2, 0x1.0db3d2p-2
	local.tee 26
	f32x4.add
	f32x4.div
	local.get 10
	f32x4.min
	local.get 15
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	local.tee 27
	f32x4.lt
	v128.bitselect
	local.get 13
	local.get 10
	f32x4.ge
	v128.bitselect
	v128.store 48:p2align=2
	local.get 1
	local.get 9
	local.get 1
	v128.load 32:p2align=2
	local.get 10
	f32.const 0x1p0
	local.get 1
	f32.load 44
	local.tee 12
	f32.div 
	f32.const 0x0p0
	local.get 12
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
	local.tee 13
	local.get 14
	f32x4.max
	local.get 10
	f32x4.min
	local.tee 15
	local.get 16
	f32x4.mul
	local.get 15
	f32x4.sqrt
	local.tee 17
	local.get 17
	local.get 17
	local.get 17
	local.get 18
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	local.get 17
	local.get 17
	local.get 17
	local.get 17
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 24
	f32x4.add
	f32x4.mul
	local.get 25
	f32x4.add
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.div
	local.get 10
	f32x4.min
	local.get 15
	local.get 27
	f32x4.lt
	v128.bitselect
	local.get 13
	local.get 10
	f32x4.ge
	v128.bitselect
	v128.store 32:p2align=2
	local.get 1
	local.get 9
	local.get 1
	v128.load 16:p2align=2
	local.get 10
	f32.const 0x1p0
	local.get 1
	f32.load 28
	local.tee 28
	f32.div 
	f32.const 0x0p0
	local.get 28
	f32.const 0x1p-10
	f32.gt 
	f32.select
	local.tee 29
	f32x4.replace_lane 0
	local.get 29
	f32x4.replace_lane 1
	local.get 29
	f32x4.replace_lane 2
	f32x4.mul
	local.tee 13
	local.get 14
	f32x4.max
	local.get 10
	f32x4.min
	local.tee 15
	local.get 16
	f32x4.mul
	local.get 15
	f32x4.sqrt
	local.tee 17
	local.get 17
	local.get 17
	local.get 17
	local.get 18
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	local.get 17
	local.get 17
	local.get 17
	local.get 17
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 24
	f32x4.add
	f32x4.mul
	local.get 25
	f32x4.add
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.div
	local.get 10
	f32x4.min
	local.get 15
	local.get 27
	f32x4.lt
	v128.bitselect
	local.get 13
	local.get 10
	f32x4.ge
	v128.bitselect
	v128.store 16:p2align=2
	local.get 1
	local.get 9
	local.get 1
	v128.load 0:p2align=2
	local.get 10
	f32.const 0x1p0
	local.get 1
	f32.load 12
	local.tee 29
	f32.div 
	f32.const 0x0p0
	local.get 29
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
	local.tee 15
	local.get 14
	f32x4.max
	local.get 10
	f32x4.min
	local.tee 14
	local.get 16
	f32x4.mul
	local.get 14
	f32x4.sqrt
	local.tee 17
	local.get 17
	local.get 17
	local.get 17
	local.get 18
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.mul
	local.get 22
	f32x4.add
	local.get 17
	local.get 17
	local.get 17
	local.get 17
	local.get 23
	f32x4.add
	f32x4.mul
	local.get 24
	f32x4.add
	f32x4.mul
	local.get 25
	f32x4.add
	f32x4.mul
	local.get 26
	f32x4.add
	f32x4.div
	local.get 10
	f32x4.min
	local.get 14
	local.get 27
	f32x4.lt
	v128.bitselect
	local.get 15
	local.get 10
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 1
	local.get 11
	f32.store 60
	local.get 1
	local.get 12
	f32.store 44
	local.get 1
	local.get 28
	f32.store 28
	local.get 1
	local.get 29
	f32.store 12
	local.get 3
	local.set 5
	br 0
	end_loop
	end_block
	block 
	local.get 2
	i32.load 12
	i32.const 1
	i32.ne 
	br_if 0
	local.get 6
	i32.const 2
	i32.shl 
	local.tee 1
	i32.eqz
	br_if 0
	local.get 4
	local.get 7
	local.get 1
	memory.copy 0, 0
	end_block
	local.get 2
	i32.const 144
	i32.add 
	global.set __stack_pointer
	end_function
