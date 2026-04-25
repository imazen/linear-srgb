.section .text.stub_linear_to_srgb_slice,"",@
	.globl	stub_linear_to_srgb_slice
.type	stub_linear_to_srgb_slice,@function
stub_linear_to_srgb_slice:
	.functype	stub_linear_to_srgb_slice (i32, i32) -> ()
	.local  	i32, i32, i32, v128, v128, v128, v128, f32, f32, f32
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
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	f32x4.mul
	local.get 7
	f32x4.sqrt
	local.tee 8
	local.get 8
	local.get 8
	local.get 8
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
	local.get 8
	local.get 8
	local.get 8
	local.get 8
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
	f32x4.min
	local.get 7
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
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
	local.tee 1
	local.set 4
	block 
	local.get 3
	i32.const -4
	i32.add 
	local.tee 0
	i32.const 4
	i32.and 
	br_if 0
	f32.const 0x0p0
	local.set 9
	block 
	local.get 1
	f32.load 0
	local.tee 10
	f32.const 0x0p0
	f32.lt 
	br_if 0
	block 
	local.get 10
	f32.const 0x1.8ea082p-9
	f32.lt 
	br_if 0
	f32.const 0x1p0
	local.set 9
	local.get 10
	f32.const 0x1p0
	f32.lt 
	i32.eqz
	br_if 1
	local.get 10
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul 
	f32.const -0x1.c2a5d6p-5
	f32.add 
	local.set 9
	br 1
	end_block
	local.get 10
	f32.const 0x1.9d70a4p3
	f32.mul 
	local.set 9
	end_block
	local.get 1
	i32.const 4
	i32.add 
	local.set 4
	local.get 1
	local.get 9
	f32.store 0
	end_block
	local.get 0
	i32.eqz
	br_if 0
	local.get 1
	local.get 3
	i32.add 
	local.set 1
	loop 
	f32.const 0x0p0
	local.set 9
	f32.const 0x0p0
	local.set 10
	block 
	local.get 4
	f32.load 0
	local.tee 11
	f32.const 0x0p0
	f32.lt 
	br_if 0
	block 
	local.get 11
	f32.const 0x1.8ea082p-9
	f32.lt 
	br_if 0
	f32.const 0x1p0
	local.set 10
	local.get 11
	f32.const 0x1p0
	f32.lt 
	i32.eqz
	br_if 1
	local.get 11
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul 
	f32.const -0x1.c2a5d6p-5
	f32.add 
	local.set 10
	br 1
	end_block
	local.get 11
	f32.const 0x1.9d70a4p3
	f32.mul 
	local.set 10
	end_block
	local.get 4
	local.get 10
	f32.store 0
	block 
	local.get 4
	i32.const 4
	i32.add 
	local.tee 3
	f32.load 0
	local.tee 10
	f32.const 0x0p0
	f32.lt 
	br_if 0
	block 
	local.get 10
	f32.const 0x1.8ea082p-9
	f32.lt 
	br_if 0
	f32.const 0x1p0
	local.set 9
	local.get 10
	f32.const 0x1p0
	f32.lt 
	i32.eqz
	br_if 1
	local.get 10
	f32.const 0x1.aaaaaap-2
	call powf
	f32.const 0x1.0e152ep0
	f32.mul 
	f32.const -0x1.c2a5d6p-5
	f32.add 
	local.set 9
	br 1
	end_block
	local.get 10
	f32.const 0x1.9d70a4p3
	f32.mul 
	local.set 9
	end_block
	local.get 3
	local.get 9
	f32.store 0
	local.get 4
	i32.const 8
	i32.add 
	local.tee 4
	local.get 1
	i32.ne 
	br_if 0
	end_loop
	end_block
	end_function
