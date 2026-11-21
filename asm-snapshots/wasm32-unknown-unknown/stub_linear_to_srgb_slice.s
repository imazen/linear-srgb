.section .text.stub_linear_to_srgb_slice,"",@
	.globl	stub_linear_to_srgb_slice
.type	stub_linear_to_srgb_slice,@function
stub_linear_to_srgb_slice:
	.functype	stub_linear_to_srgb_slice (i32, i32) -> ()
	.local  	i32, i32, i32, i32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, i32, v128, v128, v128
	global.get __stack_pointer
	i32.const 128
	i32.sub
	local.set 2
	local.get 1
	i32.const 15
	i32.and
	local.set 3
	block
	local.get 1
	i32.const 2
	i32.shl
	i32.const 2147483584
	i32.and
	local.tee 4
	i32.eqz
	br_if 0
	local.get 0
	local.get 4
	i32.add
	local.set 5
	local.get 0
	local.set 4
	loop
	local.get 4
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	local.tee 6
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 7
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 8
	local.get 4
	v128.load 0:p2align=2
	local.tee 9
	f32x4.pmax
	f32x4.pmin
	local.tee 10
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	local.tee 11
	f32x4.mul
	local.get 10
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	v128.const 0x1.bec99p4, 0x1.bec99p4, 0x1.bec99p4, 0x1.bec99p4
	local.tee 13
	f32x4.mul
	v128.const 0x1.902b84p5, 0x1.902b84p5, 0x1.902b84p5, 0x1.902b84p5
	local.tee 14
	f32x4.add
	f32x4.mul
	v128.const 0x1.72539ap3, 0x1.72539ap3, 0x1.72539ap3, 0x1.72539ap3
	local.tee 15
	f32x4.add
	f32x4.mul
	v128.const 0x1.7e7074p-4, 0x1.7e7074p-4, 0x1.7e7074p-4, 0x1.7e7074p-4
	local.tee 16
	f32x4.add
	f32x4.mul
	v128.const -0x1.bc7a84p-7, -0x1.bc7a84p-7, -0x1.bc7a84p-7, -0x1.bc7a84p-7
	local.tee 17
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	v128.const 0x1.14776ap5, 0x1.14776ap5, 0x1.14776ap5, 0x1.14776ap5
	local.tee 18
	f32x4.add
	f32x4.mul
	v128.const 0x1.66381cp5, 0x1.66381cp5, 0x1.66381cp5, 0x1.66381cp5
	local.tee 19
	f32x4.add
	f32x4.mul
	v128.const 0x1.1ff722p3, 0x1.1ff722p3, 0x1.1ff722p3, 0x1.1ff722p3
	local.tee 20
	f32x4.add
	f32x4.mul
	v128.const 0x1.0db3d2p-2, 0x1.0db3d2p-2, 0x1.0db3d2p-2, 0x1.0db3d2p-2
	local.tee 21
	f32x4.add
	f32x4.div
	local.get 10
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	local.tee 22
	f32x4.le
	v128.bitselect
	local.get 9
	local.get 7
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	i32.const 48
	i32.add
	local.tee 23
	local.get 6
	local.get 7
	local.get 8
	local.get 23
	v128.load 0:p2align=2
	local.tee 9
	f32x4.pmax
	f32x4.pmin
	local.tee 10
	local.get 11
	f32x4.mul
	local.get 10
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	local.get 13
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 18
	f32x4.add
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.div
	local.get 10
	local.get 22
	f32x4.le
	v128.bitselect
	local.get 9
	local.get 7
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	i32.const 32
	i32.add
	local.tee 23
	local.get 6
	local.get 7
	local.get 8
	local.get 23
	v128.load 0:p2align=2
	local.tee 9
	f32x4.pmax
	f32x4.pmin
	local.tee 10
	local.get 11
	f32x4.mul
	local.get 10
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	local.get 13
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 18
	f32x4.add
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.div
	local.get 10
	local.get 22
	f32x4.le
	v128.bitselect
	local.get 9
	local.get 7
	f32x4.ge
	v128.bitselect
	v128.store 0:p2align=2
	local.get 4
	i32.const 16
	i32.add
	local.tee 23
	local.get 6
	local.get 7
	local.get 8
	local.get 23
	v128.load 0:p2align=2
	local.tee 10
	f32x4.pmax
	f32x4.pmin
	local.tee 8
	local.get 11
	f32x4.mul
	local.get 8
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	local.get 13
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 18
	f32x4.add
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.div
	local.get 8
	local.get 22
	f32x4.le
	v128.bitselect
	local.get 10
	local.get 7
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
	i32.eqz
	br_if 0
	local.get 1
	i32.const 536870896
	i32.and
	i32.const 2
	i32.shl
	local.set 23
	block
	i32.const 64
	local.get 3
	i32.const 2
	i32.shl
	local.tee 4
	i32.sub
	local.tee 5
	i32.eqz
	br_if 0
	local.get 2
	local.get 4
	i32.add
	i32.const 0
	local.get 5
	memory.fill 0
	end_block
	local.get 0
	local.get 23
	i32.add
	local.set 23
	block
	local.get 4
	i32.eqz
	local.tee 5
	br_if 0
	local.get 2
	local.get 23
	local.get 4
	memory.copy 0, 0
	end_block
	local.get 2
	v128.load 0
	local.set 10
	local.get 2
	v128.load 16
	local.set 9
	local.get 2
	v128.load 32
	local.set 24
	local.get 2
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	local.tee 6
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 7
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 8
	local.get 2
	v128.load 48
	local.tee 25
	f32x4.pmax
	f32x4.pmin
	local.tee 26
	v128.const 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3, 0x1.9d70a4p3
	local.tee 11
	f32x4.mul
	local.get 26
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	v128.const 0x1.bec99p4, 0x1.bec99p4, 0x1.bec99p4, 0x1.bec99p4
	local.tee 13
	f32x4.mul
	v128.const 0x1.902b84p5, 0x1.902b84p5, 0x1.902b84p5, 0x1.902b84p5
	local.tee 14
	f32x4.add
	f32x4.mul
	v128.const 0x1.72539ap3, 0x1.72539ap3, 0x1.72539ap3, 0x1.72539ap3
	local.tee 15
	f32x4.add
	f32x4.mul
	v128.const 0x1.7e7074p-4, 0x1.7e7074p-4, 0x1.7e7074p-4, 0x1.7e7074p-4
	local.tee 16
	f32x4.add
	f32x4.mul
	v128.const -0x1.bc7a84p-7, -0x1.bc7a84p-7, -0x1.bc7a84p-7, -0x1.bc7a84p-7
	local.tee 17
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	v128.const 0x1.14776ap5, 0x1.14776ap5, 0x1.14776ap5, 0x1.14776ap5
	local.tee 18
	f32x4.add
	f32x4.mul
	v128.const 0x1.66381cp5, 0x1.66381cp5, 0x1.66381cp5, 0x1.66381cp5
	local.tee 19
	f32x4.add
	f32x4.mul
	v128.const 0x1.1ff722p3, 0x1.1ff722p3, 0x1.1ff722p3, 0x1.1ff722p3
	local.tee 20
	f32x4.add
	f32x4.mul
	v128.const 0x1.0db3d2p-2, 0x1.0db3d2p-2, 0x1.0db3d2p-2, 0x1.0db3d2p-2
	local.tee 21
	f32x4.add
	f32x4.div
	local.get 26
	v128.const 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9, 0x1.8ea082p-9
	local.tee 22
	f32x4.le
	v128.bitselect
	local.get 25
	local.get 7
	f32x4.ge
	v128.bitselect
	v128.store 112
	local.get 2
	local.get 6
	local.get 7
	local.get 8
	local.get 24
	f32x4.pmax
	f32x4.pmin
	local.tee 26
	local.get 11
	f32x4.mul
	local.get 26
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	local.get 13
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 18
	f32x4.add
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.div
	local.get 26
	local.get 22
	f32x4.le
	v128.bitselect
	local.get 24
	local.get 7
	f32x4.ge
	v128.bitselect
	v128.store 96
	local.get 2
	local.get 6
	local.get 7
	local.get 8
	local.get 9
	f32x4.pmax
	f32x4.pmin
	local.tee 24
	local.get 11
	f32x4.mul
	local.get 24
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	local.get 13
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 18
	f32x4.add
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.div
	local.get 24
	local.get 22
	f32x4.le
	v128.bitselect
	local.get 9
	local.get 7
	f32x4.ge
	v128.bitselect
	v128.store 80
	local.get 2
	local.get 6
	local.get 7
	local.get 8
	local.get 10
	f32x4.pmax
	f32x4.pmin
	local.tee 8
	local.get 11
	f32x4.mul
	local.get 8
	f32x4.sqrt
	local.tee 12
	local.get 12
	local.get 12
	local.get 12
	local.get 13
	f32x4.mul
	local.get 14
	f32x4.add
	f32x4.mul
	local.get 15
	f32x4.add
	f32x4.mul
	local.get 16
	f32x4.add
	f32x4.mul
	local.get 17
	f32x4.add
	local.get 12
	local.get 12
	local.get 12
	local.get 12
	local.get 18
	f32x4.add
	f32x4.mul
	local.get 19
	f32x4.add
	f32x4.mul
	local.get 20
	f32x4.add
	f32x4.mul
	local.get 21
	f32x4.add
	f32x4.div
	local.get 8
	local.get 22
	f32x4.le
	v128.bitselect
	local.get 10
	local.get 7
	f32x4.ge
	v128.bitselect
	v128.store 64
	local.get 5
	br_if 0
	local.get 23
	local.get 2
	i32.const 64
	i32.add
	local.get 4
	memory.copy 0, 0
	end_block
	end_function
