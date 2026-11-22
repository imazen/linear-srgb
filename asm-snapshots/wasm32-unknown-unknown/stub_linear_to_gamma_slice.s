.section .text.stub_linear_to_gamma_slice,"",@
	.globl	stub_linear_to_gamma_slice
.type	stub_linear_to_gamma_slice,@function
stub_linear_to_gamma_slice:
	.functype	stub_linear_to_gamma_slice (i32, i32, f32) -> ()
	.local  	i32, i32, i32, i32, f32, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128, v128
	block
	block
	f32.const 0x1p0
	local.get 2
	f32.div
	local.tee 2
	f32.const 0x0p0
	f32.ge
	i32.eqz
	br_if 0
	local.get 2
	f32.const 0x1p2
	f32.le
	br_if 1
	end_block
	local.get 0
	local.get 1
	local.get 2
	call linear_srgb::tf::math::gamma_outside_fit_slice
	return
	end_block
	local.get 1
	i32.const 3
	i32.and
	local.set 3
	local.get 1
	i32.const 2
	i32.shl
	local.tee 4
	i32.const 2147483632
	i32.and
	local.set 5
	local.get 0
	local.get 1
	i32.const 536870908
	i32.and
	i32.const 2
	i32.shl
	i32.add
	local.set 6
	block
	block
	local.get 2
	f32.const 0x0p0
	f32.eq
	br_if 0
	block
	local.get 5
	i32.eqz
	br_if 0
	f32.const 0x0p0
	f32.const 0x1p2
	local.get 2
	local.get 2
	f32.const 0x1p2
	f32.gt
	f32.select
	local.tee 7
	local.get 7
	i32.reinterpret_f32
	i32.const -1
	i32.add
	i32.const 8388607
	i32.lt_u
	local.tee 1
	f32.select
	f32x4.splat
	local.set 8
	f32.const 0x1p0
	f32.const 0x0p0
	local.get 1
	f32.select
	f32x4.splat
	local.set 9
	loop
	local.get 0
	local.get 0
	v128.load 0:p2align=2
	local.tee 10
	local.get 9
	v128.const 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127, 0, 0, 128, 127
	v128.const 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	local.tee 11
	local.get 8
	v128.const 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
	local.get 10
	local.get 10
	v128.const 0x1p-126, 0x1p-126, 0x1p-126, 0x1p-126
	f32x4.ge
	v128.and
	local.tee 12
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	local.tee 13
	local.get 12
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 14
	f32x4.lt
	v128.bitselect
	local.tee 12
	local.get 12
	v128.const 0, 0, 0, 0
	i32x4.eq
	local.tee 15
	v128.bitselect
	v128.const 4913933, 4913933, 4913933, 4913933
	i32x4.add
	local.tee 16
	v128.const 8388607, 8388607, 8388607, 8388607
	v128.and
	v128.const 1060439283, 1060439283, 1060439283, 1060439283
	i32x4.add
	local.tee 12
	v128.const -0x1p0, -0x1p0, -0x1p0, -0x1p0
	f32x4.add
	local.get 12
	local.get 14
	f32x4.add
	f32x4.div
	local.tee 12
	local.get 12
	local.get 12
	f32x4.mul
	local.tee 12
	local.get 12
	local.get 12
	v128.const 0x1.bcd67ep-2, 0x1.bcd67ep-2, 0x1.bcd67ep-2, 0x1.bcd67ep-2
	f32x4.mul
	v128.const 0x1.276932p-1, 0x1.276932p-1, 0x1.276932p-1, 0x1.276932p-1
	f32x4.add
	f32x4.mul
	v128.const 0x1.ec7126p-1, 0x1.ec7126p-1, 0x1.ec7126p-1, 0x1.ec7126p-1
	f32x4.add
	f32x4.mul
	v128.const 0x1.715476p1, 0x1.715476p1, 0x1.715476p1, 0x1.715476p1
	f32x4.add
	f32x4.mul
	local.get 16
	i32.const 23
	i32x4.shr_s
	v128.const -127, -127, -127, -127
	i32x4.add
	f32x4.convert_i32x4_s
	f32x4.add
	f32x4.mul
	local.tee 12
	local.get 12
	v128.const -1023672320, -1023672320, -1023672320, -1023672320
	local.get 12
	v128.const -0x1.f8p6, -0x1.f8p6, -0x1.f8p6, -0x1.f8p6
	local.tee 17
	f32x4.gt
	v128.bitselect
	local.get 11
	v128.bitselect
	local.tee 16
	local.get 16
	v128.const 1124073472, 1124073472, 1124073472, 1124073472
	local.get 16
	v128.const 0x1p7, 0x1p7, 0x1p7, 0x1p7
	local.tee 18
	f32x4.lt
	v128.bitselect
	local.get 11
	v128.bitselect
	local.tee 16
	local.get 16
	f32x4.nearest
	local.tee 16
	local.get 16
	v128.const 1123942400, 1123942400, 1123942400, 1123942400
	local.get 16
	v128.const 0x1.fcp6, 0x1.fcp6, 0x1.fcp6, 0x1.fcp6
	f32x4.lt
	v128.bitselect
	local.get 11
	v128.bitselect
	local.tee 16
	f32x4.sub
	local.tee 11
	local.get 11
	local.get 11
	local.get 11
	local.get 11
	local.get 11
	v128.const 0x1.43f274p-13, 0x1.43f274p-13, 0x1.43f274p-13, 0x1.43f274p-13
	f32x4.mul
	v128.const 0x1.5d88f2p-10, 0x1.5d88f2p-10, 0x1.5d88f2p-10, 0x1.5d88f2p-10
	f32x4.add
	f32x4.mul
	v128.const 0x1.3b2a18p-7, 0x1.3b2a18p-7, 0x1.3b2a18p-7, 0x1.3b2a18p-7
	f32x4.add
	f32x4.mul
	v128.const 0x1.c6b178p-5, 0x1.c6b178p-5, 0x1.c6b178p-5, 0x1.c6b178p-5
	f32x4.add
	f32x4.mul
	v128.const 0x1.ebfbdap-3, 0x1.ebfbdap-3, 0x1.ebfbdap-3, 0x1.ebfbdap-3
	f32x4.add
	f32x4.mul
	v128.const 0x1.62e43p-1, 0x1.62e43p-1, 0x1.62e43p-1, 0x1.62e43p-1
	f32x4.add
	f32x4.mul
	local.get 14
	f32x4.add
	local.get 16
	f32x4.nearest
	i32x4.trunc_sat_f32x4_s
	i32.const 23
	i32x4.shl
	local.get 13
	i32x4.add
	f32x4.mul
	local.get 12
	local.get 17
	f32x4.lt
	v128.bitselect
	local.get 12
	local.get 18
	f32x4.ge
	v128.bitselect
	local.get 15
	v128.bitselect
	local.get 10
	local.get 10
	f32x4.ne
	v128.bitselect
	v128.store 0:p2align=2
	local.get 0
	i32.const 16
	i32.add
	local.set 0
	local.get 5
	i32.const -16
	i32.add
	local.tee 5
	br_if 0
	end_loop
	end_block
	local.get 3
	i32.eqz
	br_if 1
	local.get 6
	local.get 3
	local.get 2
	call linear_srgb::kernels::finish_tail_cold::<(archmage::tokens::generated::wasm::Wasm128Token, f32), 4>
	return
	end_block
	block
	local.get 5
	i32.eqz
	br_if 0
	local.get 0
	local.set 1
	block
	local.get 4
	i32.const -16
	i32.add
	local.tee 4
	i32.const 16
	i32.and
	br_if 0
	local.get 0
	local.get 0
	v128.load 0:p2align=2
	local.tee 11
	local.get 11
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	f32x4.gt
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	v128.and
	local.get 11
	local.get 11
	f32x4.ne
	v128.bitselect
	v128.store 0:p2align=2
	local.get 0
	i32.const 16
	i32.add
	local.set 1
	end_block
	local.get 4
	i32.const 16
	i32.lt_u
	br_if 0
	local.get 0
	local.get 5
	i32.add
	local.set 5
	loop
	local.get 1
	local.get 1
	v128.load 0:p2align=2
	local.tee 11
	local.get 11
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.tee 10
	f32x4.gt
	v128.const 1065353216, 1065353216, 1065353216, 1065353216
	local.tee 12
	v128.and
	local.get 11
	local.get 11
	f32x4.ne
	v128.bitselect
	v128.store 0:p2align=2
	local.get 1
	i32.const 16
	i32.add
	local.tee 0
	local.get 0
	v128.load 0:p2align=2
	local.tee 11
	local.get 11
	local.get 10
	f32x4.gt
	local.get 12
	v128.and
	local.get 11
	local.get 11
	f32x4.ne
	v128.bitselect
	v128.store 0:p2align=2
	local.get 1
	i32.const 32
	i32.add
	local.tee 1
	local.get 5
	i32.ne
	br_if 0
	end_loop
	end_block
	local.get 3
	i32.eqz
	br_if 0
	local.get 6
	local.get 3
	i32.const linear_srgb::tf::math::gamma_slice_x4_cfg::tail_zero::<archmage::tokens::generated::wasm::Wasm128Token>
	call linear_srgb::kernels::finish_tail_cold::<archmage::tokens::generated::wasm::Wasm128Token, 4>
	end_block
	end_function
