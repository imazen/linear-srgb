.section .text.stub_srgb_to_linear_slice,"",@
	.globl	stub_srgb_to_linear_slice
.type	stub_srgb_to_linear_slice,@function
stub_srgb_to_linear_slice:
	.functype	stub_srgb_to_linear_slice (i32, i32) -> ()
	.local  	i32, i32, i32, v128, v128, v128
	local.get 1
	i32.const 3
	i32.and
	local.set 2
	block
	local.get 1
	i32.const 2
	i32.shl
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
	v128.const 0x1p0, 0x1p0, 0x1p0, 0x1p0
	local.tee 5
	v128.const 0x0p0, 0x0p0, 0x0p0, 0x0p0
	local.get 4
	v128.load 0:p2align=2
	local.tee 6
	f32x4.pmax
	f32x4.pmin
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
	local.get 7
	v128.const 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5, 0x1.41e42cp-5
	f32x4.le
	v128.bitselect
	local.get 6
	local.get 5
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
	i32.eqz
	br_if 0
	local.get 0
	local.get 1
	i32.const 536870908
	i32.and
	i32.const 2
	i32.shl
	i32.add
	local.get 2
	i32.const linear_srgb::kernels::float::x4::srgb_to_linear_slice_cfg::tail::<archmage::tokens::generated::wasm::Wasm128Token, 7>
	call linear_srgb::kernels::finish_tail_cold::<archmage::tokens::generated::wasm::Wasm128Token, 4>
	end_block
	end_function
