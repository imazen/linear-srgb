.section .text.stub_gamma_to_linear_slice,"",@
	.globl	stub_gamma_to_linear_slice
.type	stub_gamma_to_linear_slice,@function
stub_gamma_to_linear_slice:
	.functype	stub_gamma_to_linear_slice (i32, i32, f32) -> ()
	.local  	i32, i32, i32, i32, i32, i32, i32, v128, v128, v128
	global.get __stack_pointer
	i32.const 272
	i32.sub
	local.tee 3
	global.set __stack_pointer
	local.get 3
	local.get 2
	f32.store 8
	local.get 1
	i32.const 4
	i32.shr_u
	local.set 4
	local.get 0
	local.get 1
	i32.const 536870896
	i32.and
	i32.const 2
	i32.shl
	i32.add
	local.set 5
	i32.const 0
	local.set 6
	block
	local.get 1
	i32.const 15
	i32.and
	local.tee 7
	i32.eqz
	br_if 0
	block
	i32.const 64
	local.get 7
	i32.const 2
	i32.shl
	local.tee 1
	i32.sub
	local.tee 6
	i32.eqz
	br_if 0
	local.get 3
	i32.const 208
	i32.add
	local.get 1
	i32.add
	i32.const 0
	local.get 6
	memory.fill 0
	end_block
	block
	local.get 1
	i32.eqz
	br_if 0
	local.get 3
	i32.const 208
	i32.add
	local.get 5
	local.get 1
	memory.copy 0, 0
	end_block
	local.get 3
	local.get 3
	v128.load 256:p2align=2
	v128.store 64:p2align=2
	local.get 3
	local.get 3
	v128.load 240:p2align=2
	v128.store 48:p2align=2
	local.get 3
	local.get 3
	v128.load 224:p2align=2
	v128.store 32:p2align=2
	local.get 3
	local.get 3
	v128.load 208:p2align=2
	v128.store 16:p2align=2
	i32.const 1
	local.set 6
	end_block
	local.get 3
	local.get 6
	i32.store 12
	local.get 3
	i32.const 12
	i32.add
	i32.const 4
	i32.add
	local.tee 8
	i32.const 0
	local.get 7
	i32.select
	local.set 6
	local.get 0
	local.get 4
	i32.const 6
	i32.shl
	i32.add
	local.set 9
	block
	loop
	block
	block
	local.get 0
	i32.eqz
	br_if 0
	local.get 0
	local.get 9
	i32.eq
	br_if 0
	local.get 0
	local.set 1
	local.get 0
	i32.const 64
	i32.add
	local.set 0
	local.get 6
	local.set 4
	br 1
	end_block
	i32.const 0
	local.set 0
	local.get 6
	local.set 1
	i32.const 0
	local.set 4
	local.get 6
	i32.eqz
	br_if 2
	end_block
	local.get 1
	v128.load 0:p2align=2
	local.set 10
	local.get 1
	v128.load 16:p2align=2
	local.set 11
	local.get 1
	v128.load 32:p2align=2
	local.set 12
	local.get 3
	local.get 1
	v128.load 48:p2align=2
	v128.store 192:p2align=3
	local.get 3
	local.get 12
	v128.store 176:p2align=3
	local.get 3
	local.get 11
	v128.store 160:p2align=3
	local.get 3
	local.get 10
	v128.store 144:p2align=3
	local.get 3
	i32.const 80
	i32.add
	local.get 3
	i32.const 8
	i32.add
	local.get 3
	i32.const 144
	i32.add
	call linear_srgb::target_features::x16::gamma_to_linear_slice_wasm128::{closure#0}
	local.get 1
	local.get 3
	v128.load 128:p2align=2
	v128.store 48:p2align=2
	local.get 1
	local.get 3
	v128.load 112:p2align=2
	v128.store 32:p2align=2
	local.get 1
	local.get 3
	v128.load 96:p2align=2
	v128.store 16:p2align=2
	local.get 1
	local.get 3
	v128.load 80:p2align=2
	v128.store 0:p2align=2
	local.get 4
	local.set 6
	br 0
	end_loop
	end_block
	block
	local.get 3
	i32.load 12
	i32.const 1
	i32.ne
	br_if 0
	local.get 7
	i32.const 2
	i32.shl
	local.tee 1
	i32.eqz
	br_if 0
	local.get 5
	local.get 8
	local.get 1
	memory.copy 0, 0
	end_block
	local.get 3
	i32.const 272
	i32.add
	global.set __stack_pointer
	end_function
