.section .text.stub_linear_to_srgb_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_slice
	.p2align	2
.type	stub_linear_to_srgb_slice,@function
stub_linear_to_srgb_slice:
	.cfi_startproc
	stp x29, x30, [sp, #-32]!
	.cfi_def_cfa_offset 32
	stp x20, x19, [sp, #16]
	mov x29, sp
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	.cfi_remember_state
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB18_8
	cmp w8, #2
	b.ne .LBB18_7
.LBB18_2:
	lsl x8, x1, #2
	ands x9, x8, #0x7ffffffffffffff0
	and x8, x1, #0x3
	b.eq .LBB18_5
	mov w10, #47186
	mov w11, #25800
	mov w12, #5570
	movk w10, #16718, lsl #16
	movk w11, #16863, lsl #16
	movk w12, #16968, lsl #16
	dup v1.4s, w10
	dup v2.4s, w11
	mov w10, #10701
	mov w11, #14394
	movk w10, #16697, lsl #16
	dup v3.4s, w12
	movk w11, #15807, lsl #16
	dup v4.4s, w10
	mov w10, #15682
	dup v5.4s, w11
	mov w11, #15285
	mov w12, #7182
	movk w10, #48222, lsl #16
	movk w11, #16906, lsl #16
	movk w12, #16947, lsl #16
	dup v6.4s, w10
	dup v7.4s, w11
	dup v16.4s, w12
	mov w10, #64401
	mov w11, #55785
	mov w12, #20545
	fmov v0.4s, #1.00000000
	movk w10, #16655, lsl #16
	movk w11, #16006, lsl #16
	movk w12, #15175, lsl #16
	dup v17.4s, w10
	dup v18.4s, w11
	dup v19.4s, w12
	mov x10, x0
.LBB18_4:
	ldr q20, [x10]
	mov v23.16b, v3.16b
	mov v25.16b, v4.16b
	mov v26.16b, v16.16b
	subs x9, x9, #16
	fcmgt v21.4s, v20.4s, #0.0
	and v21.16b, v21.16b, v20.16b
	fcmge v20.4s, v20.4s, v0.4s
	fminnm v21.4s, v21.4s, v0.4s
	fsqrt v22.4s, v21.4s
	fmla v23.4s, v2.4s, v22.4s
	fadd v24.4s, v22.4s, v7.4s
	fmla v25.4s, v22.4s, v23.4s
	fmla v26.4s, v22.4s, v24.4s
	mov v23.16b, v5.16b
	mov v24.16b, v17.16b
	fmla v23.4s, v22.4s, v25.4s
	fmla v24.4s, v22.4s, v26.4s
	mov v25.16b, v6.16b
	mov v26.16b, v18.16b
	fmla v25.4s, v22.4s, v23.4s
	fmla v26.4s, v22.4s, v24.4s
	fmul v23.4s, v21.4s, v1.4s
	fcmge v21.4s, v19.4s, v21.4s
	fdiv v22.4s, v25.4s, v26.4s
	bsl v21.16b, v23.16b, v22.16b
	bsl v20.16b, v0.16b, v21.16b
	str q20, [x10], #16
	b.ne .LBB18_4
.LBB18_5:
	cbnz x8, .LBB18_9
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB18_7:
	.cfi_restore_state
	.cfi_remember_state
	mov x20, x1
	mov x19, x0
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB18_2
.LBB18_8:
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	b linear_srgb::simd::linear_to_srgb_slice_tier_scalar_t
.LBB18_9:
	.cfi_restore_state
	and x9, x1, #0x1ffffffffffffffc
	adrp x2, linear_srgb::kernels::float::x4::linear_to_srgb_slice_cfg::tail::<archmage::tokens::generated::arm::NeonToken, 7>
	add x2, x2, :lo12:linear_srgb::kernels::float::x4::linear_to_srgb_slice_cfg::tail::<archmage::tokens::generated::arm::NeonToken, 7>
	add x0, x0, x9, lsl #2
	mov x1, x8
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	b linear_srgb::kernels::finish_tail_cold::<archmage::tokens::generated::arm::NeonToken, 4>
