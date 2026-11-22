.section .text.stub_unpremultiply_linear_to_srgb_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_srgb_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_srgb_rgba_slice,@function
stub_unpremultiply_linear_to_srgb_rgba_slice:
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
	b.eq .LBB25_7
	cmp w8, #2
	b.ne .LBB25_6
.LBB25_2:
	lsl x8, x1, #2
	ands x8, x8, #0x7ffffffffffffff0
	b.eq .LBB25_5
	mov w10, #47186
	mov w11, #25800
	mov w9, #981467136
	movk w10, #16718, lsl #16
	movk w11, #16863, lsl #16
	dup v1.4s, w9
	dup v2.4s, w10
	dup v3.4s, w11
	mov w9, #5570
	mov w10, #10701
	mov w11, #14394
	movk w9, #16968, lsl #16
	movk w10, #16697, lsl #16
	movk w11, #15807, lsl #16
	dup v4.4s, w9
	dup v5.4s, w10
	dup v6.4s, w11
	mov w9, #15682
	mov w10, #15285
	mov w11, #7182
	movk w9, #48222, lsl #16
	movk w10, #16906, lsl #16
	movk w11, #16947, lsl #16
	dup v7.4s, w9
	dup v16.4s, w10
	dup v17.4s, w11
	mov w9, #64401
	mov w10, #55785
	mov w11, #20545
	fmov v0.4s, #1.00000000
	movk w9, #16655, lsl #16
	movk w10, #16006, lsl #16
	movk w11, #15175, lsl #16
	dup v18.4s, w9
	dup v19.4s, w10
	dup v20.4s, w11
.LBB25_4:
	ldr q21, [x0]
	mov v25.16b, v4.16b
	mov v27.16b, v5.16b
	mov v28.16b, v17.16b
	subs x8, x8, #16
	dup v22.4s, v21.s[3]
	fdiv v23.4s, v0.4s, v22.4s
	fcmgt v22.4s, v22.4s, v1.4s
	and v22.16b, v22.16b, v23.16b
	fmul v22.4s, v21.4s, v22.4s
	fcmgt v23.4s, v22.4s, #0.0
	and v23.16b, v23.16b, v22.16b
	fcmge v22.4s, v22.4s, v0.4s
	fminnm v23.4s, v23.4s, v0.4s
	fsqrt v24.4s, v23.4s
	fmla v25.4s, v3.4s, v24.4s
	fadd v26.4s, v24.4s, v16.4s
	fmla v27.4s, v24.4s, v25.4s
	fmla v28.4s, v24.4s, v26.4s
	mov v25.16b, v6.16b
	mov v26.16b, v18.16b
	fmla v25.4s, v24.4s, v27.4s
	fmla v26.4s, v24.4s, v28.4s
	mov v27.16b, v7.16b
	mov v28.16b, v19.16b
	fmla v27.4s, v24.4s, v25.4s
	fmla v28.4s, v24.4s, v26.4s
	fmul v25.4s, v23.4s, v2.4s
	fcmge v23.4s, v20.4s, v23.4s
	fdiv v24.4s, v27.4s, v28.4s
	bsl v23.16b, v25.16b, v24.16b
	bsl v22.16b, v0.16b, v23.16b
	mov v22.s[3], v21.s[3]
	str q22, [x0], #16
	b.ne .LBB25_4
.LBB25_5:
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB25_6:
	.cfi_restore_state
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB25_2
.LBB25_7:
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	b linear_srgb::simd::unpremultiply_linear_to_srgb_rgba_slice_tier_scalar
