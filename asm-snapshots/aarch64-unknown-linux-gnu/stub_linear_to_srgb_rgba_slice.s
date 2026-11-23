.section .text.stub_linear_to_srgb_rgba_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_rgba_slice
	.p2align	2
.type	stub_linear_to_srgb_rgba_slice,@function
stub_linear_to_srgb_rgba_slice:
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
	b.eq .LBB17_7
	cmp w8, #2
	b.ne .LBB17_6
.LBB17_2:
	lsl x8, x1, #2
	ands x8, x8, #0x7ffffffffffffff0
	b.eq .LBB17_5
	mov w9, #47186
	mov w10, #25800
	mov w11, #5570
	movk w9, #16718, lsl #16
	movk w10, #16863, lsl #16
	movk w11, #16968, lsl #16
	dup v1.4s, w9
	dup v2.4s, w10
	mov w9, #10701
	mov w10, #14394
	movk w9, #16697, lsl #16
	dup v3.4s, w11
	movk w10, #15807, lsl #16
	dup v4.4s, w9
	mov w9, #15682
	dup v5.4s, w10
	mov w10, #15285
	mov w11, #7182
	movk w9, #48222, lsl #16
	movk w10, #16906, lsl #16
	movk w11, #16947, lsl #16
	dup v6.4s, w9
	dup v7.4s, w10
	dup v16.4s, w11
	mov w9, #64401
	mov w10, #55785
	mov w11, #20545
	fmov v0.4s, #1.00000000
	movk w9, #16655, lsl #16
	movk w10, #16006, lsl #16
	movk w11, #15175, lsl #16
	dup v17.4s, w9
	dup v18.4s, w10
	dup v19.4s, w11
.LBB17_4:
	ldr q20, [x0]
	mov v23.16b, v3.16b
	mov v25.16b, v4.16b
	mov v26.16b, v16.16b
	subs x8, x8, #16
	fcmgt v21.4s, v20.4s, #0.0
	and v21.16b, v21.16b, v20.16b
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
	fcmge v22.4s, v20.4s, v0.4s
	bit v21.16b, v0.16b, v22.16b
	mov v21.s[3], v20.s[3]
	str q21, [x0], #16
	b.ne .LBB17_4
.LBB17_5:
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB17_6:
	.cfi_restore_state
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB17_2
.LBB17_7:
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	b linear_srgb::simd::linear_to_srgb_rgba_slice_tier_scalar
