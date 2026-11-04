.section .text.stub_unpremultiply_linear_to_srgb_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_srgb_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_srgb_rgba_slice,@function
stub_unpremultiply_linear_to_srgb_rgba_slice:
	.cfi_startproc
	sub sp, sp, #464
	.cfi_def_cfa_offset 464
	stp d15, d14, [sp, #336]
	stp d13, d12, [sp, #352]
	stp d11, d10, [sp, #368]
	stp d9, d8, [sp, #384]
	stp x29, x30, [sp, #400]
	stp x28, x23, [sp, #416]
	stp x22, x21, [sp, #432]
	stp x20, x19, [sp, #448]
	add x29, sp, #400
	.cfi_def_cfa w29, 64
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w28, -48
	.cfi_offset w30, -56
	.cfi_offset w29, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	.cfi_offset b10, -88
	.cfi_offset b11, -96
	.cfi_offset b12, -104
	.cfi_offset b13, -112
	.cfi_offset b14, -120
	.cfi_offset b15, -128
	.cfi_remember_state
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB16_15
	cmp w8, #2
	b.ne .LBB16_14
.LBB16_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x21, x1, #0xc
	lsr x22, x1, #4
	add x19, x0, x8, lsl #2
	lsl x20, x21, #2
	b.eq .LBB16_4
	mov w8, #64
	mov x23, x0
	mov w1, wzr
	sub x2, x8, x20
	sub x8, x29, #128
	add x0, x8, x20
	bl memset
	sub x0, x29, #128
	mov x1, x19
	mov x2, x20
	bl memcpy
	ldp q0, q1, [x29, #-128]
	mov x0, x23
	mov w9, #1
	stp q0, q1, [x29, #-192]
	ldp q0, q2, [x29, #-96]
	stp q0, q2, [x29, #-160]
	b .LBB16_5
.LBB16_4:
	mov w9, wzr
.LBB16_5:
	mov w8, #47186
	mov w10, #25800
	fmov v17.4s, #1.00000000
	movk w8, #16718, lsl #16
	movk w10, #16863, lsl #16
	cmp x21, #0
	dup v1.4s, w8
	dup v0.4s, w10
	mov w8, #5570
	movk w8, #16968, lsl #16
	mov w10, #15682
	stur w9, [x29, #-196]
	movk w10, #48222, lsl #16
	stp q0, q1, [sp, #160]
	dup v1.4s, w8
	mov w8, #10701
	movk w8, #16697, lsl #16
	ldr q3, [sp, #160]
	dup v0.4s, w8
	mov w8, #14394
	movk w8, #15807, lsl #16
	stp q0, q1, [sp, #128]
	dup v1.4s, w8
	dup v0.4s, w10
	mov w10, #15285
	add x8, x0, x22, lsl #6
	movk w10, #16906, lsl #16
	stp q0, q1, [sp, #96]
	dup v1.4s, w10
	mov w10, #7182
	movk w10, #16947, lsl #16
	dup v0.4s, w10
	mov w10, #64401
	movk w10, #16655, lsl #16
	stp q0, q1, [sp, #64]
	dup v1.4s, w10
	mov w10, #55785
	movk w10, #16006, lsl #16
	dup v0.4s, w10
	mov w10, #20545
	movk w10, #15175, lsl #16
	stp q0, q1, [sp, #32]
	dup v0.4s, w10
	sub x10, x29, #196
	add x1, x10, #4
	mov w10, #981467136
	csel x9, xzr, x1, eq
	str w10, [sp, #12]
	str q0, [sp, #16]
	b .LBB16_8
.LBB16_6:
	add x10, x0, #64
.LBB16_7:
	fmov s0, #1.00000000
	ldr s24, [x0, #12]
	ldr s1, [sp, #12]
	movi d5, #0000000000000000
	fmov v27.4s, #1.00000000
	ldr q29, [x0]
	fcmp s24, s1
	movi v7.2d, #0000000000000000
	fmov v9.4s, #1.00000000
	fdiv s25, s0, s24
	ldp q6, q4, [sp, #128]
	ldp q23, q2, [sp, #32]
	mov v11.16b, v4.16b
	mov v14.16b, v6.16b
	mov v18.16b, v2.16b
	mov v20.16b, v23.16b
	fcsel s26, s25, s5, gt
	ldr s25, [x0, #28]
	fdiv s28, s0, s25
	fcmp s25, s1
	mov v27.s[0], v26.s[0]
	mov v27.s[1], v26.s[0]
	mov v27.s[2], v26.s[0]
	ldr s26, [x0, #44]
	fdiv s31, s0, s26
	fmul v27.4s, v29.4s, v27.4s
	fcsel s28, s28, s5, gt
	fcmp s26, s1
	mov v9.s[0], v28.s[0]
	fmax v29.4s, v27.4s, v7.4s
	mov v9.s[1], v28.s[0]
	fmin v30.4s, v29.4s, v17.4s
	ldr q29, [x0, #16]
	mov v9.s[2], v28.s[0]
	ldr s28, [x0, #60]
	fcsel s12, s31, s5, gt
	fsqrt v8.4s, v30.4s
	fcmp s28, s1
	ldp q22, q1, [sp, #96]
	fmul v29.4s, v29.4s, v9.4s
	fmov v9.4s, #1.00000000
	mov v9.s[0], v12.s[0]
	fmax v31.4s, v29.4s, v7.4s
	fdiv s10, s0, s28
	ldp q21, q0, [sp, #64]
	mov v9.s[1], v12.s[0]
	fmin v31.4s, v31.4s, v17.4s
	fmla v11.4s, v3.4s, v8.4s
	fadd v13.4s, v8.4s, v0.4s
	mov v15.16b, v21.16b
	mov v16.16b, v21.16b
	mov v9.s[2], v12.s[0]
	mov v12.16b, v22.16b
	fmla v14.4s, v8.4s, v11.4s
	fmla v15.4s, v8.4s, v13.4s
	fmov v11.4s, #1.00000000
	mov v13.16b, v1.16b
	fsqrt v19.4s, v31.4s
	fmla v13.4s, v8.4s, v14.4s
	fmla v18.4s, v8.4s, v15.4s
	mov v14.16b, v23.16b
	ldr q15, [x0, #32]
	fcsel s10, s10, s5, gt
	mov v5.16b, v22.16b
	fmla v12.4s, v8.4s, v13.4s
	fmla v14.4s, v8.4s, v18.4s
	fmul v8.4s, v15.4s, v9.4s
	mov v11.s[0], v10.s[0]
	mov v13.16b, v4.16b
	mov v15.16b, v6.16b
	fmax v9.4s, v8.4s, v7.4s
	fdiv v18.4s, v12.4s, v14.4s
	ldr q12, [x0, #48]
	mov v11.s[1], v10.s[0]
	fmla v13.4s, v3.4s, v19.4s
	fadd v14.4s, v19.4s, v0.4s
	mov v11.s[2], v10.s[0]
	fmin v10.4s, v9.4s, v17.4s
	fmla v15.4s, v19.4s, v13.4s
	fmla v16.4s, v19.4s, v14.4s
	mov v14.16b, v1.16b
	fmul v9.4s, v12.4s, v11.4s
	fsqrt v11.4s, v10.4s
	fmla v14.4s, v19.4s, v15.4s
	fmax v12.4s, v9.4s, v7.4s
	mov v7.16b, v2.16b
	fmin v18.4s, v18.4s, v17.4s
	fmla v5.4s, v19.4s, v14.4s
	mov v14.16b, v22.16b
	fmla v7.4s, v19.4s, v16.4s
	mov v16.16b, v4.16b
	fmin v12.4s, v12.4s, v17.4s
	fmla v20.4s, v19.4s, v7.4s
	mov v19.16b, v21.16b
	mov v7.16b, v6.16b
	fsqrt v13.4s, v12.4s
	fadd v15.4s, v11.4s, v0.4s
	fmla v16.4s, v3.4s, v11.4s
	fmla v19.4s, v11.4s, v15.4s
	fmla v7.4s, v11.4s, v16.4s
	mov v16.16b, v1.16b
	mov v15.16b, v23.16b
	fdiv v5.4s, v5.4s, v20.4s
	mov v20.16b, v2.16b
	fmla v16.4s, v11.4s, v7.4s
	fmla v20.4s, v11.4s, v19.4s
	fmla v4.4s, v3.4s, v13.4s
	fadd v19.4s, v13.4s, v0.4s
	fmla v14.4s, v11.4s, v16.4s
	ldr q0, [sp, #16]
	fmla v6.4s, v13.4s, v4.4s
	fmla v21.4s, v13.4s, v19.4s
	fmla v15.4s, v11.4s, v20.4s
	fcmgt v20.4s, v0.4s, v30.4s
	fmla v1.4s, v13.4s, v6.4s
	fmla v2.4s, v13.4s, v21.4s
	fdiv v7.4s, v14.4s, v15.4s
	fmin v5.4s, v5.4s, v17.4s
	fmla v22.4s, v13.4s, v1.4s
	fmla v23.4s, v13.4s, v2.4s
	ldr q13, [sp, #176]
	fmul v19.4s, v30.4s, v13.4s
	fmul v30.4s, v31.4s, v13.4s
	fcmgt v31.4s, v0.4s, v31.4s
	fmul v11.4s, v10.4s, v13.4s
	fcmgt v10.4s, v0.4s, v10.4s
	fmul v13.4s, v12.4s, v13.4s
	fcmgt v12.4s, v0.4s, v12.4s
	bit v18.16b, v19.16b, v20.16b
	fcmge v19.4s, v27.4s, v17.4s
	bit v5.16b, v30.16b, v31.16b
	fdiv v16.4s, v22.4s, v23.4s
	fcmge v20.4s, v29.4s, v17.4s
	fcmge v27.4s, v8.4s, v17.4s
	fcmge v29.4s, v9.4s, v17.4s
	fmin v7.4s, v7.4s, v17.4s
	bit v18.16b, v17.16b, v19.16b
	bit v5.16b, v17.16b, v20.16b
	bit v7.16b, v11.16b, v10.16b
	stp q18, q5, [x0]
	str s24, [x0, #12]
	str s25, [x0, #28]
	bit v7.16b, v17.16b, v27.16b
	fmin v16.4s, v16.4s, v17.4s
	bit v16.16b, v13.16b, v12.16b
	bit v16.16b, v17.16b, v29.16b
	stp q7, q16, [x0, #32]
	str s26, [x0, #44]
	str s28, [x0, #60]
	mov x0, x10
.LBB16_8:
	cmp x0, x8
	ccmp x0, #0, #4, ne
	b.ne .LBB16_6
	cbz x9, .LBB16_11
	mov x10, xzr
	mov x0, x9
	mov x9, xzr
	b .LBB16_7
.LBB16_11:
	ldur w8, [x29, #-196]
	cbz w8, .LBB16_13
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB16_13:
	.cfi_def_cfa wsp, 464
	ldp x20, x19, [sp, #448]
	ldp x22, x21, [sp, #432]
	ldp x28, x23, [sp, #416]
	ldp x29, x30, [sp, #400]
	ldp d9, d8, [sp, #384]
	ldp d11, d10, [sp, #368]
	ldp d13, d12, [sp, #352]
	ldp d15, d14, [sp, #336]
	add sp, sp, #464
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w28
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	.cfi_restore b13
	.cfi_restore b14
	.cfi_restore b15
	ret
.LBB16_14:
	.cfi_restore_state
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB16_2
.LBB16_15:
	.cfi_def_cfa wsp, 464
	ldp x20, x19, [sp, #448]
	ldp x22, x21, [sp, #432]
	ldp x28, x23, [sp, #416]
	ldp x29, x30, [sp, #400]
	ldp d9, d8, [sp, #384]
	ldp d11, d10, [sp, #368]
	ldp d13, d12, [sp, #352]
	ldp d15, d14, [sp, #336]
	add sp, sp, #464
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w28
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	.cfi_restore b13
	.cfi_restore b14
	.cfi_restore b15
	b linear_srgb::simd::unpremultiply_linear_to_srgb_rgba_slice_tier_scalar
