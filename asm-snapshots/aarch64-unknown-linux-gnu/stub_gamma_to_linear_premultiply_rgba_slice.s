.section .text.stub_gamma_to_linear_premultiply_rgba_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_premultiply_rgba_slice
	.p2align	2
.type	stub_gamma_to_linear_premultiply_rgba_slice,@function
stub_gamma_to_linear_premultiply_rgba_slice:
	.cfi_startproc
	sub sp, sp, #480
	.cfi_def_cfa_offset 480
	stp d15, d14, [sp, #352]
	stp d13, d12, [sp, #368]
	stp d11, d10, [sp, #384]
	stp d9, d8, [sp, #400]
	stp x29, x30, [sp, #416]
	stp x28, x23, [sp, #432]
	stp x22, x21, [sp, #448]
	stp x20, x19, [sp, #464]
	add x29, sp, #416
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
	str q0, [sp, #192]
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB5_15
	cmp w8, #2
	b.ne .LBB5_14
.LBB5_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x21, x1, #0xc
	lsr x22, x1, #4
	add x19, x0, x8, lsl #2
	lsl x20, x21, #2
	b.eq .LBB5_4
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
	b .LBB5_5
.LBB5_4:
	mov w9, wzr
.LBB5_5:
	mov w8, #64269
	mov w10, #1267
	fmov v1.4s, #1.00000000
	movk w8, #74, lsl #16
	movk w10, #16181, lsl #16
	movi v23.4s, #67, lsl #24
	dup v2.4s, w8
	mov w8, #27455
	dup v0.4s, w10
	movk w8, #16094, lsl #16
	mov w10, #14483
	cmp x21, #0
	dup v4.4s, w8
	mov w8, #46233
	movk w10, #16246, lsl #16
	movk w8, #16147, lsl #16
	stp q0, q2, [sp, #160]
	mvni v2.4s, #127, msl #16
	dup v3.4s, w8
	mvni v0.4s, #63, msl #16
	mov w8, #43579
	movk w8, #16440, lsl #16
	stur w9, [x29, #-196]
	fneg v19.4s, v2.4s
	stp q3, q4, [sp, #128]
	dup v3.4s, w8
	mov w8, #-1023672320
	fneg v2.4s, v0.4s
	dup v21.4s, w8
	mov w8, #1123942400
	dup v0.4s, w8
	mov w8, #63802
	dup v4.4s, w10
	movk w8, #14625, lsl #16
	mov w10, #38156
	movk w10, #15389, lsl #16
	stp q0, q2, [sp, #64]
	dup v2.4s, w8
	mov w8, #50297
	movk w8, #15022, lsl #16
	stp q3, q4, [sp, #96]
	dup v0.4s, w8
	mov w8, #22716
	movk w8, #15715, lsl #16
	stp q0, q2, [sp, #32]
	dup v2.4s, w10
	dup v0.4s, w8
	mov w8, #65005
	mov w10, #29208
	movk w8, #15989, lsl #16
	movk w10, #16177, lsl #16
	dup v22.4s, w8
	dup v25.4s, w10
	sub x10, x29, #196
	add x8, x0, x22, lsl #6
	add x1, x10, #4
	stp q0, q2, [sp]
	csel x9, xzr, x1, eq
	b .LBB5_8
.LBB5_6:
	add x10, x0, #64
.LBB5_7:
	movi v17.2d, #0000000000000000
	ldp q0, q2, [x0]
	ldp q28, q27, [sp, #160]
	movi v26.4s, #127, msl #16
	fmov v29.4s, #-1.00000000
	ldr q5, [x0, #32]
	mvni v31.4s, #126
	ldp q14, q13, [sp, #96]
	mvni v10.4s, #126
	fmax v0.4s, v0.4s, v17.4s
	fmax v2.4s, v2.4s, v17.4s
	fmax v5.4s, v5.4s, v17.4s
	mov v8.16b, v13.16b
	fmin v0.4s, v0.4s, v1.4s
	fmin v2.4s, v2.4s, v1.4s
	add v7.4s, v0.4s, v27.4s
	add v24.4s, v2.4s, v27.4s
	and v3.16b, v7.16b, v26.16b
	and v6.16b, v24.16b, v26.16b
	ssra v31.4s, v7.4s, #23
	ssra v10.4s, v24.4s, #23
	add v3.4s, v3.4s, v28.4s
	scvtf v31.4s, v31.4s
	scvtf v10.4s, v10.4s
	fadd v4.4s, v3.4s, v29.4s
	fadd v3.4s, v3.4s, v1.4s
	fdiv v16.4s, v4.4s, v3.4s
	fmin v3.4s, v5.4s, v1.4s
	add v4.4s, v6.4s, v28.4s
	ldr q6, [x0, #48]
	fadd v5.4s, v4.4s, v29.4s
	fadd v4.4s, v4.4s, v1.4s
	fmax v6.4s, v6.4s, v17.4s
	add v18.4s, v3.4s, v27.4s
	and v20.16b, v18.16b, v26.16b
	fdiv v17.4s, v5.4s, v4.4s
	fmin v4.4s, v6.4s, v1.4s
	add v5.4s, v20.4s, v28.4s
	fadd v6.4s, v5.4s, v29.4s
	fadd v5.4s, v5.4s, v1.4s
	add v20.4s, v4.4s, v27.4s
	and v27.16b, v20.16b, v26.16b
	ldp q12, q26, [sp, #128]
	fdiv v5.4s, v6.4s, v5.4s
	add v6.4s, v27.4s, v28.4s
	mov v30.16b, v12.16b
	mov v11.16b, v12.16b
	mov v24.16b, v12.16b
	fadd v27.4s, v6.4s, v29.4s
	fadd v6.4s, v6.4s, v1.4s
	mov v29.16b, v12.16b
	fmul v28.4s, v17.4s, v17.4s
	mov v12.16b, v13.16b
	fmla v30.4s, v26.4s, v28.4s
	fdiv v6.4s, v27.4s, v6.4s
	fmul v27.4s, v16.4s, v16.4s
	fmul v9.4s, v5.4s, v5.4s
	fmla v29.4s, v26.4s, v27.4s
	fmla v11.4s, v26.4s, v9.4s
	fmla v8.4s, v27.4s, v29.4s
	mov v29.16b, v13.16b
	fmla v29.4s, v28.4s, v30.4s
	mov v30.16b, v14.16b
	fmla v12.4s, v9.4s, v11.4s
	fcmeq v11.4s, v0.4s, #0.0
	fmul v7.4s, v6.4s, v6.4s
	fmla v30.4s, v27.4s, v8.4s
	mov v8.16b, v14.16b
	mvni v27.4s, #126
	fmla v24.4s, v26.4s, v7.4s
	fmla v8.4s, v28.4s, v29.4s
	mov v29.16b, v13.16b
	mvni v28.4s, #126
	ssra v27.4s, v18.4s, #23
	fmla v31.4s, v16.4s, v30.4s
	mov v16.16b, v14.16b
	mvni v26.4s, #127, msl #16
	fcmeq v18.4s, v2.4s, #0.0
	ldr q30, [sp, #80]
	fmla v29.4s, v7.4s, v24.4s
	ssra v28.4s, v20.4s, #23
	mov v20.16b, v14.16b
	fmla v10.4s, v17.4s, v8.4s
	scvtf v17.4s, v27.4s
	fmla v16.4s, v9.4s, v12.4s
	fcmlt v24.4s, v0.4s, #0.0
	mov v27.16b, v11.16b
	fcmeq v0.4s, v0.4s, v19.4s
	scvtf v28.4s, v28.4s
	ldr s11, [x0, #28]
	ldr s12, [x0, #44]
	fmla v20.4s, v7.4s, v29.4s
	fcmlt v7.4s, v2.4s, #0.0
	fcmeq v29.4s, v3.4s, #0.0
	bsl v27.16b, v26.16b, v31.16b
	fmla v17.4s, v5.4s, v16.4s
	bsl v18.16b, v26.16b, v10.16b
	mov v5.16b, v24.16b
	fcmeq v2.4s, v2.4s, v19.4s
	fcmeq v16.4s, v4.4s, #0.0
	ldr s10, [x0, #12]
	ldr s14, [x0, #60]
	fmla v28.4s, v6.4s, v20.4s
	mov v6.16b, v7.16b
	fcmlt v7.4s, v3.4s, #0.0
	bsl v5.16b, v30.16b, v27.16b
	bit v17.16b, v26.16b, v29.16b
	fcmeq v3.4s, v3.4s, v19.4s
	ldr q27, [sp]
	bsl v6.16b, v30.16b, v18.16b
	ldp q24, q18, [sp, #48]
	bsl v16.16b, v26.16b, v28.16b
	ldp q26, q20, [sp, #16]
	bsl v0.16b, v19.16b, v5.16b
	fcmlt v5.4s, v4.4s, #0.0
	mov v28.16b, v22.16b
	mov v29.16b, v27.16b
	bsl v2.16b, v19.16b, v6.16b
	mov v6.16b, v7.16b
	ldr q7, [sp, #192]
	fmul v9.4s, v0.4s, v7.s[0]
	fcmeq v0.4s, v4.4s, v19.4s
	mov v4.16b, v5.16b
	bsl v6.16b, v30.16b, v17.16b
	mov v17.16b, v20.16b
	fmul v31.4s, v2.4s, v7.s[0]
	mov v2.16b, v3.16b
	bsl v4.16b, v30.16b, v16.16b
	fmax v3.4s, v9.4s, v21.4s
	bsl v2.16b, v19.16b, v6.16b
	bsl v0.16b, v19.16b, v4.16b
	fmax v4.4s, v31.4s, v21.4s
	fmin v3.4s, v3.4s, v23.4s
	fmul v30.4s, v2.4s, v7.s[0]
	fmul v8.4s, v0.4s, v7.s[0]
	fmin v0.4s, v4.4s, v23.4s
	frintn v4.4s, v3.4s
	fmax v2.4s, v30.4s, v21.4s
	fmax v5.4s, v8.4s, v21.4s
	frintn v6.4s, v0.4s
	fmin v7.4s, v2.4s, v23.4s
	fmin v2.4s, v4.4s, v18.4s
	fmin v5.4s, v5.4s, v23.4s
	fmin v13.4s, v6.4s, v18.4s
	frintn v6.4s, v7.4s
	fsub v4.4s, v3.4s, v2.4s
	fcvtns v2.4s, v2.4s
	frintn v16.4s, v5.4s
	fsub v3.4s, v0.4s, v13.4s
	fmin v15.4s, v6.4s, v18.4s
	fmla v17.4s, v24.4s, v4.4s
	mov v6.16b, v20.16b
	shl v2.4s, v2.4s, #23
	fmin v0.4s, v16.4s, v18.4s
	mov v16.16b, v26.16b
	mov v18.16b, v20.16b
	fmla v6.4s, v24.4s, v3.4s
	fsub v7.4s, v7.4s, v15.4s
	add v2.4s, v2.4s, v1.4s
	fmla v16.4s, v4.4s, v17.4s
	mov v17.16b, v26.16b
	fsub v5.4s, v5.4s, v0.4s
	fcvtns v0.4s, v0.4s
	fmla v17.4s, v3.4s, v6.4s
	mov v6.16b, v27.16b
	fmla v18.4s, v24.4s, v7.4s
	fmla v20.4s, v24.4s, v5.4s
	mov v24.16b, v26.16b
	shl v0.4s, v0.4s, #23
	fmla v6.4s, v4.4s, v16.4s
	mov v16.16b, v27.16b
	fmla v24.4s, v7.4s, v18.4s
	mov v18.16b, v26.16b
	fmov v26.4s, #1.00000000
	fmla v16.4s, v3.4s, v17.4s
	mov v17.16b, v22.16b
	add v0.4s, v0.4s, v1.4s
	fmla v18.4s, v5.4s, v20.4s
	mov v20.16b, v25.16b
	fmla v17.4s, v4.4s, v6.4s
	fmla v29.4s, v7.4s, v24.4s
	mov v24.16b, v25.16b
	fmla v28.4s, v3.4s, v16.4s
	fmov v6.4s, #1.00000000
	fmov v16.4s, #1.00000000
	fmla v27.4s, v5.4s, v18.4s
	mov v18.16b, v22.16b
	fmla v20.4s, v4.4s, v17.4s
	fmov v17.4s, #1.00000000
	fmla v24.4s, v3.4s, v28.4s
	mov v28.16b, v22.16b
	fmla v18.4s, v7.4s, v29.4s
	mov v29.16b, v25.16b
	fmla v28.4s, v5.4s, v27.4s
	fmla v6.4s, v4.4s, v20.4s
	mov v20.16b, v25.16b
	fmla v16.4s, v3.4s, v24.4s
	fcvtns v24.4s, v13.4s
	fmov v27.4s, #1.00000000
	fmla v29.4s, v7.4s, v18.4s
	fcvtns v13.4s, v15.4s
	fmov v3.4s, #1.00000000
	fmov v4.4s, #1.00000000
	fmov v18.4s, #1.00000000
	mov v17.s[0], v10.s[0]
	fmla v20.4s, v5.4s, v28.4s
	fcmge v28.4s, v9.4s, v23.4s
	fcmgt v9.4s, v21.4s, v9.4s
	shl v24.4s, v24.4s, #23
	fmul v2.4s, v6.4s, v2.4s
	fmla v26.4s, v7.4s, v29.4s
	fcmge v29.4s, v31.4s, v23.4s
	fcmgt v7.4s, v21.4s, v31.4s
	fcmge v31.4s, v30.4s, v23.4s
	fcmgt v30.4s, v21.4s, v30.4s
	mov v3.s[0], v11.s[0]
	fmla v27.4s, v5.4s, v20.4s
	mvn v5.16b, v28.16b
	shl v20.4s, v13.4s, #23
	fcmge v13.4s, v8.4s, v23.4s
	add v24.4s, v24.4s, v1.4s
	fcmgt v8.4s, v21.4s, v8.4s
	mov v4.s[0], v12.s[0]
	mov v18.s[0], v14.s[0]
	mov v17.s[1], v10.s[0]
	bic v5.16b, v5.16b, v9.16b
	mvn v9.16b, v29.16b
	add v20.4s, v20.4s, v1.4s
	fmul v16.4s, v16.4s, v24.4s
	mvn v6.16b, v31.16b
	fmul v0.4s, v27.4s, v0.4s
	mvn v24.16b, v13.16b
	mov v3.s[1], v11.s[0]
	bic v7.16b, v9.16b, v7.16b
	fmul v20.4s, v26.4s, v20.4s
	and v2.16b, v5.16b, v2.16b
	bic v6.16b, v6.16b, v30.16b
	mov v4.s[1], v12.s[0]
	mov v18.s[1], v14.s[0]
	bic v24.16b, v24.16b, v8.16b
	mov v17.s[2], v10.s[0]
	and v5.16b, v7.16b, v16.16b
	and v7.16b, v28.16b, v19.16b
	and v16.16b, v29.16b, v19.16b
	and v6.16b, v6.16b, v20.16b
	and v20.16b, v31.16b, v19.16b
	mov v3.s[2], v11.s[0]
	and v0.16b, v24.16b, v0.16b
	mov v4.s[2], v12.s[0]
	mov v18.s[2], v14.s[0]
	orr v2.16b, v2.16b, v7.16b
	and v7.16b, v13.16b, v19.16b
	orr v5.16b, v5.16b, v16.16b
	orr v6.16b, v6.16b, v20.16b
	orr v0.16b, v0.16b, v7.16b
	fmul v2.4s, v17.4s, v2.4s
	fmul v3.4s, v3.4s, v5.4s
	fmul v4.4s, v4.4s, v6.4s
	fmul v0.4s, v18.4s, v0.4s
	stp q2, q3, [x0]
	str s10, [x0, #12]
	str s11, [x0, #28]
	stp q4, q0, [x0, #32]
	str s12, [x0, #44]
	str s14, [x0, #60]
	mov x0, x10
.LBB5_8:
	cmp x0, x8
	ccmp x0, #0, #4, ne
	b.ne .LBB5_6
	cbz x9, .LBB5_11
	mov x10, xzr
	mov x0, x9
	mov x9, xzr
	b .LBB5_7
.LBB5_11:
	ldur w8, [x29, #-196]
	cbz w8, .LBB5_13
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB5_13:
	.cfi_def_cfa wsp, 480
	ldp x20, x19, [sp, #464]
	ldp x22, x21, [sp, #448]
	ldp x28, x23, [sp, #432]
	ldp x29, x30, [sp, #416]
	ldp d9, d8, [sp, #400]
	ldp d11, d10, [sp, #384]
	ldp d13, d12, [sp, #368]
	ldp d15, d14, [sp, #352]
	add sp, sp, #480
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
.LBB5_14:
	.cfi_restore_state
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB5_2
.LBB5_15:
	ldr q0, [sp, #192]
	.cfi_def_cfa wsp, 480
	ldp x20, x19, [sp, #464]
	ldp x22, x21, [sp, #448]
	ldp x28, x23, [sp, #432]
	ldp x29, x30, [sp, #416]
	ldp d9, d8, [sp, #400]
	ldp d11, d10, [sp, #384]
	ldp d13, d12, [sp, #368]
	ldp d15, d14, [sp, #352]
	add sp, sp, #480
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
	b linear_srgb::simd::gamma_to_linear_premultiply_rgba_slice_tier_scalar
