.section .text.stub_unpremultiply_linear_to_gamma_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_gamma_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_gamma_rgba_slice,@function
stub_unpremultiply_linear_to_gamma_rgba_slice:
	.cfi_startproc
	stp d15, d14, [sp, #-128]!
	.cfi_def_cfa_offset 128
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	stp x29, x30, [sp, #64]
	stp x28, x23, [sp, #80]
	stp x22, x21, [sp, #96]
	stp x20, x19, [sp, #112]
	add x29, sp, #64
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
	sub sp, sp, #400
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB15_15
	cmp w8, #2
	b.ne .LBB15_14
.LBB15_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x21, x1, #0xc
	lsr x22, x1, #4
	add x19, x0, x8, lsl #2
	lsl x20, x21, #2
	b.eq .LBB15_4
	mov w8, #64
	mov x23, x0
	mov w1, wzr
	sub x2, x8, x20
	sub x8, x29, #128
	fmov s8, s0
	add x0, x8, x20
	bl memset
	sub x0, x29, #128
	mov x1, x19
	mov x2, x20
	bl memcpy
	ldp q1, q2, [x29, #-128]
	fmov s0, s8
	mov x0, x23
	mov w9, #1
	stp q1, q2, [x29, #-192]
	ldp q1, q3, [x29, #-96]
	stp q1, q3, [x29, #-160]
	b .LBB15_5
.LBB15_4:
	mov w9, wzr
.LBB15_5:
	fmov s1, #1.00000000
	mov w8, #64269
	mov w10, #1267
	movk w8, #74, lsl #16
	movk w10, #16181, lsl #16
	fmov v10.4s, #1.00000000
	dup v2.4s, w8
	mov w8, #27455
	dup v3.4s, w10
	fdiv s0, s1, s0
	movk w8, #16094, lsl #16
	mov w10, #29208
	movk w10, #16177, lsl #16
	cmp x21, #0
	stur w9, [x29, #-196]
	stur q2, [x29, #-224]
	dup v2.4s, w8
	mov w8, #46233
	movk w8, #16147, lsl #16
	stp q2, q3, [sp, #208]
	dup v2.4s, w8
	mov w8, #14483
	movk w8, #16246, lsl #16
	ldr q8, [sp, #208]
	dup v1.4s, w8
	mov w8, #43579
	movk w8, #16440, lsl #16
	stp q0, q2, [sp, #176]
	dup v0.4s, w8
	mov w8, #-1023672320
	dup v20.4s, w8
	mov w8, #1123942400
	dup v3.4s, w8
	mov w8, #63802
	movk w8, #14625, lsl #16
	stp q0, q1, [sp, #144]
	mvni v1.4s, #127, msl #16
	dup v2.4s, w8
	mov w8, #50297
	mvni v0.4s, #63, msl #16
	movk w8, #15022, lsl #16
	fneg v26.4s, v1.4s
	stp q2, q3, [sp, #112]
	dup v2.4s, w8
	mov w8, #38156
	movk w8, #15389, lsl #16
	dup v1.4s, w8
	mov w8, #22716
	movk w8, #15715, lsl #16
	stp q1, q2, [sp, #80]
	fneg v1.4s, v0.4s
	dup v0.4s, w8
	mov w8, #65005
	movk w8, #15989, lsl #16
	stp q0, q1, [sp, #48]
	dup v1.4s, w8
	dup v0.4s, w10
	sub x10, x29, #196
	add x8, x0, x22, lsl #6
	add x1, x10, #4
	mov w10, #981467136
	csel x9, xzr, x1, eq
	stp q0, q1, [sp, #16]
	str w10, [sp, #12]
	b .LBB15_8
.LBB15_6:
	add x10, x0, #64
.LBB15_7:
	fmov s5, #1.00000000
	ldr s1, [x0, #12]
	ldr s11, [x0, #28]
	movi d6, #0000000000000000
	fmov v2.4s, #1.00000000
	ldr s12, [x0, #44]
	stur s1, [x29, #-200]
	fmov v4.4s, #1.00000000
	ldr s13, [x0, #60]
	fdiv s0, s5, s1
	ldr s7, [sp, #12]
	fmov v17.4s, #1.00000000
	ldur q16, [x29, #-224]
	fmov v19.4s, #1.00000000
	movi v22.4s, #127, msl #16
	fcmp s1, s7
	ldr q18, [sp, #224]
	fmov v23.4s, #-1.00000000
	ldr q28, [sp, #192]
	mvni v25.4s, #126
	mvni v27.4s, #126
	mov v9.16b, v28.16b
	fdiv s1, s5, s11
	fcsel s0, s0, s6, gt
	fcmp s11, s7
	mov v2.s[0], v0.s[0]
	fdiv s3, s5, s12
	mov v2.s[1], v0.s[0]
	fcsel s1, s1, s6, gt
	fcmp s12, s7
	mov v2.s[2], v0.s[0]
	ldr q0, [x0]
	mov v4.s[0], v1.s[0]
	fmul v0.4s, v0.4s, v2.4s
	movi v2.2d, #0000000000000000
	fdiv s5, s5, s13
	mov v4.s[1], v1.s[0]
	fmax v0.4s, v0.4s, v2.4s
	fcsel s3, s3, s6, gt
	fcmp s13, s7
	mov v4.s[2], v1.s[0]
	ldr q1, [x0, #16]
	mov v17.s[0], v3.s[0]
	fmin v14.4s, v0.4s, v10.4s
	fmul v0.4s, v1.4s, v4.4s
	mov v17.s[1], v3.s[0]
	add v7.4s, v14.4s, v16.4s
	fmax v0.4s, v0.4s, v2.4s
	fcsel s4, s5, s6, gt
	and v5.16b, v7.16b, v22.16b
	ssra v25.4s, v7.4s, #23
	mov v17.s[2], v3.s[0]
	ldr q3, [x0, #32]
	mov v19.s[0], v4.s[0]
	fmin v1.4s, v0.4s, v10.4s
	add v0.4s, v5.4s, v18.4s
	fmul v3.4s, v3.4s, v17.4s
	fadd v5.4s, v0.4s, v23.4s
	fadd v0.4s, v0.4s, v10.4s
	mov v19.s[1], v4.s[0]
	add v21.4s, v1.4s, v16.4s
	fmax v17.4s, v3.4s, v2.4s
	and v24.16b, v21.16b, v22.16b
	fdiv v3.4s, v5.4s, v0.4s
	ldr q0, [x0, #48]
	mov v19.s[2], v4.s[0]
	ssra v27.4s, v21.4s, #23
	fmin v4.4s, v17.4s, v10.4s
	add v5.4s, v24.4s, v18.4s
	fmul v0.4s, v0.4s, v19.4s
	fadd v17.4s, v5.4s, v23.4s
	fadd v5.4s, v5.4s, v10.4s
	add v19.4s, v4.4s, v16.4s
	fmax v24.4s, v0.4s, v2.4s
	ldr q2, [sp, #160]
	and v6.16b, v19.16b, v22.16b
	fdiv v0.4s, v17.4s, v5.4s
	mov v30.16b, v2.16b
	mov v7.16b, v2.16b
	mov v29.16b, v2.16b
	add v5.4s, v6.4s, v18.4s
	fmin v15.4s, v24.4s, v10.4s
	fadd v6.4s, v5.4s, v23.4s
	fadd v5.4s, v5.4s, v10.4s
	add v24.4s, v15.4s, v16.4s
	fdiv v5.4s, v6.4s, v5.4s
	and v17.16b, v24.16b, v22.16b
	mov v22.16b, v28.16b
	add v6.4s, v17.4s, v18.4s
	fmul v16.4s, v0.4s, v0.4s
	fadd v17.4s, v6.4s, v23.4s
	fadd v6.4s, v6.4s, v10.4s
	mov v23.16b, v28.16b
	fmla v23.4s, v8.4s, v16.4s
	fdiv v17.4s, v17.4s, v6.4s
	fmul v6.4s, v3.4s, v3.4s
	fmul v18.4s, v5.4s, v5.4s
	fmla v29.4s, v16.4s, v23.4s
	scvtf v23.4s, v25.4s
	fmla v22.4s, v8.4s, v6.4s
	scvtf v25.4s, v27.4s
	mvni v27.4s, #126
	fmla v9.4s, v8.4s, v18.4s
	ssra v27.4s, v24.4s, #23
	fmla v30.4s, v6.4s, v22.4s
	mvni v22.4s, #126
	fmla v7.4s, v18.4s, v9.4s
	ldr q9, [sp, #144]
	ssra v22.4s, v19.4s, #23
	mov v19.16b, v9.16b
	fmul v31.4s, v17.4s, v17.4s
	mov v21.16b, v9.16b
	scvtf v22.4s, v22.4s
	fmla v19.4s, v16.4s, v29.4s
	mov v16.16b, v9.16b
	fmla v21.4s, v6.4s, v30.4s
	fmla v28.4s, v8.4s, v31.4s
	fcmlt v6.4s, v14.4s, #0.0
	movi v29.4s, #67, lsl #24
	fmla v16.4s, v18.4s, v7.4s
	fcmeq v18.4s, v14.4s, #0.0
	mvni v7.4s, #127, msl #16
	fmla v23.4s, v3.4s, v21.4s
	fcmeq v3.4s, v1.4s, #0.0
	fmla v25.4s, v0.4s, v19.4s
	fmla v2.4s, v31.4s, v28.4s
	fcmeq v0.4s, v4.4s, #0.0
	fcmeq v21.4s, v14.4s, v26.4s
	fcmlt v19.4s, v4.4s, #0.0
	ldr q28, [sp, #96]
	fmla v22.4s, v5.4s, v16.4s
	mov v5.16b, v9.16b
	scvtf v16.4s, v27.4s
	bsl v18.16b, v7.16b, v23.16b
	bsl v3.16b, v7.16b, v25.16b
	ldr q23, [sp, #64]
	fmla v5.4s, v31.4s, v2.4s
	fcmlt v2.4s, v1.4s, #0.0
	fcmeq v1.4s, v1.4s, v26.4s
	bsl v0.16b, v7.16b, v22.16b
	fcmeq v22.4s, v15.4s, #0.0
	bsl v6.16b, v23.16b, v18.16b
	bsl v2.16b, v23.16b, v3.16b
	fcmeq v3.4s, v4.4s, v26.4s
	mov v4.16b, v21.16b
	fmla v16.4s, v17.4s, v5.4s
	bit v0.16b, v23.16b, v19.16b
	fcmlt v5.4s, v15.4s, #0.0
	bsl v4.16b, v26.16b, v6.16b
	mov v6.16b, v22.16b
	mov v22.16b, v28.16b
	bsl v1.16b, v26.16b, v2.16b
	fcmeq v2.4s, v15.4s, v26.4s
	bit v0.16b, v26.16b, v3.16b
	mov v3.16b, v5.16b
	bsl v6.16b, v7.16b, v16.16b
	ldr q16, [sp, #176]
	ldp q27, q7, [sp, #112]
	fmul v4.4s, v4.4s, v16.s[0]
	fmul v1.4s, v1.4s, v16.s[0]
	fmul v14.4s, v0.4s, v16.s[0]
	bsl v3.16b, v23.16b, v6.16b
	fmax v0.4s, v4.4s, v20.4s
	fmax v5.4s, v14.4s, v20.4s
	bsl v2.16b, v26.16b, v3.16b
	fmax v3.4s, v1.4s, v20.4s
	fmin v6.4s, v0.4s, v29.4s
	fmul v15.4s, v2.4s, v16.s[0]
	fmin v16.4s, v5.4s, v29.4s
	fmin v2.4s, v3.4s, v29.4s
	frintn v0.4s, v6.4s
	fmax v3.4s, v15.4s, v20.4s
	frintn v17.4s, v16.4s
	frintn v5.4s, v2.4s
	fmin v0.4s, v0.4s, v7.4s
	fmin v18.4s, v3.4s, v29.4s
	fmin v21.4s, v17.4s, v7.4s
	fmin v5.4s, v5.4s, v7.4s
	fsub v3.4s, v6.4s, v0.4s
	fcvtns v0.4s, v0.4s
	frintn v6.4s, v18.4s
	fsub v19.4s, v16.4s, v21.4s
	mov v16.16b, v28.16b
	fsub v2.4s, v2.4s, v5.4s
	fcvtns v5.4s, v5.4s
	shl v0.4s, v0.4s, #23
	fmla v16.4s, v27.4s, v3.4s
	fmin v24.4s, v6.4s, v7.4s
	ldr q7, [sp, #80]
	mov v6.16b, v28.16b
	fmla v22.4s, v27.4s, v2.4s
	shl v5.4s, v5.4s, #23
	add v0.4s, v0.4s, v10.4s
	mov v23.16b, v7.16b
	mov v25.16b, v7.16b
	fmla v6.4s, v27.4s, v19.4s
	fsub v17.4s, v18.4s, v24.4s
	add v5.4s, v5.4s, v10.4s
	fmla v23.4s, v3.4s, v16.4s
	mov v16.16b, v7.16b
	fmla v25.4s, v2.4s, v22.4s
	fmla v16.4s, v19.4s, v6.4s
	mov v6.16b, v28.16b
	ldr q28, [sp, #48]
	mov v18.16b, v28.16b
	mov v22.16b, v28.16b
	fmla v6.4s, v27.4s, v17.4s
	fmla v18.4s, v3.4s, v23.4s
	mov v23.16b, v28.16b
	fmla v22.4s, v2.4s, v25.4s
	fmla v23.4s, v19.4s, v16.4s
	mov v16.16b, v7.16b
	ldp q30, q7, [sp, #16]
	fmla v16.4s, v17.4s, v6.4s
	mov v25.16b, v7.16b
	mov v27.16b, v7.16b
	mov v6.16b, v7.16b
	fmla v25.4s, v3.4s, v18.4s
	fmla v27.4s, v2.4s, v22.4s
	mov v18.16b, v28.16b
	fmla v6.4s, v19.4s, v23.4s
	mov v22.16b, v30.16b
	mov v23.16b, v30.16b
	fmov v28.4s, #1.00000000
	fmla v18.4s, v17.4s, v16.4s
	mov v16.16b, v30.16b
	fmla v22.4s, v3.4s, v25.4s
	fmov v25.4s, #1.00000000
	fmla v23.4s, v2.4s, v27.4s
	mov v27.16b, v7.16b
	fmla v16.4s, v19.4s, v6.4s
	fmov v6.4s, #1.00000000
	fmla v27.4s, v17.4s, v18.4s
	fmla v28.4s, v3.4s, v22.4s
	fcmge v3.4s, v4.4s, v29.4s
	fmla v25.4s, v2.4s, v23.4s
	mov v2.16b, v30.16b
	fcvtns v18.4s, v21.4s
	fcmge v21.4s, v1.4s, v29.4s
	fcvtns v22.4s, v24.4s
	fcmgt v4.4s, v20.4s, v4.4s
	fmla v6.4s, v19.4s, v16.4s
	fmov v16.4s, #1.00000000
	fcmgt v1.4s, v20.4s, v1.4s
	fmla v2.4s, v17.4s, v27.4s
	mvn v19.16b, v3.16b
	fmul v0.4s, v28.4s, v0.4s
	shl v18.4s, v18.4s, #23
	fcmge v23.4s, v14.4s, v29.4s
	fmul v5.4s, v25.4s, v5.4s
	mvn v24.16b, v21.16b
	shl v22.4s, v22.4s, #23
	and v3.16b, v3.16b, v26.16b
	bic v4.16b, v19.16b, v4.16b
	fcmge v19.4s, v15.4s, v29.4s
	fmla v16.4s, v17.4s, v2.4s
	add v17.4s, v18.4s, v10.4s
	fcmgt v2.4s, v20.4s, v14.4s
	bic v1.16b, v24.16b, v1.16b
	add v22.4s, v22.4s, v10.4s
	mvn v18.16b, v23.16b
	and v0.16b, v4.16b, v0.16b
	fcmgt v4.4s, v20.4s, v15.4s
	fmul v6.4s, v6.4s, v17.4s
	mvn v17.16b, v19.16b
	and v1.16b, v1.16b, v5.16b
	fmul v5.4s, v16.4s, v22.4s
	and v16.16b, v21.16b, v26.16b
	bic v2.16b, v18.16b, v2.16b
	orr v0.16b, v0.16b, v3.16b
	and v3.16b, v23.16b, v26.16b
	bic v4.16b, v17.16b, v4.16b
	orr v1.16b, v1.16b, v16.16b
	and v2.16b, v2.16b, v6.16b
	stp q0, q1, [x0]
	and v0.16b, v4.16b, v5.16b
	and v4.16b, v19.16b, v26.16b
	orr v2.16b, v2.16b, v3.16b
	ldur s5, [x29, #-200]
	str s11, [x0, #28]
	orr v0.16b, v0.16b, v4.16b
	str s5, [x0, #12]
	stp q2, q0, [x0, #32]
	str s12, [x0, #44]
	str s13, [x0, #60]
	mov x0, x10
.LBB15_8:
	cmp x0, x8
	ccmp x0, #0, #4, ne
	b.ne .LBB15_6
	cbz x9, .LBB15_11
	mov x10, xzr
	mov x0, x9
	mov x9, xzr
	b .LBB15_7
.LBB15_11:
	ldur w8, [x29, #-196]
	cbz w8, .LBB15_13
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB15_13:
	add sp, sp, #400
	.cfi_def_cfa wsp, 128
	ldp x20, x19, [sp, #112]
	ldp x22, x21, [sp, #96]
	ldp x28, x23, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #128
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
.LBB15_14:
	.cfi_restore_state
	mov x19, x0
	fmov s8, s0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	fmov s0, s8
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB15_2
.LBB15_15:
	add sp, sp, #400
	.cfi_def_cfa wsp, 128
	ldp x20, x19, [sp, #112]
	ldp x22, x21, [sp, #96]
	ldp x28, x23, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #128
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
	b linear_srgb::simd::unpremultiply_linear_to_gamma_rgba_slice_tier_scalar
