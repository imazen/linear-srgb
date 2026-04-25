.section .text.stub_gamma_to_linear_premultiply_rgba_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_premultiply_rgba_slice
	.p2align	2
.type	stub_gamma_to_linear_premultiply_rgba_slice,@function
stub_gamma_to_linear_premultiply_rgba_slice:
	.cfi_startproc
	sub sp, sp, #336
	.cfi_def_cfa_offset 336
	stp d15, d14, [sp, #224]
	stp d13, d12, [sp, #240]
	stp d11, d10, [sp, #256]
	stp d9, d8, [sp, #272]
	stp x29, x30, [sp, #288]
	str x28, [sp, #304]
	stp x20, x19, [sp, #320]
	add x29, sp, #288
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w28, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	.cfi_offset b8, -56
	.cfi_offset b9, -64
	.cfi_offset b10, -72
	.cfi_offset b11, -80
	.cfi_offset b12, -88
	.cfi_offset b13, -96
	.cfi_offset b14, -104
	.cfi_offset b15, -112
	fmov s1, s0
	lsl x8, x1, #2
	ands x8, x8, #0x7fffffffffffffc0
	stur q1, [x29, #-80]
	b.eq .LBB5_3
	mov w9, #64269
	mov w10, #1267
	mov w11, #29208
	movk w9, #74, lsl #16
	movk w10, #16181, lsl #16
	fmov v26.4s, #1.00000000
	dup v2.4s, w9
	dup v0.4s, w10
	mov w9, #27455
	movk w9, #16094, lsl #16
	mov w10, #14483
	movk w11, #16177, lsl #16
	movk w10, #16246, lsl #16
	movi v27.4s, #67, lsl #24
	dup v25.4s, w11
	stp q0, q2, [x29, #-112]
	dup v0.4s, w9
	mov w9, #46233
	movk w9, #16147, lsl #16
	dup v2.4s, w10
	mov w10, #38156
	dup v3.4s, w9
	mov w9, #43579
	movk w10, #15389, lsl #16
	stur q0, [x29, #-128]
	mvni v0.4s, #63, msl #16
	movk w9, #16440, lsl #16
	stp q2, q3, [sp, #128]
	dup v3.4s, w9
	mov w9, #-1023672320
	fneg v0.4s, v0.4s
	dup v20.4s, w9
	mov w9, #1123942400
	mvni v2.4s, #127, msl #16
	stp q0, q3, [sp, #96]
	dup v3.4s, w9
	mov w9, #63802
	movk w9, #14625, lsl #16
	dup v0.4s, w9
	mov w9, #50297
	movk w9, #15022, lsl #16
	stp q0, q3, [sp, #64]
	dup v3.4s, w9
	dup v0.4s, w10
	mov w9, #22716
	mov w10, #65005
	movk w9, #15715, lsl #16
	movk w10, #15989, lsl #16
	stp q0, q3, [sp, #32]
	fneg v3.4s, v2.4s
	dup v0.4s, w9
	dup v24.4s, w10
	add x9, x0, #32
	stp q0, q3, [sp]
.LBB5_2:
	movi v5.2d, #0000000000000000
	ldur q0, [x9, #-32]
	movi v6.4s, #127, msl #16
	ldp q16, q7, [x29, #-112]
	fmov v17.4s, #-1.00000000
	ldp q23, q22, [sp, #128]
	ldur q21, [x29, #-128]
	mvni v13.4s, #126
	mvni v19.4s, #126
	ldr q28, [sp, #112]
	fmax v0.4s, v0.4s, v5.4s
	mov v29.16b, v25.16b
	subs x8, x8, #64
	mov v18.16b, v22.16b
	mov v14.16b, v23.16b
	fmin v30.4s, v0.4s, v26.4s
	ldp q0, q3, [x9, #-16]
	fmax v0.4s, v0.4s, v5.4s
	fmax v3.4s, v3.4s, v5.4s
	add v31.4s, v30.4s, v7.4s
	and v2.16b, v31.16b, v6.16b
	fmin v8.4s, v0.4s, v26.4s
	fmin v10.4s, v3.4s, v26.4s
	ldr q3, [x9, #16]
	ssra v13.4s, v31.4s, #23
	add v0.4s, v2.4s, v16.4s
	fmax v3.4s, v3.4s, v5.4s
	add v9.4s, v8.4s, v7.4s
	add v12.4s, v10.4s, v7.4s
	scvtf v13.4s, v13.4s
	fadd v2.4s, v0.4s, v17.4s
	fadd v0.4s, v0.4s, v26.4s
	and v4.16b, v9.16b, v6.16b
	ssra v19.4s, v9.4s, #23
	mov v9.16b, v22.16b
	fdiv v11.4s, v2.4s, v0.4s
	add v0.4s, v4.4s, v16.4s
	and v4.16b, v12.16b, v6.16b
	scvtf v19.4s, v19.4s
	fadd v2.4s, v0.4s, v17.4s
	fadd v0.4s, v0.4s, v26.4s
	fdiv v2.4s, v2.4s, v0.4s
	fmin v0.4s, v3.4s, v26.4s
	add v3.4s, v4.4s, v16.4s
	fadd v4.4s, v3.4s, v17.4s
	fadd v3.4s, v3.4s, v26.4s
	add v5.4s, v0.4s, v7.4s
	and v6.16b, v5.16b, v6.16b
	fdiv v3.4s, v4.4s, v3.4s
	add v4.4s, v6.4s, v16.4s
	mov v16.16b, v22.16b
	fadd v6.4s, v4.4s, v17.4s
	fadd v4.4s, v4.4s, v26.4s
	fmul v7.4s, v2.4s, v2.4s
	mov v17.16b, v22.16b
	mvni v22.4s, #127, msl #16
	fmla v18.4s, v21.4s, v7.4s
	fdiv v4.4s, v6.4s, v4.4s
	fmul v6.4s, v11.4s, v11.4s
	fmul v15.4s, v3.4s, v3.4s
	fmla v16.4s, v21.4s, v6.4s
	fmla v17.4s, v21.4s, v15.4s
	fmla v14.4s, v6.4s, v16.4s
	mov v16.16b, v23.16b
	fmla v16.4s, v7.4s, v18.4s
	mov v18.16b, v28.16b
	fmul v31.4s, v4.4s, v4.4s
	fmla v18.4s, v6.4s, v14.4s
	mov v6.16b, v28.16b
	mvni v14.4s, #126
	fmla v9.4s, v21.4s, v31.4s
	mov v21.16b, v23.16b
	fmla v6.4s, v7.4s, v16.4s
	mov v16.16b, v23.16b
	ssra v14.4s, v12.4s, #23
	fmla v13.4s, v11.4s, v18.4s
	mov v18.16b, v28.16b
	mvni v7.4s, #126
	mov v23.16b, v25.16b
	fmla v21.4s, v15.4s, v17.4s
	fcmeq v17.4s, v30.4s, #0.0
	fmla v16.4s, v31.4s, v9.4s
	fmla v19.4s, v2.4s, v6.4s
	scvtf v2.4s, v14.4s
	fcmlt v6.4s, v30.4s, #0.0
	ssra v7.4s, v5.4s, #23
	mov v5.16b, v28.16b
	fcmeq v9.4s, v8.4s, #0.0
	mov v28.16b, v24.16b
	ldur s30, [x9, #-20]
	fmla v18.4s, v15.4s, v21.4s
	bsl v17.16b, v22.16b, v13.16b
	fcmeq v21.4s, v10.4s, #0.0
	scvtf v7.4s, v7.4s
	fmla v5.4s, v31.4s, v16.4s
	fcmlt v16.4s, v8.4s, #0.0
	ldur s31, [x9, #-4]
	bit v19.16b, v22.16b, v9.16b
	fmla v2.4s, v3.4s, v18.4s
	mov v3.16b, v6.16b
	ldr q18, [sp, #96]
	fcmeq v6.4s, v0.4s, #0.0
	fmla v7.4s, v4.4s, v5.4s
	mov v4.16b, v16.16b
	fcmlt v5.4s, v10.4s, #0.0
	fcmlt v0.4s, v0.4s, #0.0
	ldr s10, [x9, #12]
	bsl v3.16b, v18.16b, v17.16b
	bsl v4.16b, v18.16b, v19.16b
	bit v2.16b, v22.16b, v21.16b
	fmul v13.4s, v3.4s, v1.s[0]
	mov v3.16b, v6.16b
	fmul v12.4s, v4.4s, v1.s[0]
	bit v2.16b, v18.16b, v5.16b
	bsl v3.16b, v22.16b, v7.16b
	ldp q11, q22, [sp, #32]
	fmax v4.4s, v13.4s, v20.4s
	fmul v8.4s, v2.4s, v1.s[0]
	mov v17.16b, v22.16b
	mov v19.16b, v11.16b
	bsl v0.16b, v18.16b, v3.16b
	fmax v3.4s, v12.4s, v20.4s
	fmin v2.4s, v4.4s, v27.4s
	fmul v9.4s, v0.4s, v1.s[0]
	fmin v0.4s, v3.4s, v27.4s
	fmax v3.4s, v8.4s, v20.4s
	frintn v4.4s, v2.4s
	ldp q21, q1, [sp, #64]
	fmax v5.4s, v9.4s, v20.4s
	frintn v6.4s, v0.4s
	fmin v3.4s, v3.4s, v27.4s
	fmin v18.4s, v4.4s, v1.4s
	fmin v4.4s, v5.4s, v27.4s
	fmin v14.4s, v6.4s, v1.4s
	frintn v5.4s, v3.4s
	fsub v16.4s, v2.4s, v18.4s
	mov v6.16b, v22.16b
	fcvtns v18.4s, v18.4s
	frintn v2.4s, v4.4s
	fsub v7.4s, v0.4s, v14.4s
	fmin v0.4s, v5.4s, v1.4s
	fmla v6.4s, v21.4s, v16.4s
	shl v18.4s, v18.4s, #23
	fmin v2.4s, v2.4s, v1.4s
	fmla v17.4s, v21.4s, v7.4s
	fsub v5.4s, v3.4s, v0.4s
	mov v3.16b, v11.16b
	ldr q1, [sp]
	fmla v19.4s, v16.4s, v6.4s
	fcvtns v0.4s, v0.4s
	add v18.4s, v18.4s, v26.4s
	mov v6.16b, v1.16b
	fsub v15.4s, v4.4s, v2.4s
	mov v4.16b, v22.16b
	fmla v3.4s, v7.4s, v17.4s
	mov v17.16b, v22.16b
	mov v22.16b, v24.16b
	fcvtns v2.4s, v2.4s
	fmla v6.4s, v16.4s, v19.4s
	mov v19.16b, v1.16b
	shl v0.4s, v0.4s, #23
	fmla v4.4s, v21.4s, v5.4s
	fmla v17.4s, v21.4s, v15.4s
	mov v21.16b, v11.16b
	fmla v19.4s, v7.4s, v3.4s
	mov v3.16b, v24.16b
	shl v2.4s, v2.4s, #23
	add v0.4s, v0.4s, v26.4s
	fmla v21.4s, v5.4s, v4.4s
	fmov v4.4s, #1.00000000
	fmla v11.4s, v15.4s, v17.4s
	mov v17.16b, v1.16b
	fmla v3.4s, v16.4s, v6.4s
	fmla v28.4s, v7.4s, v19.4s
	mov v19.16b, v25.16b
	fmov v6.4s, #1.00000000
	add v2.4s, v2.4s, v26.4s
	fmla v17.4s, v5.4s, v21.4s
	mov v21.16b, v1.16b
	ldr q1, [sp, #16]
	fmla v19.4s, v16.4s, v3.4s
	fmov v3.4s, #1.00000000
	fmla v29.4s, v7.4s, v28.4s
	mov v28.16b, v25.16b
	fmla v21.4s, v15.4s, v11.4s
	ldr s11, [x9, #28]
	fmla v22.4s, v5.4s, v17.4s
	mov v17.16b, v24.16b
	fmla v4.4s, v16.4s, v19.4s
	fmov v19.4s, #1.00000000
	fmov v16.4s, #1.00000000
	fmla v6.4s, v7.4s, v29.4s
	fcvtns v29.4s, v14.4s
	fmov v7.4s, #1.00000000
	fmla v17.4s, v15.4s, v21.4s
	fmov v21.4s, #1.00000000
	fcmge v14.4s, v9.4s, v27.4s
	fmla v28.4s, v5.4s, v22.4s
	fmov v22.4s, #1.00000000
	fcmgt v9.4s, v20.4s, v9.4s
	fmul v4.4s, v4.4s, v18.4s
	mov v3.s[0], v30.s[0]
	mov v16.s[0], v10.s[0]
	shl v29.4s, v29.4s, #23
	mov v7.s[0], v31.s[0]
	fmla v23.4s, v15.4s, v17.4s
	fcmge v17.4s, v13.4s, v27.4s
	fcmgt v13.4s, v20.4s, v13.4s
	fmla v19.4s, v5.4s, v28.4s
	fcmge v5.4s, v12.4s, v27.4s
	fcmgt v28.4s, v20.4s, v12.4s
	fcmge v12.4s, v8.4s, v27.4s
	add v29.4s, v29.4s, v26.4s
	fcmgt v8.4s, v20.4s, v8.4s
	mov v22.s[0], v11.s[0]
	mov v3.s[1], v30.s[0]
	mov v7.s[1], v31.s[0]
	fmla v21.4s, v15.4s, v23.4s
	mvn v23.16b, v17.16b
	and v17.16b, v17.16b, v1.16b
	fmul v6.4s, v6.4s, v29.4s
	mvn v29.16b, v14.16b
	fmul v0.4s, v19.4s, v0.4s
	mvn v18.16b, v12.16b
	mov v16.s[1], v10.s[0]
	bic v23.16b, v23.16b, v13.16b
	mvn v13.16b, v5.16b
	mov v22.s[1], v11.s[0]
	fmul v2.4s, v21.4s, v2.4s
	bic v19.16b, v29.16b, v9.16b
	and v5.16b, v5.16b, v1.16b
	bic v18.16b, v18.16b, v8.16b
	mov v3.s[2], v30.s[0]
	mov v7.s[2], v31.s[0]
	bic v28.16b, v13.16b, v28.16b
	and v4.16b, v23.16b, v4.16b
	mov v16.s[2], v10.s[0]
	mov v22.s[2], v11.s[0]
	and v0.16b, v18.16b, v0.16b
	and v2.16b, v19.16b, v2.16b
	and v18.16b, v12.16b, v1.16b
	and v6.16b, v28.16b, v6.16b
	orr v4.16b, v4.16b, v17.16b
	and v17.16b, v14.16b, v1.16b
	ldur q1, [x29, #-80]
	orr v0.16b, v0.16b, v18.16b
	orr v5.16b, v6.16b, v5.16b
	orr v2.16b, v2.16b, v17.16b
	fmul v3.4s, v3.4s, v4.4s
	fmul v0.4s, v16.4s, v0.4s
	fmul v4.4s, v7.4s, v5.4s
	fmul v2.4s, v22.4s, v2.4s
	stp q3, q4, [x9, #-32]
	stp q0, q2, [x9]
	stur s30, [x9, #-20]
	stur s31, [x9, #-4]
	str s10, [x9, #12]
	str s11, [x9, #28]
	add x9, x9, #64
	b.ne .LBB5_2
.LBB5_3:
	tst x1, #0xc
	b.eq .LBB5_15
	lsr x8, x1, #4
	and x9, x1, #0xc
	neg x19, x9
	add x8, x0, x8, lsl #6
	add x20, x8, #8
	b .LBB5_6
.LBB5_5:
	fmul s0, s9, s3
	adds x19, x19, #4
	str s0, [x20], #16
	b.eq .LBB5_15
.LBB5_6:
	ldur s2, [x20, #-8]
	movi d8, #0000000000000000
	movi d0, #0000000000000000
	ldr s9, [x20, #4]
	fcmp s2, #0.0
	b.ls .LBB5_9
	fmov s0, #1.00000000
	fcmp s2, s0
	b.ge .LBB5_9
	fmov s0, s2
	bl powf
	ldur q1, [x29, #-80]
.LBB5_9:
	ldur s2, [x20, #-4]
	fmul s0, s9, s0
	fcmp s2, #0.0
	stur s0, [x20, #-8]
	b.ls .LBB5_12
	fmov s8, #1.00000000
	fcmp s2, s8
	b.ge .LBB5_12
	fmov s0, s2
	bl powf
	fmov s8, s0
	ldur q1, [x29, #-80]
.LBB5_12:
	ldr s0, [x20]
	fmul s2, s9, s8
	movi d3, #0000000000000000
	fcmp s0, #0.0
	stur s2, [x20, #-4]
	b.ls .LBB5_5
	fmov s3, #1.00000000
	fcmp s0, s3
	b.ge .LBB5_5
	bl powf
	fmov s3, s0
	ldur q1, [x29, #-80]
	b .LBB5_5
.LBB5_15:
	.cfi_def_cfa wsp, 336
	ldp x20, x19, [sp, #320]
	ldr x28, [sp, #304]
	ldp x29, x30, [sp, #288]
	ldp d9, d8, [sp, #272]
	ldp d11, d10, [sp, #256]
	ldp d13, d12, [sp, #240]
	ldp d15, d14, [sp, #224]
	add sp, sp, #336
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
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
