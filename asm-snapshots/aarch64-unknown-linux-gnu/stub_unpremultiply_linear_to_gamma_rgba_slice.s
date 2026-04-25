.section .text.stub_unpremultiply_linear_to_gamma_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_gamma_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_gamma_rgba_slice,@function
stub_unpremultiply_linear_to_gamma_rgba_slice:
	.cfi_startproc
	sub sp, sp, #368
	.cfi_def_cfa_offset 368
	stp d15, d14, [sp, #256]
	stp d13, d12, [sp, #272]
	stp d11, d10, [sp, #288]
	stp d9, d8, [sp, #304]
	stp x29, x30, [sp, #320]
	str x28, [sp, #336]
	stp x20, x19, [sp, #352]
	add x29, sp, #320
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
	lsl x8, x1, #2
	str s0, [x29, #24]
	ands x8, x8, #0x7fffffffffffffc0
	b.eq .LBB15_3
	fmov s1, #1.00000000
	ldr s0, [x29, #24]
	mov w9, #64269
	movk w9, #74, lsl #16
	mov w10, #27455
	mov w11, #29208
	dup v2.4s, w9
	mov w9, #1267
	movk w10, #16094, lsl #16
	fdiv s0, s1, s0
	movk w9, #16181, lsl #16
	fmov v3.4s, #1.00000000
	dup v1.4s, w9
	mov w9, #46233
	movk w11, #16177, lsl #16
	movk w9, #16147, lsl #16
	stp q0, q2, [x29, #-96]
	dup v0.4s, w10
	mov w10, #14483
	movk w10, #16246, lsl #16
	dup v2.4s, w10
	mov w10, #38156
	stp q0, q1, [x29, #-128]
	dup v1.4s, w9
	mov w9, #43579
	movk w9, #16440, lsl #16
	mvni v0.4s, #63, msl #16
	movk w10, #15389, lsl #16
	stur q1, [x29, #-144]
	dup v1.4s, w9
	mov w9, #-1023672320
	dup v24.4s, w9
	mov w9, #1123942400
	stp q1, q2, [sp, #144]
	fneg v2.4s, v0.4s
	dup v0.4s, w9
	mov w9, #63802
	mvni v1.4s, #127, msl #16
	movk w9, #14625, lsl #16
	stp q0, q2, [sp, #112]
	dup v2.4s, w9
	mov w9, #50297
	movk w9, #15022, lsl #16
	dup v0.4s, w9
	mov w9, #22716
	movk w9, #15715, lsl #16
	stp q0, q2, [sp, #80]
	fneg v0.4s, v1.4s
	dup v2.4s, w10
	mov w10, #65005
	dup v1.4s, w9
	add x9, x0, #32
	movk w10, #15989, lsl #16
	stp q0, q2, [sp, #48]
	dup v0.4s, w10
	mov w10, #981467136
	str w10, [x29, #28]
	stp q0, q1, [sp, #16]
	dup v0.4s, w11
	str q0, [sp]
.LBB15_2:
	fmov s5, #1.00000000
	ldur s11, [x9, #-20]
	ldur s12, [x9, #-4]
	ldr s16, [x29, #28]
	movi d7, #0000000000000000
	fmov v2.4s, #1.00000000
	ldr s13, [x9, #12]
	fmov v6.4s, #1.00000000
	ldr s14, [x9, #28]
	fdiv s0, s5, s11
	fcmp s11, s16
	movi v18.2d, #0000000000000000
	ldur q19, [x29, #-80]
	movi v20.4s, #127, msl #16
	ldur q21, [x29, #-112]
	fmov v23.4s, #-1.00000000
	ldp q26, q8, [x29, #-144]
	mvni v27.4s, #126
	ldr q10, [sp, #160]
	mvni v29.4s, #126
	mvni v30.4s, #126
	mov v9.16b, v26.16b
	mov v25.16b, v26.16b
	mov v28.16b, v10.16b
	fdiv s1, s5, s12
	fcsel s0, s0, s7, gt
	fcmp s12, s16
	mov v2.s[0], v0.s[0]
	fdiv s4, s5, s13
	mov v2.s[1], v0.s[0]
	fcsel s1, s1, s7, gt
	fcmp s13, s16
	mov v2.s[2], v0.s[0]
	ldur q0, [x9, #-32]
	mov v6.s[0], v1.s[0]
	fmul v17.4s, v0.4s, v2.4s
	fmov v2.4s, #1.00000000
	mov v6.s[1], v1.s[0]
	fdiv s0, s5, s14
	fmax v17.4s, v17.4s, v18.4s
	fcsel s22, s4, s7, gt
	fcmp s14, s16
	mov v6.s[2], v1.s[0]
	ldur q1, [x9, #-16]
	mov v2.s[0], v22.s[0]
	fmin v15.4s, v17.4s, v3.4s
	fmul v4.4s, v1.4s, v6.4s
	fmov v6.4s, #1.00000000
	mov v2.s[1], v22.s[0]
	add v1.4s, v15.4s, v19.4s
	fmax v17.4s, v4.4s, v18.4s
	fcsel s4, s0, s7, gt
	and v5.16b, v1.16b, v20.16b
	ssra v27.4s, v1.4s, #23
	mov v2.s[2], v22.s[0]
	ldr q22, [x9]
	mov v1.16b, v10.16b
	subs x8, x8, #64
	mov v6.s[0], v4.s[0]
	fmin v0.4s, v17.4s, v3.4s
	add v5.4s, v5.4s, v21.4s
	fmul v17.4s, v22.4s, v2.4s
	fadd v22.4s, v5.4s, v23.4s
	fadd v5.4s, v5.4s, v3.4s
	mov v6.s[1], v4.s[0]
	add v2.4s, v0.4s, v19.4s
	fmax v7.4s, v17.4s, v18.4s
	and v16.16b, v2.16b, v20.16b
	fdiv v17.4s, v22.4s, v5.4s
	ldr q5, [x9, #16]
	mov v6.s[2], v4.s[0]
	ssra v29.4s, v2.4s, #23
	fmin v4.4s, v7.4s, v3.4s
	add v7.4s, v16.4s, v21.4s
	fmul v5.4s, v5.4s, v6.4s
	fadd v16.4s, v7.4s, v23.4s
	fadd v7.4s, v7.4s, v3.4s
	add v6.4s, v4.4s, v19.4s
	fmax v5.4s, v5.4s, v18.4s
	and v22.16b, v6.16b, v20.16b
	fdiv v7.4s, v16.4s, v7.4s
	ssra v30.4s, v6.4s, #23
	add v16.4s, v22.4s, v21.4s
	fmin v22.4s, v5.4s, v3.4s
	fadd v5.4s, v16.4s, v23.4s
	fadd v16.4s, v16.4s, v3.4s
	add v18.4s, v22.4s, v19.4s
	fdiv v5.4s, v5.4s, v16.4s
	and v19.16b, v18.16b, v20.16b
	add v16.4s, v19.4s, v21.4s
	fmul v20.4s, v7.4s, v7.4s
	fadd v19.4s, v16.4s, v23.4s
	fadd v16.4s, v16.4s, v3.4s
	mov v23.16b, v26.16b
	fmla v25.4s, v8.4s, v20.4s
	fdiv v16.4s, v19.4s, v16.4s
	fmul v19.4s, v17.4s, v17.4s
	fmul v21.4s, v5.4s, v5.4s
	fmla v23.4s, v8.4s, v19.4s
	fmla v9.4s, v8.4s, v21.4s
	fmla v28.4s, v19.4s, v23.4s
	mov v23.16b, v10.16b
	fmla v1.4s, v21.4s, v9.4s
	ldr q9, [sp, #144]
	fmla v23.4s, v20.4s, v25.4s
	scvtf v25.4s, v27.4s
	scvtf v27.4s, v30.4s
	movi v30.4s, #67, lsl #24
	mov v2.16b, v9.16b
	fmul v31.4s, v16.4s, v16.4s
	mov v6.16b, v9.16b
	fmla v2.4s, v19.4s, v28.4s
	mov v19.16b, v26.16b
	fmla v6.4s, v20.4s, v23.4s
	mov v20.16b, v9.16b
	scvtf v23.4s, v29.4s
	mvni v26.4s, #126
	fmla v19.4s, v8.4s, v31.4s
	fmla v25.4s, v17.4s, v2.4s
	fcmeq v2.4s, v0.4s, #0.0
	fmla v20.4s, v21.4s, v1.4s
	fcmeq v21.4s, v15.4s, #0.0
	fmla v23.4s, v7.4s, v6.4s
	fcmeq v6.4s, v4.4s, #0.0
	ssra v26.4s, v18.4s, #23
	fcmlt v0.4s, v0.4s, #0.0
	fcmlt v7.4s, v15.4s, #0.0
	fmla v10.4s, v31.4s, v19.4s
	mvni v19.4s, #127, msl #16
	mov v1.16b, v2.16b
	fmla v27.4s, v5.4s, v20.4s
	mov v5.16b, v9.16b
	mov v18.16b, v21.16b
	fcmlt v2.4s, v4.4s, #0.0
	mov v4.16b, v6.16b
	scvtf v17.4s, v26.4s
	ldr q20, [sp, #128]
	fcmeq v6.4s, v22.4s, #0.0
	bsl v1.16b, v19.16b, v23.16b
	fmla v5.4s, v31.4s, v10.4s
	ldr q9, [sp]
	bsl v18.16b, v19.16b, v25.16b
	bsl v4.16b, v19.16b, v27.16b
	ldp q28, q23, [sp, #96]
	ldp q29, q27, [sp, #64]
	bsl v0.16b, v20.16b, v1.16b
	mov v1.16b, v2.16b
	fmla v17.4s, v16.4s, v5.4s
	bsl v7.16b, v20.16b, v18.16b
	ldur q18, [x29, #-96]
	fcmlt v5.4s, v22.4s, #0.0
	mov v22.16b, v27.16b
	mov v25.16b, v29.16b
	mov v26.16b, v29.16b
	bsl v1.16b, v20.16b, v4.16b
	ldp q8, q31, [sp, #16]
	bsl v6.16b, v19.16b, v17.16b
	fmul v2.4s, v0.4s, v18.s[0]
	fmul v4.4s, v7.4s, v18.s[0]
	fmul v1.4s, v1.4s, v18.s[0]
	bsl v5.16b, v20.16b, v6.16b
	fmax v6.4s, v2.4s, v24.4s
	fmax v0.4s, v4.4s, v24.4s
	fmax v7.4s, v1.4s, v24.4s
	fmin v16.4s, v0.4s, v30.4s
	fmul v0.4s, v5.4s, v18.s[0]
	fmin v5.4s, v6.4s, v30.4s
	fmin v6.4s, v7.4s, v30.4s
	frintn v7.4s, v16.4s
	fmax v17.4s, v0.4s, v24.4s
	frintn v18.4s, v5.4s
	frintn v19.4s, v6.4s
	fmin v7.4s, v7.4s, v23.4s
	fmin v17.4s, v17.4s, v30.4s
	fmin v18.4s, v18.4s, v23.4s
	fmin v19.4s, v19.4s, v23.4s
	fsub v16.4s, v16.4s, v7.4s
	frintn v20.4s, v17.4s
	fsub v5.4s, v5.4s, v18.4s
	fcvtns v7.4s, v7.4s
	fcvtns v18.4s, v18.4s
	fsub v21.4s, v6.4s, v19.4s
	mov v6.16b, v27.16b
	fcvtns v19.4s, v19.4s
	fmin v20.4s, v20.4s, v23.4s
	mov v23.16b, v27.16b
	fmla v22.4s, v28.4s, v5.4s
	fmla v6.4s, v28.4s, v16.4s
	shl v7.4s, v7.4s, #23
	shl v18.4s, v18.4s, #23
	shl v19.4s, v19.4s, #23
	fmla v23.4s, v28.4s, v21.4s
	fmla v26.4s, v5.4s, v22.4s
	mov v22.16b, v31.16b
	add v7.4s, v7.4s, v3.4s
	fmla v25.4s, v16.4s, v6.4s
	fsub v6.4s, v17.4s, v20.4s
	mov v17.16b, v29.16b
	fcvtns v20.4s, v20.4s
	add v18.4s, v18.4s, v3.4s
	fmla v17.4s, v21.4s, v23.4s
	mov v23.16b, v27.16b
	mov v27.16b, v31.16b
	fmla v22.4s, v16.4s, v25.4s
	mov v25.16b, v31.16b
	shl v20.4s, v20.4s, #23
	fmla v23.4s, v28.4s, v6.4s
	fmla v27.4s, v5.4s, v26.4s
	mov v26.16b, v29.16b
	fmla v25.4s, v21.4s, v17.4s
	mov v17.16b, v8.16b
	mov v28.16b, v8.16b
	mov v29.16b, v9.16b
	fmla v17.4s, v16.4s, v22.4s
	fmla v26.4s, v6.4s, v23.4s
	mov v22.16b, v8.16b
	fmla v28.4s, v5.4s, v27.4s
	mov v27.16b, v31.16b
	mov v23.16b, v9.16b
	fmla v22.4s, v21.4s, v25.4s
	fmov v25.4s, #1.00000000
	fmla v27.4s, v6.4s, v26.4s
	mov v26.16b, v9.16b
	fmla v23.4s, v16.4s, v17.4s
	fmov v17.4s, #1.00000000
	fmla v29.4s, v5.4s, v28.4s
	fmov v28.4s, #1.00000000
	fmla v26.4s, v21.4s, v22.4s
	mov v22.16b, v8.16b
	fmla v25.4s, v16.4s, v23.4s
	fcmge v16.4s, v4.4s, v30.4s
	fcmgt v4.4s, v24.4s, v4.4s
	fmla v17.4s, v5.4s, v29.4s
	mov v5.16b, v9.16b
	fcmge v23.4s, v2.4s, v30.4s
	fmla v22.4s, v6.4s, v27.4s
	fcmgt v2.4s, v24.4s, v2.4s
	fmla v28.4s, v21.4s, v26.4s
	fmov v21.4s, #1.00000000
	mvn v26.16b, v16.16b
	fmul v7.4s, v25.4s, v7.4s
	mvn v25.16b, v23.16b
	fmla v5.4s, v6.4s, v22.4s
	fcmge v22.4s, v1.4s, v30.4s
	fcmgt v1.4s, v24.4s, v1.4s
	bic v4.16b, v26.16b, v4.16b
	fcmge v26.4s, v0.4s, v30.4s
	fcmgt v0.4s, v24.4s, v0.4s
	bic v2.16b, v25.16b, v2.16b
	fmla v21.4s, v6.4s, v5.4s
	fmul v6.4s, v17.4s, v18.4s
	and v4.16b, v4.16b, v7.16b
	add v7.4s, v20.4s, v3.4s
	add v5.4s, v19.4s, v3.4s
	ldr q19, [sp, #48]
	mvn v17.16b, v22.16b
	mvn v18.16b, v26.16b
	and v16.16b, v16.16b, v19.16b
	and v2.16b, v2.16b, v6.16b
	fmul v6.4s, v21.4s, v7.4s
	and v7.16b, v23.16b, v19.16b
	fmul v5.4s, v28.4s, v5.4s
	bic v1.16b, v17.16b, v1.16b
	bic v0.16b, v18.16b, v0.16b
	orr v4.16b, v4.16b, v16.16b
	orr v2.16b, v2.16b, v7.16b
	and v0.16b, v0.16b, v6.16b
	and v1.16b, v1.16b, v5.16b
	and v5.16b, v22.16b, v19.16b
	stp q4, q2, [x9, #-32]
	and v4.16b, v26.16b, v19.16b
	stur s11, [x9, #-20]
	orr v1.16b, v1.16b, v5.16b
	stur s12, [x9, #-4]
	orr v0.16b, v0.16b, v4.16b
	stp q1, q0, [x9]
	str s13, [x9, #12]
	str s14, [x9, #28]
	add x9, x9, #64
	b.ne .LBB15_2
.LBB15_3:
	tst x1, #0xc
	b.eq .LBB15_17
	fmov s10, #1.00000000
	ldr s0, [x29, #24]
	lsr x8, x1, #4
	mov w10, #981467136
	and x9, x1, #0xc
	add x8, x0, x8, lsl #6
	fmov s11, w10
	neg x19, x9
	fdiv s8, s10, s0
	add x20, x8, #8
	b .LBB15_6
.LBB15_5:
	str s0, [x20], #16
	adds x19, x19, #4
	b.eq .LBB15_17
.LBB15_6:
	ldr s0, [x20, #4]
	fcmp s0, s11
	b.le .LBB15_16
	fdiv s12, s10, s0
	ldur s0, [x20, #-8]
	movi d9, #0000000000000000
	fmul s1, s12, s0
	movi d0, #0000000000000000
	fcmp s1, #0.0
	b.ls .LBB15_10
	fmov s0, #1.00000000
	fcmp s1, s0
	b.ge .LBB15_10
	fmov s0, s1
	fmov s1, s8
	bl powf
.LBB15_10:
	ldur s1, [x20, #-4]
	stur s0, [x20, #-8]
	fmul s1, s12, s1
	fcmp s1, #0.0
	b.ls .LBB15_13
	fmov s9, #1.00000000
	fcmp s1, s9
	b.ge .LBB15_13
	fmov s0, s1
	fmov s1, s8
	bl powf
	fmov s9, s0
.LBB15_13:
	ldr s0, [x20]
	stur s9, [x20, #-4]
	fmul s1, s12, s0
	movi d0, #0000000000000000
	fcmp s1, #0.0
	b.ls .LBB15_5
	fmov s0, #1.00000000
	fcmp s1, s0
	b.ge .LBB15_5
	fmov s0, s1
	fmov s1, s8
	bl powf
	b .LBB15_5
.LBB15_16:
	stur xzr, [x20, #-8]
	str wzr, [x20], #16
	adds x19, x19, #4
	b.ne .LBB15_6
.LBB15_17:
	.cfi_def_cfa wsp, 368
	ldp x20, x19, [sp, #352]
	ldr x28, [sp, #336]
	ldp x29, x30, [sp, #320]
	ldp d9, d8, [sp, #304]
	ldp d11, d10, [sp, #288]
	ldp d13, d12, [sp, #272]
	ldp d15, d14, [sp, #256]
	add sp, sp, #368
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
