.section .text.stub_linear_to_gamma_slice,"ax",@progbits
	.globl	stub_linear_to_gamma_slice
	.p2align	2
.type	stub_linear_to_gamma_slice,@function
stub_linear_to_gamma_slice:
	.cfi_startproc
	sub sp, sp, #320
	.cfi_def_cfa_offset 320
	stp d15, d14, [sp, #208]
	stp d13, d12, [sp, #224]
	stp d11, d10, [sp, #240]
	stp d9, d8, [sp, #256]
	stp x29, x30, [sp, #272]
	stp x28, x21, [sp, #288]
	stp x20, x19, [sp, #304]
	add x29, sp, #272
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
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
	str s0, [sp, #12]
	ands x9, x8, #0x7fffffffffffffc0
	b.eq .LBB7_3
	mov w10, #64269
	fmov s5, #1.00000000
	mov w11, #27455
	movk w10, #74, lsl #16
	movk w11, #16094, lsl #16
	mvni v21.4s, #63, msl #16
	dup v0.4s, w10
	mov w10, #1267
	mov w12, #29208
	movk w10, #16181, lsl #16
	fmov v2.4s, #1.00000000
	movk w12, #16177, lsl #16
	dup v1.4s, w10
	mov w10, #46233
	dup v31.4s, w12
	stur q0, [x29, #-80]
	ldr s0, [sp, #12]
	movk w10, #16147, lsl #16
	add x9, x0, x9
	fdiv s0, s5, s0
	stur q0, [x29, #-96]
	dup v0.4s, w11
	mov w11, #14483
	movk w11, #16246, lsl #16
	stp q0, q1, [x29, #-128]
	dup v1.4s, w10
	dup v0.4s, w11
	mov w10, #43579
	mov w11, #38156
	movk w10, #16440, lsl #16
	movk w11, #15389, lsl #16
	stp q0, q1, [sp, #112]
	fneg v1.4s, v21.4s
	dup v3.4s, w10
	mov w10, #-1023672320
	mvni v0.4s, #127, msl #16
	dup v22.4s, w10
	mov w10, #1123942400
	stp q1, q3, [sp, #80]
	dup v3.4s, w10
	mov w10, #63802
	movk w10, #14625, lsl #16
	fneg v28.4s, v0.4s
	dup v1.4s, w10
	mov w10, #50297
	movk w10, #15022, lsl #16
	stp q1, q3, [sp, #48]
	dup v3.4s, w10
	dup v1.4s, w11
	mov w10, #22716
	mov w11, #65005
	movk w10, #15715, lsl #16
	movk w11, #15989, lsl #16
	dup v29.4s, w10
	dup v30.4s, w11
	mov x10, x0
	stp q1, q3, [sp, #16]
.LBB7_2:
	movi v3.2d, #0000000000000000
	ldp q8, q10, [x10]
	ldur q5, [x29, #-80]
	movi v6.4s, #127, msl #16
	ldp q24, q16, [x29, #-128]
	fmov v17.4s, #-1.00000000
	ldp q15, q13, [x10, #32]
	mvni v21.4s, #126
	fmax v8.4s, v8.4s, v3.4s
	fmax v10.4s, v10.4s, v3.4s
	fmax v15.4s, v15.4s, v3.4s
	fmax v13.4s, v13.4s, v3.4s
	fmin v8.4s, v8.4s, v2.4s
	fmin v10.4s, v10.4s, v2.4s
	fmin v13.4s, v13.4s, v2.4s
	add v9.4s, v8.4s, v5.4s
	add v7.4s, v13.4s, v5.4s
	and v11.16b, v9.16b, v6.16b
	and v20.16b, v7.16b, v6.16b
	add v11.4s, v11.4s, v16.4s
	fadd v12.4s, v11.4s, v17.4s
	fadd v14.4s, v11.4s, v2.4s
	add v11.4s, v10.4s, v5.4s
	and v1.16b, v11.16b, v6.16b
	ssra v21.4s, v11.4s, #23
	fdiv v14.4s, v12.4s, v14.4s
	fmin v12.4s, v15.4s, v2.4s
	add v1.4s, v1.4s, v16.4s
	add v15.4s, v12.4s, v5.4s
	ldp q19, q5, [sp, #112]
	fadd v0.4s, v1.4s, v17.4s
	fadd v1.4s, v1.4s, v2.4s
	and v4.16b, v15.16b, v6.16b
	mov v23.16b, v5.16b
	mov v18.16b, v19.16b
	mov v26.16b, v19.16b
	mov v11.16b, v5.16b
	fdiv v27.4s, v0.4s, v1.4s
	add v1.4s, v4.4s, v16.4s
	ldr q0, [sp, #96]
	mov v25.16b, v0.16b
	fadd v4.4s, v1.4s, v17.4s
	fadd v1.4s, v1.4s, v2.4s
	fdiv v1.4s, v4.4s, v1.4s
	add v4.4s, v20.4s, v16.4s
	mvni v16.4s, #126
	fadd v20.4s, v4.4s, v17.4s
	fadd v4.4s, v4.4s, v2.4s
	mov v17.16b, v5.16b
	fmul v3.4s, v27.4s, v27.4s
	ssra v16.4s, v9.4s, #23
	scvtf v16.4s, v16.4s
	fmla v23.4s, v24.4s, v3.4s
	fdiv v4.4s, v20.4s, v4.4s
	fmul v20.4s, v14.4s, v14.4s
	fmla v26.4s, v3.4s, v23.4s
	mvni v23.4s, #126
	fmul v6.4s, v1.4s, v1.4s
	fmla v17.4s, v24.4s, v20.4s
	ssra v23.4s, v15.4s, #23
	fmla v18.4s, v20.4s, v17.4s
	mov v17.16b, v5.16b
	mov v5.16b, v19.16b
	fmla v17.4s, v24.4s, v6.4s
	fmla v25.4s, v20.4s, v18.4s
	mov v20.16b, v19.16b
	mov v18.16b, v0.16b
	fmul v9.4s, v4.4s, v4.4s
	fcmeq v19.4s, v8.4s, #0.0
	fmla v20.4s, v6.4s, v17.4s
	fmla v18.4s, v3.4s, v26.4s
	mvni v3.4s, #126
	scvtf v17.4s, v21.4s
	mov v21.16b, v0.16b
	fmla v16.4s, v14.4s, v25.4s
	fmla v11.4s, v24.4s, v9.4s
	movi v24.4s, #67, lsl #24
	ssra v3.4s, v7.4s, #23
	scvtf v7.4s, v23.4s
	fmla v21.4s, v6.4s, v20.4s
	mvni v20.4s, #127, msl #16
	fcmeq v6.4s, v10.4s, #0.0
	fmla v17.4s, v27.4s, v18.4s
	fcmlt v18.4s, v8.4s, #0.0
	fmla v5.4s, v9.4s, v11.4s
	scvtf v3.4s, v3.4s
	bit v16.16b, v20.16b, v19.16b
	ldr q19, [sp, #80]
	fmla v7.4s, v1.4s, v21.4s
	fcmlt v1.4s, v10.4s, #0.0
	mov v21.16b, v29.16b
	bsl v6.16b, v20.16b, v17.16b
	fcmeq v17.4s, v13.4s, #0.0
	fmla v0.4s, v9.4s, v5.4s
	fcmeq v5.4s, v12.4s, #0.0
	bit v16.16b, v19.16b, v18.16b
	ldp q23, q18, [sp, #48]
	ldp q26, q25, [sp, #16]
	bsl v1.16b, v19.16b, v6.16b
	ldur q6, [x29, #-96]
	fmla v3.4s, v4.4s, v0.4s
	mov v4.16b, v5.16b
	fcmlt v0.4s, v12.4s, #0.0
	fmul v11.4s, v16.4s, v6.s[0]
	fcmlt v5.4s, v13.4s, #0.0
	bsl v4.16b, v20.16b, v7.16b
	fmul v10.4s, v1.4s, v6.s[0]
	bit v3.16b, v20.16b, v17.16b
	mov v17.16b, v25.16b
	mov v20.16b, v26.16b
	fmax v1.4s, v11.4s, v22.4s
	bsl v0.16b, v19.16b, v4.16b
	bit v3.16b, v19.16b, v5.16b
	mov v19.16b, v26.16b
	fmin v1.4s, v1.4s, v24.4s
	fmul v9.4s, v0.4s, v6.s[0]
	fmax v0.4s, v10.4s, v22.4s
	fmul v8.4s, v3.4s, v6.s[0]
	frintn v4.4s, v1.4s
	fmax v3.4s, v9.4s, v22.4s
	fmin v0.4s, v0.4s, v24.4s
	fmax v5.4s, v8.4s, v22.4s
	fmin v4.4s, v4.4s, v18.4s
	fmin v3.4s, v3.4s, v24.4s
	frintn v6.4s, v0.4s
	fmin v5.4s, v5.4s, v24.4s
	fsub v1.4s, v1.4s, v4.4s
	fcvtns v4.4s, v4.4s
	frintn v7.4s, v3.4s
	fmin v6.4s, v6.4s, v18.4s
	frintn v16.4s, v5.4s
	fmla v17.4s, v23.4s, v1.4s
	shl v4.4s, v4.4s, #23
	fmin v7.4s, v7.4s, v18.4s
	fsub v0.4s, v0.4s, v6.4s
	fcvtns v6.4s, v6.4s
	fmin v16.4s, v16.4s, v18.4s
	mov v18.16b, v25.16b
	add v4.4s, v4.4s, v2.4s
	fmla v19.4s, v1.4s, v17.4s
	mov v17.16b, v25.16b
	fsub v3.4s, v3.4s, v7.4s
	fmla v18.4s, v23.4s, v0.4s
	fcvtns v7.4s, v7.4s
	fsub v5.4s, v5.4s, v16.4s
	fcvtns v16.4s, v16.4s
	shl v6.4s, v6.4s, #23
	fmla v21.4s, v1.4s, v19.4s
	mov v19.16b, v26.16b
	fmla v17.4s, v23.4s, v3.4s
	fmla v20.4s, v0.4s, v18.4s
	mov v18.16b, v25.16b
	shl v7.4s, v7.4s, #23
	shl v16.4s, v16.4s, #23
	add v6.4s, v6.4s, v2.4s
	fmla v18.4s, v23.4s, v5.4s
	mov v23.16b, v29.16b
	fmla v19.4s, v3.4s, v17.4s
	mov v17.16b, v30.16b
	add v7.4s, v7.4s, v2.4s
	add v16.4s, v16.4s, v2.4s
	fmla v23.4s, v0.4s, v20.4s
	mov v20.16b, v26.16b
	fmla v17.4s, v1.4s, v21.4s
	mov v21.16b, v29.16b
	fmla v20.4s, v5.4s, v18.4s
	mov v18.16b, v30.16b
	fmla v21.4s, v3.4s, v19.4s
	mov v19.16b, v31.16b
	fmla v18.4s, v0.4s, v23.4s
	fmov v23.4s, #1.00000000
	fmla v19.4s, v1.4s, v17.4s
	mov v17.16b, v29.16b
	fmla v17.4s, v5.4s, v20.4s
	mov v20.16b, v30.16b
	fmla v23.4s, v1.4s, v19.4s
	mov v19.16b, v30.16b
	mov v1.16b, v31.16b
	fmla v20.4s, v3.4s, v21.4s
	mov v21.16b, v31.16b
	fmla v19.4s, v5.4s, v17.4s
	fmov v17.4s, #1.00000000
	fmul v4.4s, v23.4s, v4.4s
	fmla v21.4s, v0.4s, v18.4s
	fmov v18.4s, #1.00000000
	fmla v1.4s, v3.4s, v20.4s
	mov v20.16b, v31.16b
	fmla v18.4s, v0.4s, v21.4s
	fmov v0.4s, #1.00000000
	fmla v20.4s, v5.4s, v19.4s
	fcmge v21.4s, v11.4s, v24.4s
	fcmge v19.4s, v10.4s, v24.4s
	fmla v17.4s, v3.4s, v1.4s
	fcmge v1.4s, v9.4s, v24.4s
	fcmgt v3.4s, v22.4s, v9.4s
	fcmge v9.4s, v8.4s, v24.4s
	fcmgt v11.4s, v22.4s, v11.4s
	fcmgt v10.4s, v22.4s, v10.4s
	fmla v0.4s, v5.4s, v20.4s
	fcmgt v20.4s, v22.4s, v8.4s
	fmul v6.4s, v18.4s, v6.4s
	mvn v5.16b, v21.16b
	mvn v8.16b, v19.16b
	fmul v7.4s, v17.4s, v7.4s
	mvn v23.16b, v1.16b
	mvn v18.16b, v9.16b
	and v17.16b, v19.16b, v28.16b
	and v1.16b, v1.16b, v28.16b
	fmul v0.4s, v0.4s, v16.4s
	bic v5.16b, v5.16b, v11.16b
	bic v8.16b, v8.16b, v10.16b
	bic v3.16b, v23.16b, v3.16b
	bic v16.16b, v18.16b, v20.16b
	and v4.16b, v5.16b, v4.16b
	and v5.16b, v21.16b, v28.16b
	and v6.16b, v8.16b, v6.16b
	and v3.16b, v3.16b, v7.16b
	and v7.16b, v9.16b, v28.16b
	and v0.16b, v16.16b, v0.16b
	orr v4.16b, v4.16b, v5.16b
	orr v5.16b, v6.16b, v17.16b
	orr v1.16b, v3.16b, v1.16b
	orr v0.16b, v0.16b, v7.16b
	stp q4, q5, [x10]
	stp q1, q0, [x10, #32]
	add x10, x10, #64
	cmp x10, x9
	b.ne .LBB7_2
.LBB7_3:
	ands x19, x8, #0x3c
	b.eq .LBB7_9
	fmov s0, #1.00000000
	ldr s1, [sp, #12]
	and x8, x1, #0x1ffffffffffffff0
	add x21, x0, x8, lsl #2
	fdiv s8, s0, s1
	mov x20, x21
	b .LBB7_6
.LBB7_5:
	subs x19, x19, #4
	str s0, [x21]
	mov x21, x20
	b.eq .LBB7_9
.LBB7_6:
	ldr s1, [x20], #4
	movi d0, #0000000000000000
	fcmp s1, #0.0
	b.ls .LBB7_5
	fmov s0, #1.00000000
	fcmp s1, s0
	b.ge .LBB7_5
	fmov s0, s1
	fmov s1, s8
	bl powf
	b .LBB7_5
.LBB7_9:
	.cfi_def_cfa wsp, 320
	ldp x20, x19, [sp, #304]
	ldp x28, x21, [sp, #288]
	ldp x29, x30, [sp, #272]
	ldp d9, d8, [sp, #256]
	ldp d11, d10, [sp, #240]
	ldp d13, d12, [sp, #224]
	ldp d15, d14, [sp, #208]
	add sp, sp, #320
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
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
