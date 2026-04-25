.section .text.stub_gamma_to_linear_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_slice
	.p2align	2
.type	stub_gamma_to_linear_slice,@function
stub_gamma_to_linear_slice:
	.cfi_startproc
	sub sp, sp, #304
	.cfi_def_cfa_offset 304
	stp d15, d14, [sp, #192]
	stp d13, d12, [sp, #208]
	stp d11, d10, [sp, #224]
	stp d9, d8, [sp, #240]
	stp x29, x30, [sp, #256]
	stp x28, x21, [sp, #272]
	stp x20, x19, [sp, #288]
	add x29, sp, #256
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
	fmov s1, s0
	lsl x8, x1, #2
	ands x9, x8, #0x7fffffffffffffc0
	stur q1, [x29, #-80]
	b.eq .LBB6_3
	mov w10, #64269
	mov w11, #1267
	mvni v19.4s, #63, msl #16
	movk w10, #74, lsl #16
	movk w11, #16181, lsl #16
	mov w12, #29208
	dup v1.4s, w10
	dup v0.4s, w11
	mov w10, #27455
	movk w10, #16094, lsl #16
	mov w11, #14483
	fneg v2.4s, v19.4s
	movk w11, #16246, lsl #16
	fmov v22.4s, #1.00000000
	movk w12, #16177, lsl #16
	stp q0, q1, [x29, #-112]
	dup v1.4s, w10
	mov w10, #46233
	movk w10, #16147, lsl #16
	dup v29.4s, w12
	add x9, x0, x9
	dup v0.4s, w10
	mov w10, #43579
	movk w10, #16440, lsl #16
	stp q0, q1, [sp, #112]
	dup v1.4s, w11
	dup v0.4s, w10
	mov w10, #-1023672320
	mov w11, #38156
	dup v20.4s, w10
	mov w10, #1123942400
	movk w11, #15389, lsl #16
	stp q0, q1, [sp, #80]
	dup v1.4s, w10
	mov w10, #63802
	movk w10, #14625, lsl #16
	mvni v0.4s, #127, msl #16
	stp q1, q2, [sp, #48]
	dup v2.4s, w10
	mov w10, #50297
	movk w10, #15022, lsl #16
	fneg v26.4s, v0.4s
	dup v1.4s, w10
	mov w10, #22716
	movk w10, #15715, lsl #16
	dup v27.4s, w10
	mov x10, x0
	stp q1, q2, [sp, #16]
	dup v1.4s, w11
	mov w11, #65005
	movk w11, #15989, lsl #16
	dup v28.4s, w11
	str q1, [sp]
.LBB6_2:
	movi v0.2d, #0000000000000000
	ldp q30, q8, [x10]
	ldp q3, q2, [x29, #-112]
	movi v1.4s, #127, msl #16
	fmov v4.4s, #-1.00000000
	ldp q13, q11, [x10, #32]
	ldp q25, q17, [sp, #80]
	mvni v6.4s, #126
	fmax v30.4s, v30.4s, v0.4s
	fmax v8.4s, v8.4s, v0.4s
	mvni v19.4s, #126
	fmax v13.4s, v13.4s, v0.4s
	fmax v11.4s, v11.4s, v0.4s
	mov v16.16b, v17.16b
	mov v24.16b, v17.16b
	mov v23.16b, v25.16b
	fmin v30.4s, v30.4s, v22.4s
	fmin v8.4s, v8.4s, v22.4s
	fmin v11.4s, v11.4s, v22.4s
	add v31.4s, v30.4s, v2.4s
	and v9.16b, v31.16b, v1.16b
	ssra v6.4s, v31.4s, #23
	add v9.4s, v9.4s, v3.4s
	scvtf v6.4s, v6.4s
	fadd v10.4s, v9.4s, v4.4s
	fadd v12.4s, v9.4s, v22.4s
	add v9.4s, v8.4s, v2.4s
	and v14.16b, v9.16b, v1.16b
	ssra v19.4s, v9.4s, #23
	fdiv v12.4s, v10.4s, v12.4s
	fmin v10.4s, v13.4s, v22.4s
	add v13.4s, v14.4s, v3.4s
	fadd v14.4s, v13.4s, v4.4s
	fadd v15.4s, v13.4s, v22.4s
	add v13.4s, v10.4s, v2.4s
	add v2.4s, v11.4s, v2.4s
	and v0.16b, v13.16b, v1.16b
	and v5.16b, v2.16b, v1.16b
	fdiv v14.4s, v14.4s, v15.4s
	add v0.4s, v0.4s, v3.4s
	add v5.4s, v5.4s, v3.4s
	ldp q3, q1, [sp, #112]
	fadd v15.4s, v0.4s, v4.4s
	fadd v0.4s, v0.4s, v22.4s
	mov v7.16b, v3.16b
	mov v21.16b, v3.16b
	mov v9.16b, v3.16b
	fdiv v0.4s, v15.4s, v0.4s
	fadd v15.4s, v5.4s, v4.4s
	fadd v5.4s, v5.4s, v22.4s
	fmul v18.4s, v14.4s, v14.4s
	fmla v21.4s, v1.4s, v18.4s
	fdiv v5.4s, v15.4s, v5.4s
	fmul v15.4s, v12.4s, v12.4s
	fmla v24.4s, v18.4s, v21.4s
	mvni v21.4s, #126
	fmul v4.4s, v0.4s, v0.4s
	fmla v7.4s, v1.4s, v15.4s
	ssra v21.4s, v13.4s, #23
	fmla v16.4s, v15.4s, v7.4s
	mov v7.16b, v3.16b
	fcmeq v3.4s, v10.4s, #0.0
	fmla v7.4s, v1.4s, v4.4s
	fmla v23.4s, v15.4s, v16.4s
	mov v15.16b, v17.16b
	mov v16.16b, v25.16b
	fmul v31.4s, v5.4s, v5.4s
	fmla v15.4s, v4.4s, v7.4s
	fmla v16.4s, v18.4s, v24.4s
	mvni v7.4s, #126
	scvtf v24.4s, v19.4s
	fmla v6.4s, v12.4s, v23.4s
	fcmeq v18.4s, v30.4s, #0.0
	fmla v9.4s, v1.4s, v31.4s
	mov v19.16b, v25.16b
	mvni v23.4s, #127, msl #16
	ssra v7.4s, v2.4s, #23
	scvtf v2.4s, v21.4s
	fcmlt v21.4s, v30.4s, #0.0
	ldur q1, [x29, #-80]
	fmla v19.4s, v4.4s, v15.4s
	fcmeq v4.4s, v8.4s, #0.0
	fmla v24.4s, v14.4s, v16.4s
	fmla v17.4s, v31.4s, v9.4s
	bit v6.16b, v23.16b, v18.16b
	ldr q18, [sp, #64]
	scvtf v7.4s, v7.4s
	fmla v2.4s, v0.4s, v19.4s
	fcmlt v0.4s, v8.4s, #0.0
	bsl v4.16b, v23.16b, v24.16b
	fmla v25.4s, v31.4s, v17.4s
	bit v6.16b, v18.16b, v21.16b
	fcmeq v17.4s, v11.4s, #0.0
	bit v2.16b, v23.16b, v3.16b
	bsl v0.16b, v18.16b, v4.16b
	fcmlt v3.4s, v11.4s, #0.0
	fmla v7.4s, v5.4s, v25.4s
	fcmlt v5.4s, v10.4s, #0.0
	fmul v9.4s, v6.4s, v1.s[0]
	mov v4.16b, v17.16b
	ldp q12, q25, [sp]
	mov v10.16b, v27.16b
	mov v11.16b, v27.16b
	fmul v8.4s, v0.4s, v1.s[0]
	bsl v4.16b, v23.16b, v7.16b
	bit v2.16b, v18.16b, v5.16b
	fmax v0.4s, v9.4s, v20.4s
	movi v23.4s, #67, lsl #24
	mov v17.16b, v25.16b
	mov v19.16b, v12.16b
	mov v21.16b, v12.16b
	bsl v3.16b, v18.16b, v4.16b
	fmul v31.4s, v2.4s, v1.s[0]
	fmax v2.4s, v8.4s, v20.4s
	fmin v0.4s, v0.4s, v23.4s
	ldp q24, q18, [sp, #32]
	fmul v30.4s, v3.4s, v1.s[0]
	fmax v3.4s, v31.4s, v20.4s
	fmin v2.4s, v2.4s, v23.4s
	frintn v4.4s, v0.4s
	fmin v3.4s, v3.4s, v23.4s
	fmax v5.4s, v30.4s, v20.4s
	frintn v6.4s, v2.4s
	fmin v4.4s, v4.4s, v18.4s
	frintn v7.4s, v3.4s
	fmin v5.4s, v5.4s, v23.4s
	fmin v6.4s, v6.4s, v18.4s
	fsub v0.4s, v0.4s, v4.4s
	fcvtns v4.4s, v4.4s
	fmin v7.4s, v7.4s, v18.4s
	frintn v16.4s, v5.4s
	fsub v2.4s, v2.4s, v6.4s
	fmla v17.4s, v24.4s, v0.4s
	fcvtns v6.4s, v6.4s
	shl v4.4s, v4.4s, #23
	fsub v3.4s, v3.4s, v7.4s
	fmin v16.4s, v16.4s, v18.4s
	mov v18.16b, v25.16b
	fmla v19.4s, v0.4s, v17.4s
	mov v17.16b, v25.16b
	fcvtns v7.4s, v7.4s
	shl v6.4s, v6.4s, #23
	add v4.4s, v4.4s, v22.4s
	fmla v18.4s, v24.4s, v2.4s
	fmla v17.4s, v24.4s, v3.4s
	fsub v5.4s, v5.4s, v16.4s
	fcvtns v16.4s, v16.4s
	fmla v10.4s, v0.4s, v19.4s
	mov v19.16b, v12.16b
	shl v7.4s, v7.4s, #23
	add v6.4s, v6.4s, v22.4s
	fmla v21.4s, v2.4s, v18.4s
	mov v18.16b, v25.16b
	fmla v19.4s, v3.4s, v17.4s
	mov v17.16b, v28.16b
	shl v16.4s, v16.4s, #23
	add v7.4s, v7.4s, v22.4s
	fmla v18.4s, v24.4s, v5.4s
	fmla v17.4s, v0.4s, v10.4s
	mov v10.16b, v27.16b
	fmla v11.4s, v2.4s, v21.4s
	mov v21.16b, v12.16b
	add v16.4s, v16.4s, v22.4s
	fmla v10.4s, v3.4s, v19.4s
	mov v19.16b, v29.16b
	fmla v21.4s, v5.4s, v18.4s
	mov v18.16b, v28.16b
	fmla v19.4s, v0.4s, v17.4s
	mov v17.16b, v27.16b
	fmla v18.4s, v2.4s, v11.4s
	fmov v11.4s, #1.00000000
	fmla v17.4s, v5.4s, v21.4s
	mov v21.16b, v28.16b
	fmla v11.4s, v0.4s, v19.4s
	mov v19.16b, v28.16b
	mov v0.16b, v29.16b
	fmla v21.4s, v3.4s, v10.4s
	mov v10.16b, v29.16b
	fmla v19.4s, v5.4s, v17.4s
	fmov v17.4s, #1.00000000
	fmla v10.4s, v2.4s, v18.4s
	fmov v18.4s, #1.00000000
	fmul v4.4s, v11.4s, v4.4s
	fmla v0.4s, v3.4s, v21.4s
	mov v21.16b, v29.16b
	fmla v18.4s, v2.4s, v10.4s
	fmov v2.4s, #1.00000000
	fmla v21.4s, v5.4s, v19.4s
	fcmge v10.4s, v9.4s, v23.4s
	fcmge v19.4s, v8.4s, v23.4s
	fmla v17.4s, v3.4s, v0.4s
	fcmge v0.4s, v31.4s, v23.4s
	fcmgt v3.4s, v20.4s, v31.4s
	fcmge v31.4s, v30.4s, v23.4s
	fcmgt v9.4s, v20.4s, v9.4s
	fcmgt v8.4s, v20.4s, v8.4s
	fmla v2.4s, v5.4s, v21.4s
	fcmgt v21.4s, v20.4s, v30.4s
	fmul v6.4s, v18.4s, v6.4s
	mvn v5.16b, v10.16b
	mvn v30.16b, v19.16b
	fmul v7.4s, v17.4s, v7.4s
	mvn v11.16b, v0.16b
	mvn v18.16b, v31.16b
	and v17.16b, v19.16b, v26.16b
	and v0.16b, v0.16b, v26.16b
	fmul v2.4s, v2.4s, v16.4s
	bic v5.16b, v5.16b, v9.16b
	bic v30.16b, v30.16b, v8.16b
	bic v3.16b, v11.16b, v3.16b
	bic v16.16b, v18.16b, v21.16b
	and v4.16b, v5.16b, v4.16b
	and v5.16b, v10.16b, v26.16b
	and v6.16b, v30.16b, v6.16b
	and v3.16b, v3.16b, v7.16b
	and v7.16b, v31.16b, v26.16b
	and v2.16b, v16.16b, v2.16b
	orr v4.16b, v4.16b, v5.16b
	orr v5.16b, v6.16b, v17.16b
	orr v0.16b, v3.16b, v0.16b
	orr v2.16b, v2.16b, v7.16b
	stp q4, q5, [x10]
	stp q0, q2, [x10, #32]
	add x10, x10, #64
	cmp x10, x9
	b.ne .LBB6_2
.LBB6_3:
	ands x19, x8, #0x3c
	b.eq .LBB6_9
	and x8, x1, #0x1ffffffffffffff0
	add x21, x0, x8, lsl #2
	mov x20, x21
	b .LBB6_6
.LBB6_5:
	subs x19, x19, #4
	str s0, [x21]
	mov x21, x20
	b.eq .LBB6_9
.LBB6_6:
	ldr s2, [x20], #4
	movi d0, #0000000000000000
	fcmp s2, #0.0
	b.ls .LBB6_5
	fmov s0, #1.00000000
	fcmp s2, s0
	b.ge .LBB6_5
	fmov s0, s2
	bl powf
	ldur q1, [x29, #-80]
	b .LBB6_5
.LBB6_9:
	.cfi_def_cfa wsp, 304
	ldp x20, x19, [sp, #288]
	ldp x28, x21, [sp, #272]
	ldp x29, x30, [sp, #256]
	ldp d9, d8, [sp, #240]
	ldp d11, d10, [sp, #224]
	ldp d13, d12, [sp, #208]
	ldp d15, d14, [sp, #192]
	add sp, sp, #304
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
