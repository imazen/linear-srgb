.section .text.stub_gamma_to_linear_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_slice
	.p2align	2
.type	stub_gamma_to_linear_slice,@function
stub_gamma_to_linear_slice:
	.cfi_startproc
	stp d15, d14, [sp, #-112]!
	.cfi_def_cfa_offset 112
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	stp x29, x30, [sp, #64]
	str x28, [sp, #80]
	stp x20, x19, [sp, #96]
	add x29, sp, #64
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
	.cfi_remember_state
	sub sp, sp, #400
	str s0, [x29, #28]
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB9_18
	cmp w8, #2
	b.ne .LBB9_17
.LBB9_2:
	ldr s0, [x29, #28]
	fcmp s0, #0.0
	b.lt .LBB9_19
	fmov s1, #4.00000000
	ldr s0, [x29, #28]
	fcmp s0, s1
	b.hi .LBB9_19
	ldr s0, [x29, #28]
	lsl x8, x1, #2
	and x9, x1, #0x1ffffffffffffff0
	add x19, x0, x9, lsl #2
	fcmp s0, #0.0
	and x10, x8, #0x7fffffffffffffc0
	and x8, x1, #0xf
	add x9, x0, x10
	b.ne .LBB9_10
	cbz x10, .LBB9_8
	fmov v0.4s, #1.00000000
.LBB9_7:
	ldp q1, q2, [x0]
	ldp q5, q6, [x0, #32]
	fcmgt v3.4s, v1.4s, #0.0
	fcmeq v4.4s, v1.4s, v1.4s
	fcmgt v7.4s, v2.4s, #0.0
	fcmeq v16.4s, v2.4s, v2.4s
	fcmgt v17.4s, v5.4s, #0.0
	fcmeq v18.4s, v5.4s, v5.4s
	fcmgt v19.4s, v6.4s, #0.0
	fcmeq v20.4s, v6.4s, v6.4s
	bic v1.16b, v1.16b, v4.16b
	and v3.16b, v3.16b, v0.16b
	and v4.16b, v7.16b, v0.16b
	bic v2.16b, v2.16b, v16.16b
	bic v5.16b, v5.16b, v18.16b
	and v7.16b, v17.16b, v0.16b
	bic v6.16b, v6.16b, v20.16b
	and v16.16b, v19.16b, v0.16b
	orr v1.16b, v3.16b, v1.16b
	orr v2.16b, v4.16b, v2.16b
	orr v3.16b, v7.16b, v5.16b
	orr v4.16b, v16.16b, v6.16b
	stp q1, q2, [x0]
	stp q3, q4, [x0, #32]
	add x0, x0, #64
	cmp x0, x9
	b.ne .LBB9_7
.LBB9_8:
	cbz x8, .LBB9_16
	lsl x20, x8, #2
	mov w8, #64
	mov w1, wzr
	sub x2, x8, x20
	sub x8, x29, #192
	add x0, x8, x20
	bl memset
	sub x0, x29, #192
	mov x1, x19
	mov x2, x20
	bl memcpy
	ldp q0, q1, [x29, #-192]
	fmov v2.4s, #1.00000000
	ldp q5, q6, [x29, #-160]
	fcmgt v3.4s, v0.4s, #0.0
	fcmeq v4.4s, v0.4s, v0.4s
	fcmgt v7.4s, v1.4s, #0.0
	fcmeq v16.4s, v1.4s, v1.4s
	fcmgt v17.4s, v5.4s, #0.0
	fcmeq v18.4s, v5.4s, v5.4s
	fcmgt v19.4s, v6.4s, #0.0
	fcmeq v20.4s, v6.4s, v6.4s
	bic v0.16b, v0.16b, v4.16b
	and v3.16b, v3.16b, v2.16b
	and v4.16b, v7.16b, v2.16b
	bic v1.16b, v1.16b, v16.16b
	bic v5.16b, v5.16b, v18.16b
	and v7.16b, v17.16b, v2.16b
	bic v6.16b, v6.16b, v20.16b
	and v2.16b, v19.16b, v2.16b
	orr v0.16b, v3.16b, v0.16b
	orr v1.16b, v4.16b, v1.16b
	orr v3.16b, v7.16b, v5.16b
	orr v2.16b, v2.16b, v6.16b
	stp q0, q1, [x29, #-128]
	stur q3, [x29, #-96]
	b .LBB9_15
.LBB9_10:
	cbz x10, .LBB9_13
	fmov s0, #4.00000000
	ldr s1, [x29, #28]
	mov w10, #64269
	mov w11, #64269
	movk w10, #74, lsl #16
	fmov v6.4s, #1.00000000
	movk w11, #16330, lsl #16
	dup v3.4s, w10
	mov w10, #1267
	fcmp s1, s0
	dup v2.4s, w11
	mov w11, #27455
	movk w10, #16181, lsl #16
	movk w11, #16094, lsl #16
	stp q2, q3, [sp, #192]
	fcsel s0, s0, s1, gt
	dup v2.4s, w10
	dup v1.4s, w11
	mov w10, #46233
	mov w11, #8388607
	movk w10, #16147, lsl #16
	fmov s3, w11
	stp q1, q2, [sp, #160]
	dup v2.4s, w10
	mov w10, #14483
	movk w10, #16246, lsl #16
	dup v1.4s, w10
	fmov w10, s0
	sub w10, w10, #1
	stp q1, q2, [sp, #128]
	movi d1, #0000000000000000
	cmp w10, w11
	fmov s2, w10
	mov w10, #43579
	movk w10, #16440, lsl #16
	mov w11, #-1023672320
	dup v4.4s, w10
	mov w10, #1123942400
	dup v21.4s, w11
	cmhi v2.4s, v3.4s, v2.4s
	dup v3.4s, w10
	mov w11, #63802
	mov w10, #50297
	movk w11, #14625, lsl #16
	fcsel s29, s1, s0, lo
	movk w10, #15022, lsl #16
	dup v5.4s, w11
	mov w11, #38156
	stp q3, q4, [sp, #96]
	dup v4.4s, w10
	mvni v3.4s, #127, msl #16
	mov w10, #22716
	and v2.16b, v2.16b, v6.16b
	movk w11, #15389, lsl #16
	movk w10, #15715, lsl #16
	stp q4, q5, [sp, #64]
	dup v5.4s, w11
	mov w11, #65005
	dup v4.4s, w10
	mov w10, #29208
	movk w11, #15989, lsl #16
	movk w10, #16177, lsl #16
	fneg v3.4s, v3.4s
	dup v2.4s, v2.s[0]
	dup v30.4s, w11
	dup v31.4s, w10
	stp q4, q5, [sp, #32]
	stp q2, q3, [sp]
.LBB9_12:
	movi v7.4s, #128, lsl #16
	ldp q8, q9, [x0]
	ldp q10, q12, [x0, #32]
	movi v22.4s, #127, msl #16
	fmov v23.4s, #-1.00000000
	mvni v14.4s, #126
	fcmge v0.4s, v8.4s, v7.4s
	fcmge v1.4s, v9.4s, v7.4s
	fcmge v3.4s, v10.4s, v7.4s
	and v0.16b, v0.16b, v8.16b
	and v1.16b, v1.16b, v9.16b
	and v4.16b, v3.16b, v10.16b
	fminnm v0.4s, v0.4s, v6.4s
	fminnm v1.4s, v1.4s, v6.4s
	fminnm v4.4s, v4.4s, v6.4s
	cmeq v2.4s, v0.4s, #0
	cmeq v5.4s, v1.4s, #0
	stur q2, [x29, #-208]
	mov v3.16b, v5.16b
	ldp q19, q18, [sp, #192]
	stur q5, [x29, #-224]
	ldr q20, [sp, #176]
	ldr q25, [sp, #128]
	add v0.4s, v0.4s, v18.4s
	add v1.4s, v1.4s, v18.4s
	mov v26.16b, v25.16b
	bit v0.16b, v19.16b, v2.16b
	bsl v3.16b, v19.16b, v1.16b
	and v2.16b, v0.16b, v22.16b
	and v5.16b, v3.16b, v22.16b
	add v2.4s, v2.4s, v20.4s
	add v5.4s, v5.4s, v20.4s
	fadd v1.4s, v2.4s, v23.4s
	fadd v2.4s, v2.4s, v6.4s
	fdiv v1.4s, v1.4s, v2.4s
	fcmge v2.4s, v12.4s, v7.4s
	cmeq v7.4s, v4.4s, #0
	add v4.4s, v4.4s, v18.4s
	str q7, [sp, #224]
	and v2.16b, v2.16b, v12.16b
	bit v4.16b, v19.16b, v7.16b
	fadd v7.4s, v5.4s, v23.4s
	fadd v5.4s, v5.4s, v6.4s
	fminnm v16.4s, v2.4s, v6.4s
	and v17.16b, v4.16b, v22.16b
	ssra v14.4s, v4.4s, #23
	fdiv v2.4s, v7.4s, v5.4s
	cmeq v15.4s, v16.4s, #0
	add v5.4s, v16.4s, v18.4s
	add v7.4s, v17.4s, v20.4s
	bit v5.16b, v19.16b, v15.16b
	fadd v16.4s, v7.4s, v23.4s
	fadd v7.4s, v7.4s, v6.4s
	ldp q19, q27, [sp, #144]
	and v17.16b, v5.16b, v22.16b
	mov v13.16b, v19.16b
	mov v28.16b, v19.16b
	mvni v22.4s, #126
	fdiv v7.4s, v16.4s, v7.4s
	add v16.4s, v17.4s, v20.4s
	mvni v20.4s, #126
	ssra v22.4s, v3.4s, #23
	mov v3.16b, v25.16b
	fmul v18.4s, v2.4s, v2.4s
	fadd v17.4s, v16.4s, v23.4s
	fadd v16.4s, v16.4s, v6.4s
	ssra v20.4s, v0.4s, #23
	mov v0.16b, v25.16b
	mov v23.16b, v19.16b
	scvtf v22.4s, v22.4s
	fmla v13.4s, v27.4s, v18.4s
	scvtf v20.4s, v20.4s
	fdiv v16.4s, v17.4s, v16.4s
	fmul v17.4s, v1.4s, v1.4s
	fmla v0.4s, v18.4s, v13.4s
	mov v13.16b, v19.16b
	ldr q19, [sp, #112]
	fmul v11.4s, v7.4s, v7.4s
	mov v4.16b, v19.16b
	fmla v28.4s, v27.4s, v17.4s
	fmla v23.4s, v27.4s, v11.4s
	fmla v26.4s, v17.4s, v28.4s
	mvni v28.4s, #126
	fmla v3.4s, v11.4s, v23.4s
	mov v23.16b, v19.16b
	ssra v28.4s, v5.4s, #23
	fmla v4.4s, v17.4s, v26.4s
	scvtf v5.4s, v14.4s
	movi v26.4s, #67, lsl #24
	fmul v24.4s, v16.4s, v16.4s
	fmla v23.4s, v18.4s, v0.4s
	mov v0.16b, v19.16b
	fmla v20.4s, v1.4s, v4.4s
	mov v1.16b, v19.16b
	fmla v13.4s, v27.4s, v24.4s
	fmla v0.4s, v11.4s, v3.4s
	fmla v22.4s, v2.4s, v23.4s
	scvtf v2.4s, v28.4s
	ldp q27, q23, [sp, #80]
	ldp q14, q28, [sp, #48]
	fmla v25.4s, v24.4s, v13.4s
	fmla v5.4s, v7.4s, v0.4s
	fmul v0.4s, v20.4s, v29.s[0]
	fmul v17.4s, v22.4s, v29.s[0]
	fmla v1.4s, v24.4s, v25.4s
	fmul v18.4s, v5.4s, v29.s[0]
	mov v24.16b, v28.16b
	fmaxnm v3.4s, v17.4s, v21.4s
	mov v25.16b, v14.16b
	fmla v2.4s, v16.4s, v1.4s
	fmaxnm v1.4s, v0.4s, v21.4s
	fminnm v7.4s, v3.4s, v26.4s
	fmul v16.4s, v2.4s, v29.s[0]
	fmaxnm v2.4s, v18.4s, v21.4s
	fminnm v1.4s, v1.4s, v26.4s
	frintn v4.4s, v7.4s
	fmaxnm v3.4s, v16.4s, v21.4s
	fminnm v20.4s, v2.4s, v26.4s
	frintn v2.4s, v1.4s
	fminnm v4.4s, v4.4s, v23.4s
	fminnm v22.4s, v3.4s, v26.4s
	fminnm v5.4s, v2.4s, v23.4s
	frintn v3.4s, v20.4s
	fsub v7.4s, v7.4s, v4.4s
	fcvtns v4.4s, v4.4s
	frintn v2.4s, v22.4s
	fsub v19.4s, v1.4s, v5.4s
	fminnm v3.4s, v3.4s, v23.4s
	fcvtns v5.4s, v5.4s
	shl v4.4s, v4.4s, #23
	fminnm v2.4s, v2.4s, v23.4s
	mov v23.16b, v28.16b
	fsub v11.4s, v20.4s, v3.4s
	mov v20.16b, v28.16b
	fcvtns v3.4s, v3.4s
	shl v5.4s, v5.4s, #23
	add v4.4s, v4.4s, v6.4s
	fmla v23.4s, v27.4s, v19.4s
	fsub v1.4s, v22.4s, v2.4s
	mov v22.16b, v14.16b
	fmla v20.4s, v27.4s, v7.4s
	fmla v24.4s, v27.4s, v11.4s
	fcvtns v2.4s, v2.4s
	shl v3.4s, v3.4s, #23
	add v5.4s, v5.4s, v6.4s
	fmla v22.4s, v19.4s, v23.4s
	mov v23.16b, v28.16b
	fmla v25.4s, v7.4s, v20.4s
	mov v20.16b, v14.16b
	add v3.4s, v3.4s, v6.4s
	shl v2.4s, v2.4s, #23
	fmla v23.4s, v27.4s, v1.4s
	ldr q27, [sp, #32]
	fmla v20.4s, v11.4s, v24.4s
	mov v13.16b, v27.16b
	mov v24.16b, v27.16b
	add v2.4s, v2.4s, v6.4s
	fmla v13.4s, v19.4s, v22.4s
	mov v22.16b, v14.16b
	fmla v24.4s, v7.4s, v25.4s
	mov v25.16b, v27.16b
	fmov v14.4s, #1.00000000
	fmla v22.4s, v1.4s, v23.4s
	mov v23.16b, v30.16b
	fmla v25.4s, v11.4s, v20.4s
	mov v20.16b, v30.16b
	fmla v23.4s, v19.4s, v13.4s
	mov v13.16b, v27.16b
	fmla v20.4s, v7.4s, v24.4s
	mov v24.16b, v30.16b
	fmla v13.4s, v1.4s, v22.4s
	mov v22.16b, v31.16b
	fmla v24.4s, v11.4s, v25.4s
	mov v25.16b, v31.16b
	fmla v22.4s, v19.4s, v23.4s
	mov v23.16b, v30.16b
	fmla v25.4s, v7.4s, v20.4s
	fmov v20.4s, #1.00000000
	fmla v23.4s, v1.4s, v13.4s
	mov v13.16b, v31.16b
	fmla v14.4s, v19.4s, v22.4s
	mov v19.16b, v31.16b
	fcmge v22.4s, v0.4s, v26.4s
	fmla v20.4s, v7.4s, v25.4s
	fmov v7.4s, #1.00000000
	fcmge v25.4s, v18.4s, v26.4s
	fmla v13.4s, v11.4s, v24.4s
	fmov v24.4s, #1.00000000
	fcmgt v0.4s, v21.4s, v0.4s
	fmla v19.4s, v1.4s, v23.4s
	fcmge v23.4s, v17.4s, v26.4s
	fcmgt v17.4s, v21.4s, v17.4s
	fcmgt v18.4s, v21.4s, v18.4s
	fmul v5.4s, v14.4s, v5.4s
	fmla v24.4s, v11.4s, v13.4s
	fcmge v11.4s, v16.4s, v26.4s
	fcmgt v16.4s, v21.4s, v16.4s
	fmla v7.4s, v1.4s, v19.4s
	mvn v1.16b, v22.16b
	mvn v19.16b, v23.16b
	mvn v13.16b, v25.16b
	bic v0.16b, v1.16b, v0.16b
	fmul v1.4s, v20.4s, v4.4s
	mvn v4.16b, v11.16b
	fmul v3.4s, v24.4s, v3.4s
	fmul v2.4s, v7.4s, v2.4s
	bic v17.16b, v19.16b, v17.16b
	bic v18.16b, v13.16b, v18.16b
	ldr q19, [sp, #16]
	bic v4.16b, v4.16b, v16.16b
	and v0.16b, v0.16b, v5.16b
	and v5.16b, v22.16b, v19.16b
	and v1.16b, v17.16b, v1.16b
	and v7.16b, v23.16b, v19.16b
	and v3.16b, v18.16b, v3.16b
	and v16.16b, v25.16b, v19.16b
	and v2.16b, v4.16b, v2.16b
	and v4.16b, v11.16b, v19.16b
	orr v0.16b, v0.16b, v5.16b
	orr v1.16b, v1.16b, v7.16b
	ldr q7, [sp, #224]
	orr v3.16b, v3.16b, v16.16b
	ldr q16, [sp]
	orr v2.16b, v2.16b, v4.16b
	ldp q5, q4, [x29, #-224]
	bit v3.16b, v16.16b, v7.16b
	fcmeq v7.4s, v10.4s, v10.4s
	bit v0.16b, v16.16b, v4.16b
	fcmeq v4.4s, v8.4s, v8.4s
	bit v1.16b, v16.16b, v5.16b
	fcmeq v5.4s, v9.4s, v9.4s
	bit v2.16b, v16.16b, v15.16b
	fcmeq v16.4s, v12.4s, v12.4s
	bif v3.16b, v10.16b, v7.16b
	bif v0.16b, v8.16b, v4.16b
	bif v1.16b, v9.16b, v5.16b
	bif v2.16b, v12.16b, v16.16b
	stp q0, q1, [x0]
	stp q3, q2, [x0, #32]
	add x0, x0, #64
	cmp x0, x9
	b.ne .LBB9_12
.LBB9_13:
	cbz x8, .LBB9_16
	lsl x20, x8, #2
	mov w8, #64
	mov w1, wzr
	ldr s8, [x29, #28]
	sub x2, x8, x20
	sub x8, x29, #192
	add x0, x8, x20
	bl memset
	sub x0, x29, #192
	mov x1, x19
	mov x2, x20
	bl memcpy
	movi v7.4s, #128, lsl #16
	ldp q6, q1, [x29, #-192]
	fmov v3.4s, #1.00000000
	mov w8, #64269
	movi v25.4s, #127, msl #16
	movk w8, #74, lsl #16
	fmov v27.4s, #-1.00000000
	mov w9, #27455
	dup v20.4s, w8
	mov w8, #64269
	movk w9, #16094, lsl #16
	fcmge v2.4s, v6.4s, v7.4s
	fcmge v5.4s, v1.4s, v7.4s
	movk w8, #16330, lsl #16
	dup v21.4s, w8
	mov w8, #1267
	fmov s31, #4.00000000
	movk w8, #16181, lsl #16
	mvni v9.4s, #126
	movi d12, #0000000000000000
	dup v26.4s, w8
	mov w8, #46233
	mov w10, #-1023672320
	and v2.16b, v2.16b, v6.16b
	and v5.16b, v5.16b, v1.16b
	movk w8, #16147, lsl #16
	fcmp s8, s31
	fminnm v4.4s, v2.4s, v3.4s
	fminnm v16.4s, v5.4s, v3.4s
	fcsel s31, s31, s8, gt
	cmeq v0.4s, v4.4s, #0
	add v4.4s, v4.4s, v20.4s
	cmeq v2.4s, v16.4s, #0
	add v16.4s, v16.4s, v20.4s
	mov v17.16b, v0.16b
	mov v23.16b, v2.16b
	stur q0, [x29, #-208]
	bsl v17.16b, v21.16b, v4.16b
	ldp q4, q5, [x29, #-160]
	bsl v23.16b, v21.16b, v16.16b
	fcmge v19.4s, v4.4s, v7.4s
	and v18.16b, v17.16b, v25.16b
	ssra v9.4s, v17.4s, #23
	and v22.16b, v23.16b, v25.16b
	add v18.4s, v18.4s, v26.4s
	and v19.16b, v19.16b, v4.16b
	add v22.4s, v22.4s, v26.4s
	fadd v16.4s, v18.4s, v27.4s
	fadd v18.4s, v18.4s, v3.4s
	fminnm v19.4s, v19.4s, v3.4s
	fadd v24.4s, v22.4s, v27.4s
	fadd v22.4s, v22.4s, v3.4s
	fdiv v18.4s, v16.4s, v18.4s
	fcmge v16.4s, v5.4s, v7.4s
	cmeq v7.4s, v19.4s, #0
	add v19.4s, v19.4s, v20.4s
	and v16.16b, v16.16b, v5.16b
	bit v19.16b, v21.16b, v7.16b
	fminnm v28.4s, v16.4s, v3.4s
	and v29.16b, v19.16b, v25.16b
	fdiv v22.4s, v24.4s, v22.4s
	cmeq v16.4s, v28.4s, #0
	add v20.4s, v28.4s, v20.4s
	add v28.4s, v29.4s, v26.4s
	mov v24.16b, v16.16b
	bsl v24.16b, v21.16b, v20.16b
	fadd v20.4s, v28.4s, v27.4s
	fadd v21.4s, v28.4s, v3.4s
	dup v28.4s, w9
	mov w9, #43579
	movk w9, #16440, lsl #16
	dup v13.4s, w9
	fmov w9, s31
	and v25.16b, v24.16b, v25.16b
	fdiv v20.4s, v20.4s, v21.4s
	fmul v30.4s, v22.4s, v22.4s
	sub w9, w9, #1
	add v21.4s, v25.4s, v26.4s
	dup v26.4s, w8
	mov w8, #14483
	movk w8, #16246, lsl #16
	fadd v25.4s, v21.4s, v27.4s
	fadd v21.4s, v21.4s, v3.4s
	fmul v27.4s, v18.4s, v18.4s
	mov v29.16b, v26.16b
	mov v10.16b, v26.16b
	mov v15.16b, v26.16b
	fdiv v21.4s, v25.4s, v21.4s
	dup v25.4s, w8
	fmla v29.4s, v28.4s, v27.4s
	fmla v10.4s, v28.4s, v30.4s
	mov w8, #8388607
	cmp w9, w8
	mov v11.16b, v25.16b
	mov v17.16b, v25.16b
	fmul v14.4s, v20.4s, v20.4s
	fmla v11.4s, v27.4s, v29.4s
	mvni v29.4s, #126
	fmla v17.4s, v30.4s, v10.4s
	mov v10.16b, v13.16b
	fmla v15.4s, v28.4s, v14.4s
	ssra v29.4s, v23.4s, #23
	scvtf v23.4s, v9.4s
	mvni v9.4s, #126
	fmla v10.4s, v27.4s, v11.4s
	mov v11.16b, v13.16b
	mvni v27.4s, #126
	ssra v9.4s, v19.4s, #23
	fmul v8.4s, v21.4s, v21.4s
	fmla v11.4s, v30.4s, v17.4s
	fcsel s30, s12, s31, lo
	fmla v23.4s, v18.4s, v10.4s
	ssra v27.4s, v24.4s, #23
	mov v24.16b, v13.16b
	dup v17.4s, w10
	mov w10, #1123942400
	fmla v26.4s, v28.4s, v8.4s
	scvtf v28.4s, v29.4s
	mov v29.16b, v25.16b
	fmul v18.4s, v23.4s, v30.s[0]
	scvtf v23.4s, v9.4s
	fmla v29.4s, v14.4s, v15.4s
	fmla v25.4s, v8.4s, v26.4s
	fmla v28.4s, v22.4s, v11.4s
	scvtf v26.4s, v27.4s
	movi v22.4s, #67, lsl #24
	dup v27.4s, w10
	mov w10, #50297
	movk w10, #15022, lsl #16
	fmla v24.4s, v14.4s, v29.4s
	dup v9.4s, w10
	mov w10, #63802
	fmla v13.4s, v8.4s, v25.4s
	fmul v19.4s, v28.4s, v30.s[0]
	fmaxnm v25.4s, v18.4s, v17.4s
	movk w10, #14625, lsl #16
	dup v11.4s, w10
	mov w10, #38156
	mov v14.16b, v9.16b
	fmla v23.4s, v20.4s, v24.4s
	movk w10, #15389, lsl #16
	fmla v26.4s, v21.4s, v13.4s
	fmaxnm v24.4s, v19.4s, v17.4s
	fminnm v25.4s, v25.4s, v22.4s
	mov v13.16b, v9.16b
	dup v10.4s, w10
	mov w10, #22716
	movk w10, #15715, lsl #16
	fmul v20.4s, v23.4s, v30.s[0]
	fmul v21.4s, v26.4s, v30.s[0]
	fminnm v24.4s, v24.4s, v22.4s
	frintn v23.4s, v25.4s
	mov v15.16b, v10.16b
	fmaxnm v26.4s, v20.4s, v17.4s
	fmaxnm v28.4s, v21.4s, v17.4s
	frintn v29.4s, v24.4s
	fminnm v23.4s, v23.4s, v27.4s
	fminnm v31.4s, v26.4s, v22.4s
	fminnm v8.4s, v28.4s, v22.4s
	fminnm v26.4s, v29.4s, v27.4s
	fsub v29.4s, v25.4s, v23.4s
	fcvtns v23.4s, v23.4s
	frintn v25.4s, v31.4s
	frintn v12.4s, v8.4s
	fsub v30.4s, v24.4s, v26.4s
	fmla v13.4s, v11.4s, v29.4s
	fcvtns v26.4s, v26.4s
	shl v23.4s, v23.4s, #23
	fminnm v28.4s, v25.4s, v27.4s
	fminnm v27.4s, v12.4s, v27.4s
	fmla v14.4s, v11.4s, v30.4s
	dup v12.4s, w10
	fmla v15.4s, v29.4s, v13.4s
	mov w10, #65005
	shl v26.4s, v26.4s, #23
	movk w10, #15989, lsl #16
	add v23.4s, v23.4s, v3.4s
	fsub v25.4s, v31.4s, v28.4s
	mov v31.16b, v10.16b
	mov v13.16b, v12.16b
	fsub v24.4s, v8.4s, v27.4s
	dup v8.4s, w10
	mov w10, #29208
	movk w10, #16177, lsl #16
	fcvtns v28.4s, v28.4s
	fcvtns v27.4s, v27.4s
	fmla v31.4s, v30.4s, v14.4s
	mov v14.16b, v9.16b
	fmla v13.4s, v29.4s, v15.4s
	mov v15.16b, v12.16b
	dup v0.4s, w10
	add v26.4s, v26.4s, v3.4s
	fmla v9.4s, v11.4s, v24.4s
	fmla v14.4s, v11.4s, v25.4s
	mov v11.16b, v8.16b
	shl v27.4s, v27.4s, #23
	fmla v15.4s, v30.4s, v31.4s
	mov v31.16b, v0.16b
	fmla v11.4s, v29.4s, v13.4s
	mov v13.16b, v10.16b
	fmla v10.4s, v24.4s, v9.4s
	fmov v9.4s, #1.00000000
	fmla v13.4s, v25.4s, v14.4s
	mov v14.16b, v8.16b
	fmla v31.4s, v29.4s, v11.4s
	fmov v11.4s, #1.00000000
	fmla v14.4s, v30.4s, v15.4s
	mov v15.16b, v12.16b
	fmla v12.4s, v24.4s, v10.4s
	fmov v10.4s, #1.00000000
	fmla v9.4s, v29.4s, v31.4s
	fmov v29.4s, #1.00000000
	mvni v31.4s, #127, msl #16
	fmla v15.4s, v25.4s, v13.4s
	mov v13.16b, v0.16b
	fmla v13.4s, v30.4s, v14.4s
	mov v14.16b, v8.16b
	fmla v8.4s, v24.4s, v12.4s
	fmov s12, w9
	fmul v23.4s, v9.4s, v23.4s
	fmla v14.4s, v25.4s, v15.4s
	fmla v10.4s, v30.4s, v13.4s
	mov v13.16b, v0.16b
	fmov s30, w8
	fmla v0.4s, v24.4s, v8.4s
	fcmge v8.4s, v20.4s, v22.4s
	fcmgt v20.4s, v17.4s, v20.4s
	fmla v13.4s, v25.4s, v14.4s
	cmhi v30.4s, v30.4s, v12.4s
	fcmge v12.4s, v18.4s, v22.4s
	fcmge v14.4s, v19.4s, v22.4s
	fcmge v22.4s, v21.4s, v22.4s
	fcmgt v18.4s, v17.4s, v18.4s
	fcmgt v19.4s, v17.4s, v19.4s
	fcmgt v17.4s, v17.4s, v21.4s
	shl v21.4s, v28.4s, #23
	fmla v29.4s, v24.4s, v0.4s
	fneg v28.4s, v31.4s
	mvn v0.16b, v8.16b
	fmla v11.4s, v25.4s, v13.4s
	mvn v25.16b, v12.16b
	add v24.4s, v27.4s, v3.4s
	add v21.4s, v21.4s, v3.4s
	mvn v31.16b, v14.16b
	mvn v27.16b, v22.16b
	bic v0.16b, v0.16b, v20.16b
	and v3.16b, v30.16b, v3.16b
	bic v18.16b, v25.16b, v18.16b
	fmul v25.4s, v10.4s, v26.4s
	fmul v20.4s, v29.4s, v24.4s
	fmul v21.4s, v11.4s, v21.4s
	bic v19.16b, v31.16b, v19.16b
	bic v17.16b, v27.16b, v17.16b
	and v24.16b, v12.16b, v28.16b
	dup v3.4s, v3.s[0]
	and v18.16b, v18.16b, v23.16b
	and v23.16b, v14.16b, v28.16b
	and v19.16b, v19.16b, v25.16b
	and v25.16b, v8.16b, v28.16b
	and v17.16b, v17.16b, v20.16b
	and v0.16b, v0.16b, v21.16b
	and v21.16b, v22.16b, v28.16b
	fcmeq v20.4s, v6.4s, v6.4s
	orr v18.16b, v18.16b, v24.16b
	orr v19.16b, v19.16b, v23.16b
	orr v0.16b, v0.16b, v25.16b
	orr v17.16b, v17.16b, v21.16b
	ldur q21, [x29, #-208]
	bit v18.16b, v3.16b, v21.16b
	fcmeq v21.4s, v1.4s, v1.4s
	bsl v2.16b, v3.16b, v19.16b
	fcmeq v19.4s, v4.4s, v4.4s
	bit v0.16b, v3.16b, v7.16b
	fcmeq v7.4s, v5.4s, v5.4s
	bif v3.16b, v17.16b, v16.16b
	bit v6.16b, v18.16b, v20.16b
	bit v1.16b, v2.16b, v21.16b
	bif v0.16b, v4.16b, v19.16b
	mov v2.16b, v7.16b
	bsl v2.16b, v3.16b, v5.16b
	stp q6, q1, [x29, #-128]
	stur q0, [x29, #-96]
.LBB9_15:
	sub x1, x29, #128
	mov x0, x19
	mov x2, x20
	stur q2, [x29, #-80]
	bl memcpy
.LBB9_16:
	add sp, sp, #400
	.cfi_def_cfa wsp, 112
	ldp x20, x19, [sp, #96]
	ldr x28, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #112
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
.LBB9_17:
	.cfi_restore_state
	.cfi_remember_state
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB9_2
.LBB9_18:
	ldr s0, [x29, #28]
	add sp, sp, #400
	.cfi_def_cfa wsp, 112
	ldp x20, x19, [sp, #96]
	ldr x28, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #112
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
	b linear_srgb::simd::gamma_to_linear_slice_tier_scalar_t
.LBB9_19:
	.cfi_restore_state
	ldr s0, [x29, #28]
	add sp, sp, #400
	.cfi_def_cfa wsp, 112
	ldp x20, x19, [sp, #96]
	ldr x28, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #112
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
	b linear_srgb::tf::math::gamma_outside_fit_slice
