.section .text.stub_linear_to_gamma_slice,"ax",@progbits
	.globl	stub_linear_to_gamma_slice
	.p2align	2
.type	stub_linear_to_gamma_slice,@function
stub_linear_to_gamma_slice:
	.cfi_startproc
	stp d15, d14, [sp, #-160]!
	.cfi_def_cfa_offset 160
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	stp x29, x30, [sp, #64]
	str x28, [sp, #80]
	stp x26, x25, [sp, #96]
	stp x24, x23, [sp, #112]
	stp x22, x21, [sp, #128]
	stp x20, x19, [sp, #144]
	add x29, sp, #64
	.cfi_def_cfa w29, 96
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -56
	.cfi_offset w26, -64
	.cfi_offset w28, -80
	.cfi_offset w30, -88
	.cfi_offset w29, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	.cfi_offset b10, -120
	.cfi_offset b11, -128
	.cfi_offset b12, -136
	.cfi_offset b13, -144
	.cfi_offset b14, -152
	.cfi_offset b15, -160
	.cfi_remember_state
	sub sp, sp, #576
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB7_35
	cmp w8, #2
	b.ne .LBB7_34
.LBB7_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x21, x1, #0xf
	lsr x22, x1, #4
	add x19, x0, x8, lsl #2
	lsl x20, x21, #2
	b.eq .LBB7_4
	mov w8, #64
	mov x23, x0
	mov w1, wzr
	sub x2, x8, x20
	sub x8, x29, #256
	fmov s8, s0
	add x0, x8, x20
	bl memset
	sub x0, x29, #256
	mov x1, x19
	mov x2, x20
	bl memcpy
	ldp q1, q2, [x29, #-256]
	add x8, sp, #65
	fmov s0, s8
	mov x0, x23
	add x9, sp, #113
	stur q1, [x8, #255]
	ldp q1, q3, [x29, #-224]
	add x8, sp, #81
	stur q2, [x8, #255]
	add x8, sp, #97
	stur q1, [x8, #255]
	mov w8, #1
	stur q3, [x9, #255]
	b .LBB7_5
.LBB7_4:
	mov w8, wzr
.LBB7_5:
	fmov s1, #1.00000000
	add x9, sp, #316
	cmp x21, #0
	add x1, x9, #4
	movi d2, #0000000000000000
	add x22, x0, x22, lsl #6
	csel x21, xzr, x1, eq
	str w8, [sp, #316]
	fdiv s8, s1, s0
	fmov s1, #4.00000000
	fcmp s8, s1
	fcsel s0, s1, s8, gt
	fmov w9, s0
	sub w10, w9, #1
	mov w9, #8388607
	cmp w10, w9
	fcsel s0, s2, s0, lo
	fcmp s8, #0.0
	fccmp s8, s1, #2, ge
	str q0, [sp, #272]
	b.hi .LBB7_36
	mov w8, #39322
	movk w8, #16409, lsl #16
	fmov s1, w8
	fcmp s8, s1
	b.ne .LBB7_15
	mov w9, #64269
	mov w10, #64269
	fmov v12.4s, #1.00000000
	movk w9, #74, lsl #16
	movk w10, #16330, lsl #16
	dup v1.4s, w9
	dup v0.4s, w10
	mov w9, #1267
	movk w9, #16181, lsl #16
	mov w10, #14483
	movk w10, #16246, lsl #16
	stp q0, q1, [sp, #224]
	dup v1.4s, w9
	mov w9, #27455
	movk w9, #16094, lsl #16
	dup v0.4s, w9
	mov w9, #46233
	movk w9, #16147, lsl #16
	stp q0, q1, [sp, #192]
	dup v1.4s, w9
	dup v0.4s, w10
	mov w9, #43579
	mov w10, #29208
	movk w9, #16440, lsl #16
	movk w10, #16177, lsl #16
	stp q0, q1, [sp, #160]
	dup v1.4s, w9
	dup v0.4s, w8
	mov w8, #-1023672320
	mov w9, #38156
	dup v20.4s, w8
	mov w8, #1123942400
	movk w9, #15389, lsl #16
	stp q0, q1, [sp, #128]
	dup v1.4s, w8
	mov w8, #63802
	movk w8, #14625, lsl #16
	mvni v0.4s, #127, msl #16
	dup v2.4s, w8
	mov w8, #50297
	movk w8, #15022, lsl #16
	stp q2, q1, [sp, #96]
	dup v1.4s, w8
	dup v2.4s, w9
	mov w8, #22716
	mov w9, #65005
	movk w8, #15715, lsl #16
	movk w9, #15989, lsl #16
	stp q2, q1, [sp, #64]
	fneg v1.4s, v0.4s
	dup v0.4s, w8
	stp q0, q1, [sp, #32]
	dup v1.4s, w9
	dup v0.4s, w10
	stp q0, q1, [sp]
	b .LBB7_9
.LBB7_8:
	stp q4, q3, [x8]
	stp q2, q0, [x8, #32]
.LBB7_9:
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB7_12
	cbz x21, .LBB7_41
	mov x0, xzr
	mov x8, x21
	mov x21, xzr
	b .LBB7_13
.LBB7_12:
	add x9, x0, #64
	mov x8, x0
	mov x0, x9
.LBB7_13:
	movi v16.4s, #128, lsl #16
	ldp q30, q31, [x8]
	ldp q21, q19, [sp, #224]
	movi v22.4s, #127, msl #16
	ldp q8, q10, [x8, #32]
	fmov v25.4s, #-1.00000000
	ldr q24, [sp, #208]
	ldp q18, q15, [sp, #144]
	fcmge v0.4s, v30.4s, v16.4s
	fcmge v2.4s, v31.4s, v16.4s
	mvni v29.4s, #126
	fcmge v4.4s, v8.4s, v16.4s
	ldr q26, [sp, #64]
	ldr q23, [sp, #32]
	mov v9.16b, v18.16b
	mov v14.16b, v15.16b
	and v0.16b, v0.16b, v30.16b
	and v2.16b, v2.16b, v31.16b
	and v5.16b, v4.16b, v8.16b
	fminnm v0.4s, v0.4s, v12.4s
	fminnm v2.4s, v2.4s, v12.4s
	fminnm v5.4s, v5.4s, v12.4s
	cmeq v3.4s, v0.4s, #0
	add v0.4s, v0.4s, v19.4s
	bit v0.16b, v21.16b, v3.16b
	cmeq v1.4s, v2.4s, #0
	add v2.4s, v2.4s, v19.4s
	mov v4.16b, v1.16b
	stp q1, q3, [sp, #272]
	cmeq v1.4s, v5.4s, #0
	and v3.16b, v0.16b, v22.16b
	bsl v4.16b, v21.16b, v2.16b
	str q1, [sp, #256]
	add v3.4s, v3.4s, v24.4s
	fadd v2.4s, v3.4s, v25.4s
	fadd v3.4s, v3.4s, v12.4s
	and v7.16b, v4.16b, v22.16b
	ssra v29.4s, v4.4s, #23
	mvni v4.4s, #126
	fdiv v6.4s, v2.4s, v3.4s
	fcmge v2.4s, v10.4s, v16.4s
	add v3.4s, v5.4s, v19.4s
	add v5.4s, v7.4s, v24.4s
	mov v16.16b, v1.16b
	and v2.16b, v2.16b, v10.16b
	bsl v16.16b, v21.16b, v3.16b
	fadd v3.4s, v5.4s, v25.4s
	fadd v5.4s, v5.4s, v12.4s
	fminnm v7.4s, v2.4s, v12.4s
	and v17.16b, v16.16b, v22.16b
	fdiv v2.4s, v3.4s, v5.4s
	cmeq v13.4s, v7.4s, #0
	add v3.4s, v7.4s, v19.4s
	add v5.4s, v17.4s, v24.4s
	mov v7.16b, v13.16b
	bsl v7.16b, v21.16b, v3.16b
	fadd v3.4s, v5.4s, v25.4s
	fadd v5.4s, v5.4s, v12.4s
	and v17.16b, v7.16b, v22.16b
	fdiv v5.4s, v3.4s, v5.4s
	mvni v22.4s, #126
	ssra v4.4s, v7.4s, #23
	fmul v21.4s, v2.4s, v2.4s
	add v3.4s, v17.4s, v24.4s
	ldp q11, q24, [sp, #176]
	ssra v22.4s, v0.4s, #23
	fadd v17.4s, v3.4s, v25.4s
	fadd v3.4s, v3.4s, v12.4s
	mov v19.16b, v11.16b
	mov v25.16b, v15.16b
	mov v28.16b, v11.16b
	mov v0.16b, v11.16b
	scvtf v22.4s, v22.4s
	fdiv v3.4s, v17.4s, v3.4s
	fmul v17.4s, v6.4s, v6.4s
	fmla v28.4s, v24.4s, v21.4s
	fmul v27.4s, v5.4s, v5.4s
	fmla v19.4s, v24.4s, v17.4s
	fmla v14.4s, v21.4s, v28.4s
	mov v28.16b, v15.16b
	fmla v0.4s, v24.4s, v27.4s
	fmla v25.4s, v17.4s, v19.4s
	fmla v28.4s, v27.4s, v0.4s
	mov v0.16b, v18.16b
	fmla v9.4s, v17.4s, v25.4s
	mov v25.16b, v11.16b
	mvni v17.4s, #126
	fmul v19.4s, v3.4s, v3.4s
	fmla v0.4s, v21.4s, v14.4s
	mov v21.16b, v18.16b
	ssra v17.4s, v16.4s, #23
	scvtf v16.4s, v29.4s
	fmla v22.4s, v6.4s, v9.4s
	mov v6.16b, v15.16b
	fmla v25.4s, v24.4s, v19.4s
	ldp q24, q1, [sp, #112]
	scvtf v7.4s, v17.4s
	fmla v21.4s, v27.4s, v28.4s
	mov v27.16b, v26.16b
	fmla v16.4s, v2.4s, v0.4s
	scvtf v2.4s, v4.4s
	fmul v14.4s, v22.4s, v1.4s
	ldp q29, q28, [sp, #80]
	fmla v6.4s, v19.4s, v25.4s
	mov v25.16b, v23.16b
	fmla v7.4s, v5.4s, v21.4s
	fmul v0.4s, v16.4s, v1.4s
	mov v22.16b, v29.16b
	fmaxnm v4.4s, v14.4s, v20.4s
	fmla v18.4s, v19.4s, v6.4s
	fmul v15.4s, v7.4s, v1.4s
	fmaxnm v5.4s, v0.4s, v20.4s
	fmla v2.4s, v3.4s, v18.4s
	movi v18.4s, #67, lsl #24
	fminnm v3.4s, v4.4s, v18.4s
	fmul v4.4s, v2.4s, v1.4s
	fmaxnm v2.4s, v15.4s, v20.4s
	fminnm v5.4s, v5.4s, v18.4s
	frintn v6.4s, v3.4s
	fmaxnm v7.4s, v4.4s, v20.4s
	fminnm v17.4s, v2.4s, v18.4s
	frintn v2.4s, v5.4s
	fminnm v6.4s, v6.4s, v24.4s
	fminnm v7.4s, v7.4s, v18.4s
	frintn v19.4s, v17.4s
	fminnm v2.4s, v2.4s, v24.4s
	fsub v16.4s, v3.4s, v6.4s
	frintn v21.4s, v7.4s
	fminnm v3.4s, v19.4s, v24.4s
	fsub v1.4s, v5.4s, v2.4s
	fcvtns v6.4s, v6.4s
	fcvtns v2.4s, v2.4s
	fmla v22.4s, v28.4s, v16.4s
	fminnm v9.4s, v21.4s, v24.4s
	mov v21.16b, v26.16b
	fsub v19.4s, v17.4s, v3.4s
	mov v17.16b, v29.16b
	mov v24.16b, v23.16b
	fcvtns v3.4s, v3.4s
	shl v6.4s, v6.4s, #23
	shl v2.4s, v2.4s, #23
	fmla v21.4s, v16.4s, v22.4s
	fsub v7.4s, v7.4s, v9.4s
	mov v22.16b, v29.16b
	fmla v17.4s, v28.4s, v1.4s
	fcvtns v9.4s, v9.4s
	add v6.4s, v6.4s, v12.4s
	shl v3.4s, v3.4s, #23
	add v2.4s, v2.4s, v12.4s
	fmla v22.4s, v28.4s, v19.4s
	fmla v25.4s, v16.4s, v21.4s
	mov v21.16b, v29.16b
	fmla v27.4s, v1.4s, v17.4s
	mov v17.16b, v26.16b
	add v3.4s, v3.4s, v12.4s
	fmla v21.4s, v28.4s, v7.4s
	ldp q5, q28, [sp]
	fmla v17.4s, v19.4s, v22.4s
	fmla v24.4s, v1.4s, v27.4s
	mov v27.16b, v23.16b
	mov v11.16b, v28.16b
	mov v29.16b, v5.16b
	mov v22.16b, v28.16b
	fmla v27.4s, v19.4s, v17.4s
	mov v17.16b, v5.16b
	fmla v11.4s, v16.4s, v25.4s
	mov v25.16b, v26.16b
	mov v26.16b, v28.16b
	fmla v25.4s, v7.4s, v21.4s
	fmov v21.4s, #1.00000000
	fmla v26.4s, v1.4s, v24.4s
	fmla v29.4s, v16.4s, v11.4s
	fmla v22.4s, v19.4s, v27.4s
	fcmgt v27.4s, v8.4s, v10.4s
	fcmge v24.4s, v31.4s, v30.4s
	fcmgt v11.4s, v30.4s, v31.4s
	fmla v23.4s, v7.4s, v25.4s
	fcmge v25.4s, v10.4s, v8.4s
	fmla v21.4s, v16.4s, v29.4s
	mov v29.16b, v28.16b
	mov v16.16b, v5.16b
	fmov v28.4s, #1.00000000
	fmla v17.4s, v19.4s, v22.4s
	fmov v22.4s, #1.00000000
	orr v24.16b, v11.16b, v24.16b
	fcmgt v11.4s, v20.4s, v14.4s
	fmla v29.4s, v7.4s, v23.4s
	fmla v16.4s, v1.4s, v26.4s
	orr v25.16b, v27.16b, v25.16b
	fmov v27.4s, #1.00000000
	fmul v6.4s, v21.4s, v6.4s
	mvn v24.16b, v24.16b
	fmla v5.4s, v7.4s, v29.4s
	fmla v28.4s, v1.4s, v16.4s
	fcmge v16.4s, v14.4s, v18.4s
	fcmge v29.4s, v0.4s, v18.4s
	fmla v27.4s, v19.4s, v17.4s
	fcmge v17.4s, v15.4s, v18.4s
	fcmge v19.4s, v4.4s, v18.4s
	fcmgt v0.4s, v20.4s, v0.4s
	fcmgt v4.4s, v20.4s, v4.4s
	ldr q18, [sp, #48]
	fmla v22.4s, v7.4s, v5.4s
	shl v7.4s, v9.4s, #23
	mvn v5.16b, v16.16b
	fcmgt v9.4s, v20.4s, v15.4s
	mvn v14.16b, v29.16b
	fmul v2.4s, v28.4s, v2.4s
	mvn v21.16b, v19.16b
	fmul v3.4s, v27.4s, v3.4s
	add v7.4s, v7.4s, v12.4s
	bic v5.16b, v5.16b, v11.16b
	mvn v11.16b, v17.16b
	bic v0.16b, v14.16b, v0.16b
	bic v4.16b, v21.16b, v4.16b
	orn v21.16b, v24.16b, v25.16b
	fmul v7.4s, v22.4s, v7.4s
	bic v27.16b, v11.16b, v9.16b
	and v5.16b, v5.16b, v6.16b
	and v0.16b, v0.16b, v2.16b
	and v6.16b, v29.16b, v18.16b
	and v2.16b, v27.16b, v3.16b
	and v3.16b, v16.16b, v18.16b
	and v16.16b, v19.16b, v18.16b
	and v4.16b, v4.16b, v7.16b
	and v7.16b, v17.16b, v18.16b
	orr v0.16b, v0.16b, v6.16b
	addp d6, v21.2d
	orr v3.16b, v5.16b, v3.16b
	orr v2.16b, v2.16b, v7.16b
	ldp q1, q7, [sp, #272]
	orr v5.16b, v4.16b, v16.16b
	fmov x9, d6
	bic v4.16b, v3.16b, v7.16b
	bic v3.16b, v0.16b, v1.16b
	ldr q0, [sp, #256]
	bic v2.16b, v2.16b, v0.16b
	bic v0.16b, v5.16b, v13.16b
	cbz x9, .LBB7_8
	fcmeq v5.4s, v30.4s, v30.4s
	fcmeq v6.4s, v10.4s, v10.4s
	fcmeq v7.4s, v8.4s, v8.4s
	fcmeq v16.4s, v31.4s, v31.4s
	bif v4.16b, v30.16b, v5.16b
	bif v2.16b, v8.16b, v7.16b
	bif v0.16b, v10.16b, v6.16b
	bif v3.16b, v31.16b, v16.16b
	b .LBB7_8
.LBB7_15:
	mov w8, #21845
	movk w8, #16085, lsl #16
	fmov s1, w8
	fcmp s8, s1
	b.ne .LBB7_24
	mov w9, #64269
	mov w10, #1267
	fmov v1.4s, #1.00000000
	movk w9, #74, lsl #16
	movk w10, #16181, lsl #16
	dup v2.4s, w9
	dup v0.4s, w10
	mov w9, #27455
	movk w9, #16094, lsl #16
	mov w10, #43579
	movk w10, #16440, lsl #16
	stp q0, q2, [sp, #272]
	dup v2.4s, w9
	mov w9, #46233
	movk w9, #16147, lsl #16
	dup v0.4s, w9
	mov w9, #14483
	movk w9, #16246, lsl #16
	stp q0, q2, [sp, #240]
	dup v2.4s, w9
	dup v0.4s, w10
	mov w9, #50297
	mov w10, #38156
	ldr q22, [sp, #256]
	movk w9, #15022, lsl #16
	movk w10, #15389, lsl #16
	stp q0, q2, [sp, #208]
	dup v2.4s, w8
	mov w8, #63802
	movk w8, #14625, lsl #16
	dup v0.4s, w8
	mov w8, #22716
	movk w8, #15715, lsl #16
	stp q0, q2, [sp, #176]
	dup v2.4s, w9
	dup v0.4s, w10
	mov w9, #65005
	mov w10, #29208
	movk w9, #15989, lsl #16
	movk w10, #16177, lsl #16
	stp q0, q2, [sp, #144]
	dup v2.4s, w8
	dup v0.4s, w9
	dup v24.4s, w10
	stp q0, q2, [sp, #112]
	b .LBB7_18
.LBB7_17:
	stp q30, q27, [x8]
	stp q2, q0, [x8, #32]
.LBB7_18:
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB7_21
	cbz x21, .LBB7_41
	mov x0, xzr
	mov x8, x21
	mov x21, xzr
	b .LBB7_22
.LBB7_21:
	add x9, x0, #64
	mov x8, x0
	mov x0, x9
.LBB7_22:
	movi v3.4s, #128, lsl #16
	ldp q25, q26, [x8]
	ldp q6, q5, [sp, #272]
	movi v4.4s, #127, msl #16
	fmov v7.4s, #-1.00000000
	ldp q21, q20, [sp, #224]
	mvni v16.4s, #126
	mvni v19.4s, #126
	fcmge v0.4s, v25.4s, v3.4s
	fcmge v2.4s, v26.4s, v3.4s
	mov v18.16b, v20.16b
	mov v23.16b, v21.16b
	mov v17.16b, v20.16b
	and v0.16b, v0.16b, v25.16b
	and v2.16b, v2.16b, v26.16b
	fminnm v27.4s, v0.4s, v1.4s
	fminnm v30.4s, v2.4s, v1.4s
	add v0.4s, v27.4s, v5.4s
	add v11.4s, v30.4s, v5.4s
	and v28.16b, v0.16b, v4.16b
	and v10.16b, v11.16b, v4.16b
	ssra v16.4s, v11.4s, #23
	mov v11.16b, v21.16b
	add v2.4s, v28.4s, v6.4s
	ldp q28, q29, [x8, #32]
	scvtf v16.4s, v16.4s
	fcmge v31.4s, v28.4s, v3.4s
	fadd v8.4s, v2.4s, v7.4s
	fadd v2.4s, v2.4s, v1.4s
	and v31.16b, v31.16b, v28.16b
	fdiv v9.4s, v8.4s, v2.4s
	fcmge v8.4s, v29.4s, v3.4s
	add v2.4s, v10.4s, v6.4s
	fminnm v31.4s, v31.4s, v1.4s
	and v8.16b, v8.16b, v29.16b
	fadd v10.4s, v2.4s, v7.4s
	fadd v2.4s, v2.4s, v1.4s
	add v12.4s, v31.4s, v5.4s
	fminnm v8.4s, v8.4s, v1.4s
	and v13.16b, v12.16b, v4.16b
	fdiv v10.4s, v10.4s, v2.4s
	ssra v19.4s, v12.4s, #23
	add v14.4s, v8.4s, v5.4s
	add v2.4s, v13.4s, v6.4s
	and v15.16b, v14.16b, v4.16b
	mvni v4.4s, #126
	fadd v13.4s, v2.4s, v7.4s
	fadd v2.4s, v2.4s, v1.4s
	ssra v4.4s, v0.4s, #23
	mvni v0.4s, #126
	fdiv v2.4s, v13.4s, v2.4s
	add v13.4s, v15.4s, v6.4s
	ssra v0.4s, v14.4s, #23
	scvtf v4.4s, v4.4s
	fadd v15.4s, v13.4s, v7.4s
	fadd v13.4s, v13.4s, v1.4s
	mov v7.16b, v20.16b
	fmul v5.4s, v10.4s, v10.4s
	scvtf v0.4s, v0.4s
	fmla v18.4s, v22.4s, v5.4s
	fdiv v13.4s, v15.4s, v13.4s
	fmul v15.4s, v9.4s, v9.4s
	fmul v3.4s, v2.4s, v2.4s
	fmla v7.4s, v22.4s, v15.4s
	fmla v17.4s, v22.4s, v3.4s
	fmla v23.4s, v15.4s, v7.4s
	mov v7.16b, v21.16b
	fmla v7.4s, v5.4s, v18.4s
	mov v18.16b, v21.16b
	ldr q21, [sp, #208]
	fmla v11.4s, v3.4s, v17.4s
	fmul v6.4s, v13.4s, v13.4s
	mov v12.16b, v21.16b
	mov v17.16b, v21.16b
	fmla v12.4s, v5.4s, v7.4s
	mov v5.16b, v21.16b
	fmla v17.4s, v15.4s, v23.4s
	fmla v20.4s, v22.4s, v6.4s
	scvtf v7.4s, v19.4s
	fmla v4.4s, v9.4s, v17.4s
	fmla v16.4s, v10.4s, v12.4s
	fmla v18.4s, v6.4s, v20.4s
	mov v20.16b, v21.16b
	mov v21.16b, v24.16b
	fmla v20.4s, v3.4s, v11.4s
	fmla v5.4s, v6.4s, v18.4s
	fmla v7.4s, v2.4s, v20.4s
	mov v20.16b, v24.16b
	fmla v0.4s, v13.4s, v5.4s
	ldr q5, [sp, #192]
	fmul v2.4s, v4.4s, v5.4s
	fmul v3.4s, v16.4s, v5.4s
	ldr q16, [sp, #144]
	fmul v4.4s, v7.4s, v5.4s
	ldp q6, q7, [sp, #160]
	fmul v5.4s, v0.4s, v5.4s
	frintn v14.4s, v2.4s
	frintn v15.4s, v3.4s
	frintn v0.4s, v4.4s
	frintn v13.4s, v5.4s
	fsub v12.4s, v2.4s, v14.4s
	fsub v9.4s, v3.4s, v15.4s
	mov v2.16b, v6.16b
	fsub v10.4s, v4.4s, v0.4s
	mov v3.16b, v6.16b
	mov v4.16b, v6.16b
	fsub v11.4s, v5.4s, v13.4s
	mov v5.16b, v6.16b
	mov v6.16b, v16.16b
	fcvtns v14.4s, v14.4s
	fcvtns v15.4s, v15.4s
	fcvtns v0.4s, v0.4s
	fmla v2.4s, v7.4s, v12.4s
	fmla v3.4s, v7.4s, v9.4s
	fcvtns v13.4s, v13.4s
	fmla v4.4s, v7.4s, v10.4s
	fmla v5.4s, v7.4s, v11.4s
	mov v7.16b, v16.16b
	shl v0.4s, v0.4s, #23
	fmla v6.4s, v12.4s, v2.4s
	mov v2.16b, v16.16b
	fmla v7.4s, v9.4s, v3.4s
	mov v3.16b, v16.16b
	ldp q19, q16, [sp, #112]
	add v0.4s, v0.4s, v1.4s
	fmla v2.4s, v10.4s, v4.4s
	fmla v3.4s, v11.4s, v5.4s
	mov v4.16b, v16.16b
	mov v5.16b, v16.16b
	mov v18.16b, v19.16b
	mov v17.16b, v19.16b
	fmla v4.4s, v12.4s, v6.4s
	mov v6.16b, v16.16b
	fmla v5.4s, v9.4s, v7.4s
	mov v7.16b, v19.16b
	fmla v16.4s, v11.4s, v3.4s
	mov v3.16b, v24.16b
	fmla v6.4s, v10.4s, v2.4s
	fmov v2.4s, #1.00000000
	fmla v7.4s, v12.4s, v4.4s
	mov v4.16b, v24.16b
	fmla v17.4s, v9.4s, v5.4s
	fmla v19.4s, v11.4s, v16.4s
	fcmge v16.4s, v26.4s, v25.4s
	fmov v5.4s, #1.00000000
	fmla v18.4s, v10.4s, v6.4s
	fmov v6.4s, #1.00000000
	fmla v3.4s, v12.4s, v7.4s
	fcmgt v7.4s, v25.4s, v26.4s
	fmla v20.4s, v9.4s, v17.4s
	fmla v21.4s, v11.4s, v19.4s
	fcmgt v19.4s, v28.4s, v29.4s
	fmov v17.4s, #1.00000000
	fmla v4.4s, v10.4s, v18.4s
	fcmge v18.4s, v29.4s, v28.4s
	fmla v2.4s, v12.4s, v3.4s
	orr v3.16b, v7.16b, v16.16b
	shl v7.4s, v14.4s, #23
	shl v16.4s, v15.4s, #23
	fmla v5.4s, v9.4s, v20.4s
	fmla v17.4s, v11.4s, v21.4s
	fmla v6.4s, v10.4s, v4.4s
	shl v4.4s, v13.4s, #23
	orr v18.16b, v19.16b, v18.16b
	mvn v3.16b, v3.16b
	add v7.4s, v7.4s, v1.4s
	add v16.4s, v16.4s, v1.4s
	add v4.4s, v4.4s, v1.4s
	orn v3.16b, v3.16b, v18.16b
	fmul v2.4s, v2.4s, v7.4s
	fmul v5.4s, v5.4s, v16.4s
	fmul v0.4s, v6.4s, v0.4s
	cmeq v6.4s, v27.4s, #0
	cmeq v7.4s, v30.4s, #0
	fmul v4.4s, v17.4s, v4.4s
	cmeq v16.4s, v31.4s, #0
	cmeq v17.4s, v8.4s, #0
	addp d3, v3.2d
	bic v30.16b, v2.16b, v6.16b
	bic v27.16b, v5.16b, v7.16b
	bic v2.16b, v0.16b, v16.16b
	bic v0.16b, v4.16b, v17.16b
	fmov x9, d3
	cbz x9, .LBB7_17
	fcmeq v3.4s, v25.4s, v25.4s
	fcmeq v4.4s, v29.4s, v29.4s
	fcmeq v5.4s, v28.4s, v28.4s
	fcmeq v6.4s, v26.4s, v26.4s
	bif v30.16b, v25.16b, v3.16b
	bif v2.16b, v28.16b, v5.16b
	bif v0.16b, v29.16b, v4.16b
	bif v27.16b, v26.16b, v6.16b
	b .LBB7_17
.LBB7_24:
	mov w8, #64269
	fmov s3, w10
	fmov s4, w9
	movk w8, #74, lsl #16
	mov w9, #1267
	fmov v7.4s, #1.00000000
	dup v1.4s, w8
	mov w8, #64269
	movk w9, #16181, lsl #16
	movk w8, #16330, lsl #16
	mvni v17.4s, #127, msl #16
	cmhi v16.4s, v4.4s, v3.4s
	dup v0.4s, w8
	mov w8, #27455
	fmov v22.4s, #1.00000000
	movk w8, #16094, lsl #16
	mov w10, #63802
	mov w11, #50297
	and v20.16b, v16.16b, v7.16b
	mov w12, #38156
	mov w13, #22716
	stp q0, q1, [sp, #240]
	dup v1.4s, w9
	dup v0.4s, w8
	mov w8, #46233
	mov w9, #14483
	mov w14, #65005
	movk w8, #16147, lsl #16
	movk w9, #16246, lsl #16
	mov w15, #29208
	stp q0, q1, [sp, #208]
	fneg v1.4s, v17.4s
	dup v0.4s, w8
	mov w8, #43579
	movk w10, #14625, lsl #16
	movk w11, #15022, lsl #16
	movk w8, #16440, lsl #16
	movk w12, #15389, lsl #16
	movk w13, #15715, lsl #16
	movk w14, #15989, lsl #16
	movk w15, #16177, lsl #16
	stp q0, q1, [sp, #176]
	dup v1.4s, w9
	dup v0.4s, v20.s[0]
	mov w9, #1123942400
	stp q0, q1, [sp, #144]
	dup v0.4s, w8
	mov w8, #-1023672320
	str q0, [sp, #128]
	b .LBB7_26
.LBB7_25:
	stp q27, q28, [x16]
	stp q29, q30, [x16, #32]
.LBB7_26:
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB7_30
	cbz x21, .LBB7_41
	mov x0, xzr
	mov x16, x21
	mov x21, xzr
	fcmp s8, #0.0
	ldp q23, q24, [x16]
	ldp q25, q26, [x16, #32]
	b.eq .LBB7_31
.LBB7_29:
	movi v0.4s, #128, lsl #16
	ldp q3, q2, [sp, #240]
	movi v4.4s, #127, msl #16
	ldr q5, [sp, #224]
	fmov v6.4s, #-1.00000000
	ldp q20, q17, [sp, #160]
	fcmge v27.4s, v23.4s, v0.4s
	fcmge v29.4s, v24.4s, v0.4s
	fcmge v9.4s, v25.4s, v0.4s
	mov v18.16b, v17.16b
	mov v7.16b, v17.16b
	mov v19.16b, v20.16b
	and v27.16b, v27.16b, v23.16b
	and v29.16b, v29.16b, v24.16b
	and v9.16b, v9.16b, v25.16b
	fminnm v28.4s, v27.4s, v22.4s
	fminnm v29.4s, v29.4s, v22.4s
	fminnm v11.4s, v9.4s, v22.4s
	fmov s27, s8
	mov v8.16b, v17.16b
	cmeq v1.4s, v28.4s, #0
	add v28.4s, v28.4s, v2.4s
	mov v31.16b, v1.16b
	str q1, [sp, #288]
	ldr q1, [sp, #128]
	bsl v31.16b, v3.16b, v28.16b
	cmeq v28.4s, v29.4s, #0
	add v29.4s, v29.4s, v2.4s
	mov v10.16b, v28.16b
	and v30.16b, v31.16b, v4.16b
	bsl v10.16b, v3.16b, v29.16b
	add v30.4s, v30.4s, v5.4s
	fadd v29.4s, v30.4s, v6.4s
	fadd v30.4s, v30.4s, v22.4s
	and v12.16b, v10.16b, v4.16b
	add v12.4s, v12.4s, v5.4s
	fdiv v9.4s, v29.4s, v30.4s
	fcmge v30.4s, v26.4s, v0.4s
	cmeq v29.4s, v11.4s, #0
	add v11.4s, v11.4s, v2.4s
	ldr q0, [sp, #208]
	fadd v13.4s, v12.4s, v6.4s
	fadd v12.4s, v12.4s, v22.4s
	and v30.16b, v30.16b, v26.16b
	bit v11.16b, v3.16b, v29.16b
	fminnm v14.4s, v30.4s, v22.4s
	and v15.16b, v11.16b, v4.16b
	fdiv v12.4s, v13.4s, v12.4s
	cmeq v30.4s, v14.4s, #0
	add v13.4s, v14.4s, v2.4s
	add v14.4s, v15.4s, v5.4s
	bit v13.16b, v3.16b, v30.16b
	fadd v15.4s, v14.4s, v6.4s
	fadd v14.4s, v14.4s, v22.4s
	mvni v3.4s, #126
	and v2.16b, v13.16b, v4.16b
	mvni v4.4s, #126
	ssra v3.4s, v31.4s, #23
	mov v31.16b, v1.16b
	fdiv v14.4s, v15.4s, v14.4s
	add v2.4s, v2.4s, v5.4s
	ssra v4.4s, v10.4s, #23
	scvtf v3.4s, v3.4s
	mvni v10.4s, #126
	fmul v5.4s, v12.4s, v12.4s
	fadd v15.4s, v2.4s, v6.4s
	fadd v2.4s, v2.4s, v22.4s
	scvtf v4.4s, v4.4s
	ssra v10.4s, v11.4s, #23
	fmla v7.4s, v0.4s, v5.4s
	fdiv v2.4s, v15.4s, v2.4s
	fmul v15.4s, v9.4s, v9.4s
	fmul v6.4s, v14.4s, v14.4s
	fmla v18.4s, v0.4s, v15.4s
	fmla v19.4s, v15.4s, v18.4s
	mov v18.16b, v20.16b
	fmla v18.4s, v5.4s, v7.4s
	mov v7.16b, v17.16b
	dup v17.4s, w13
	fmla v31.4s, v15.4s, v19.4s
	mov v19.16b, v1.16b
	mov v15.16b, v20.16b
	fmul v16.4s, v2.4s, v2.4s
	fmla v7.4s, v0.4s, v6.4s
	fmla v19.4s, v5.4s, v18.4s
	mov v5.16b, v20.16b
	mvni v18.4s, #126
	fmla v3.4s, v9.4s, v31.4s
	dup v31.4s, w8
	fmla v8.4s, v0.4s, v16.4s
	ldr q0, [sp, #272]
	fmla v5.4s, v6.4s, v7.4s
	ssra v18.4s, v13.4s, #23
	mov v7.16b, v1.16b
	fmla v4.4s, v12.4s, v19.4s
	mov v19.16b, v1.16b
	movi v1.4s, #67, lsl #24
	fmul v9.4s, v3.4s, v0.s[0]
	fmla v15.4s, v16.4s, v8.4s
	scvtf v8.4s, v10.4s
	scvtf v3.4s, v18.4s
	fmla v7.4s, v6.4s, v5.4s
	dup v6.4s, w9
	fmul v11.4s, v4.4s, v0.s[0]
	fmaxnm v4.4s, v9.4s, v31.4s
	fmla v19.4s, v16.4s, v15.4s
	dup v15.4s, w12
	fmla v8.4s, v14.4s, v7.4s
	dup v14.4s, w11
	fminnm v4.4s, v4.4s, v1.4s
	mov v21.16b, v15.16b
	fmla v3.4s, v2.4s, v19.4s
	fmaxnm v2.4s, v11.4s, v31.4s
	mov v20.16b, v14.16b
	fmul v12.4s, v8.4s, v0.s[0]
	dup v8.4s, w10
	mov v18.16b, v14.16b
	frintn v5.4s, v4.4s
	fmul v10.4s, v3.4s, v0.s[0]
	fminnm v2.4s, v2.4s, v1.4s
	dup v0.4s, w14
	fmaxnm v3.4s, v12.4s, v31.4s
	fminnm v5.4s, v5.4s, v6.4s
	fmaxnm v7.4s, v10.4s, v31.4s
	frintn v16.4s, v2.4s
	fminnm v3.4s, v3.4s, v1.4s
	fsub v4.4s, v4.4s, v5.4s
	fcvtns v5.4s, v5.4s
	fminnm v7.4s, v7.4s, v1.4s
	fminnm v16.4s, v16.4s, v6.4s
	frintn v19.4s, v3.4s
	fmla v20.4s, v8.4s, v4.4s
	shl v5.4s, v5.4s, #23
	frintn v13.4s, v7.4s
	fsub v2.4s, v2.4s, v16.4s
	fcvtns v16.4s, v16.4s
	fminnm v19.4s, v19.4s, v6.4s
	add v5.4s, v5.4s, v22.4s
	fmla v21.4s, v4.4s, v20.4s
	mov v20.16b, v15.16b
	fminnm v6.4s, v13.4s, v6.4s
	fmla v18.4s, v8.4s, v2.4s
	shl v16.4s, v16.4s, #23
	fsub v3.4s, v3.4s, v19.4s
	fcvtns v19.4s, v19.4s
	add v16.4s, v16.4s, v22.4s
	fsub v13.4s, v7.4s, v6.4s
	mov v7.16b, v17.16b
	fmla v20.4s, v2.4s, v18.4s
	mov v18.16b, v14.16b
	fcvtns v6.4s, v6.4s
	shl v19.4s, v19.4s, #23
	fmla v7.4s, v4.4s, v21.4s
	mov v21.16b, v17.16b
	fmla v18.4s, v8.4s, v3.4s
	fmla v14.4s, v8.4s, v13.4s
	mov v8.16b, v15.16b
	shl v6.4s, v6.4s, #23
	fmla v21.4s, v2.4s, v20.4s
	mov v20.16b, v0.16b
	fmla v8.4s, v3.4s, v18.4s
	dup v18.4s, w15
	fmla v15.4s, v13.4s, v14.4s
	fmla v20.4s, v4.4s, v7.4s
	mov v7.16b, v0.16b
	mov v14.16b, v17.16b
	add v6.4s, v6.4s, v22.4s
	fmla v7.4s, v2.4s, v21.4s
	mov v21.16b, v18.16b
	fmla v14.4s, v3.4s, v8.4s
	fmla v17.4s, v13.4s, v15.4s
	fmov v8.4s, #1.00000000
	fmov v15.4s, #1.00000000
	fmla v21.4s, v4.4s, v20.4s
	mov v20.16b, v18.16b
	fmla v20.4s, v2.4s, v7.4s
	mov v7.16b, v0.16b
	fmla v0.4s, v13.4s, v17.4s
	mov v17.16b, v18.16b
	fmla v8.4s, v4.4s, v21.4s
	fmov v4.4s, #1.00000000
	fmov v21.4s, #1.00000000
	fmla v7.4s, v3.4s, v14.4s
	fmla v15.4s, v2.4s, v20.4s
	fcmge v2.4s, v9.4s, v1.4s
	fcmgt v20.4s, v31.4s, v9.4s
	fcmge v9.4s, v11.4s, v1.4s
	fmla v18.4s, v13.4s, v0.4s
	fcmgt v0.4s, v31.4s, v12.4s
	fmul v5.4s, v8.4s, v5.4s
	fmov s8, s27
	fmla v17.4s, v3.4s, v7.4s
	fcmgt v7.4s, v31.4s, v11.4s
	fcmge v11.4s, v12.4s, v1.4s
	fcmge v12.4s, v10.4s, v1.4s
	fcmgt v31.4s, v31.4s, v10.4s
	fmul v16.4s, v15.4s, v16.4s
	fmla v21.4s, v13.4s, v18.4s
	add v18.4s, v19.4s, v22.4s
	ldr q1, [sp, #288]
	fmla v4.4s, v3.4s, v17.4s
	mvn v3.16b, v2.16b
	mvn v17.16b, v9.16b
	mvn v19.16b, v11.16b
	mvn v10.16b, v12.16b
	mov v27.16b, v1.16b
	fmul v6.4s, v21.4s, v6.4s
	bic v3.16b, v3.16b, v20.16b
	bic v7.16b, v17.16b, v7.16b
	fmul v4.4s, v4.4s, v18.4s
	bic v0.16b, v19.16b, v0.16b
	bic v17.16b, v10.16b, v31.16b
	and v3.16b, v3.16b, v5.16b
	and v5.16b, v7.16b, v16.16b
	ldr q16, [sp, #192]
	and v0.16b, v0.16b, v4.16b
	and v4.16b, v17.16b, v6.16b
	and v2.16b, v2.16b, v16.16b
	and v6.16b, v9.16b, v16.16b
	and v7.16b, v11.16b, v16.16b
	and v16.16b, v12.16b, v16.16b
	orr v2.16b, v3.16b, v2.16b
	orr v3.16b, v5.16b, v6.16b
	orr v0.16b, v0.16b, v7.16b
	orr v4.16b, v4.16b, v16.16b
	ldr q5, [sp, #144]
	bsl v27.16b, v5.16b, v2.16b
	bsl v28.16b, v5.16b, v3.16b
	bsl v29.16b, v5.16b, v0.16b
	bsl v30.16b, v5.16b, v4.16b
	b .LBB7_32
.LBB7_30:
	add x17, x0, #64
	mov x16, x0
	mov x0, x17
	fcmp s8, #0.0
	ldp q23, q24, [x16]
	ldp q25, q26, [x16, #32]
	b.ne .LBB7_29
.LBB7_31:
	fcmgt v0.4s, v23.4s, #0.0
	fcmgt v2.4s, v24.4s, #0.0
	fcmgt v3.4s, v25.4s, #0.0
	fcmgt v4.4s, v26.4s, #0.0
	and v27.16b, v0.16b, v22.16b
	and v28.16b, v2.16b, v22.16b
	and v29.16b, v3.16b, v22.16b
	and v30.16b, v4.16b, v22.16b
.LBB7_32:
	fcmge v0.4s, v24.4s, v23.4s
	fcmgt v2.4s, v23.4s, v24.4s
	fcmge v3.4s, v26.4s, v25.4s
	fcmgt v4.4s, v25.4s, v26.4s
	orr v0.16b, v2.16b, v0.16b
	orr v2.16b, v4.16b, v3.16b
	mvn v0.16b, v0.16b
	orn v0.16b, v0.16b, v2.16b
	addp d0, v0.2d
	fmov x17, d0
	cbz x17, .LBB7_25
	fcmeq v0.4s, v23.4s, v23.4s
	fcmeq v2.4s, v26.4s, v26.4s
	fcmeq v3.4s, v25.4s, v25.4s
	fcmeq v4.4s, v24.4s, v24.4s
	bif v27.16b, v23.16b, v0.16b
	bif v29.16b, v25.16b, v3.16b
	bif v30.16b, v26.16b, v2.16b
	bif v28.16b, v24.16b, v4.16b
	b .LBB7_25
.LBB7_34:
	mov x19, x0
	fmov s8, s0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	fmov s0, s8
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB7_2
.LBB7_35:
	add sp, sp, #576
	.cfi_def_cfa wsp, 160
	ldp x20, x19, [sp, #144]
	ldr x28, [sp, #80]
	ldp x22, x21, [sp, #128]
	ldp x24, x23, [sp, #112]
	ldp x26, x25, [sp, #96]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #160
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w26
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
	b linear_srgb::simd::linear_to_gamma_slice_tier_scalar_t
.LBB7_36:
	.cfi_restore_state
	mov x23, x1
	sub x24, x29, #192
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB7_40
.LBB7_37:
	mov x1, x23
	cbz x21, .LBB7_41
	mov x25, xzr
	mov x0, x21
	mov x21, xzr
.LBB7_39:
	ldp q0, q1, [x0]
	sub x8, x29, #192
	mov x26, x0
	stp q0, q1, [x24, #64]
	ldp q0, q2, [x0, #32]
	sub x0, x29, #128
	stp q0, q2, [x24, #96]
	fmov s0, s8
	bl linear_srgb::tf::math::gamma_outside_fit::<16>
	ldp q0, q1, [x24]
	mov x0, x25
	ldp q2, q3, [x24, #32]
	stp q0, q1, [x26]
	stp q2, q3, [x26, #32]
	cmp x25, x22
	ccmp x25, #0, #4, ne
	b.eq .LBB7_37
.LBB7_40:
	add x25, x0, #64
	b .LBB7_39
.LBB7_41:
	ldr w8, [sp, #316]
	cbz w8, .LBB7_43
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB7_43:
	add sp, sp, #576
	.cfi_def_cfa wsp, 160
	ldp x20, x19, [sp, #144]
	ldr x28, [sp, #80]
	ldp x22, x21, [sp, #128]
	ldp x24, x23, [sp, #112]
	ldp x26, x25, [sp, #96]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #160
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w26
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
