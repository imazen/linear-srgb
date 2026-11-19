.section .text.stub_gamma_to_linear_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_slice
	.p2align	2
.type	stub_gamma_to_linear_slice,@function
stub_gamma_to_linear_slice:
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
	fmov s8, s0
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB6_36
	cmp w8, #2
	b.ne .LBB6_35
.LBB6_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x21, x1, #0xf
	lsr x22, x1, #4
	add x19, x0, x8, lsl #2
	lsl x20, x21, #2
	b.eq .LBB6_4
	mov w8, #64
	mov x23, x0
	mov w1, wzr
	sub x2, x8, x20
	sub x8, x29, #256
	add x0, x8, x20
	bl memset
	sub x0, x29, #256
	mov x1, x19
	mov x2, x20
	bl memcpy
	ldp q1, q2, [x29, #-256]
	add x8, sp, #65
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
	b .LBB6_5
.LBB6_4:
	mov w8, wzr
.LBB6_5:
	fmov s2, #4.00000000
	add x9, sp, #316
	cmp x21, #0
	add x1, x9, #4
	movi d3, #0000000000000000
	mov w10, #8388607
	csel x21, xzr, x1, eq
	add x22, x0, x22, lsl #6
	sub x23, x29, #192
	fcmp s8, s2
	str w8, [sp, #316]
	fcsel s1, s2, s8, gt
	fmov w9, s1
	sub w9, w9, #1
	cmp w9, w10
	fcsel s1, s3, s1, lo
	fcmp s8, #0.0
	str q1, [sp, #288]
	b.lt .LBB6_37
	fcmp s8, s2
	b.hi .LBB6_37
	mov w8, #39322
	movk w8, #16409, lsl #16
	fmov s2, w8
	fcmp s8, s2
	b.ne .LBB6_16
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
	b .LBB6_10
.LBB6_9:
	stp q4, q3, [x8]
	stp q2, q0, [x8, #32]
.LBB6_10:
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB6_13
	cbz x21, .LBB6_42
	mov x0, xzr
	mov x8, x21
	mov x21, xzr
	b .LBB6_14
.LBB6_13:
	add x9, x0, #64
	mov x8, x0
	mov x0, x9
.LBB6_14:
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
	cbz x9, .LBB6_9
	fcmeq v5.4s, v30.4s, v30.4s
	fcmeq v6.4s, v10.4s, v10.4s
	fcmeq v7.4s, v8.4s, v8.4s
	fcmeq v16.4s, v31.4s, v31.4s
	bif v4.16b, v30.16b, v5.16b
	bif v2.16b, v8.16b, v7.16b
	bif v0.16b, v10.16b, v6.16b
	bif v3.16b, v31.16b, v16.16b
	b .LBB6_9
.LBB6_16:
	mov w8, #21845
	movk w8, #16085, lsl #16
	fmov s2, w8
	fcmp s8, s2
	b.ne .LBB6_25
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
	b .LBB6_19
.LBB6_18:
	stp q30, q27, [x8]
	stp q2, q0, [x8, #32]
.LBB6_19:
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB6_22
	cbz x21, .LBB6_42
	mov x0, xzr
	mov x8, x21
	mov x21, xzr
	b .LBB6_23
.LBB6_22:
	add x9, x0, #64
	mov x8, x0
	mov x0, x9
.LBB6_23:
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
	cbz x9, .LBB6_18
	fcmeq v3.4s, v25.4s, v25.4s
	fcmeq v4.4s, v29.4s, v29.4s
	fcmeq v5.4s, v28.4s, v28.4s
	fcmeq v6.4s, v26.4s, v26.4s
	bif v30.16b, v25.16b, v3.16b
	bif v2.16b, v28.16b, v5.16b
	bif v0.16b, v29.16b, v4.16b
	bif v27.16b, v26.16b, v6.16b
	b .LBB6_18
.LBB6_25:
	mov w8, #64269
	fmov s4, w9
	fmov s5, w10
	movk w8, #74, lsl #16
	mov w9, #1267
	fmov v16.4s, #1.00000000
	dup v2.4s, w8
	mov w8, #64269
	movk w9, #16181, lsl #16
	movk w8, #16330, lsl #16
	mvni v18.4s, #127, msl #16
	cmhi v17.4s, v5.4s, v4.4s
	dup v1.4s, w8
	mov w8, #27455
	fmov v0.4s, #1.00000000
	movk w8, #16094, lsl #16
	mov w10, #63802
	mov w11, #50297
	dup v3.4s, w8
	mov w8, #46233
	mov w12, #38156
	stp q1, q2, [sp, #256]
	dup v2.4s, w9
	movk w8, #16147, lsl #16
	and v1.16b, v17.16b, v16.16b
	mov w9, #14483
	mov w13, #22716
	movk w9, #16246, lsl #16
	mov w14, #65005
	mov w15, #29208
	stp q3, q2, [sp, #224]
	fneg v2.4s, v18.4s
	dup v3.4s, w8
	dup v1.4s, v1.s[0]
	mov w8, #43579
	movk w10, #14625, lsl #16
	movk w8, #16440, lsl #16
	movk w11, #15022, lsl #16
	movk w12, #15389, lsl #16
	movk w13, #15715, lsl #16
	movk w14, #15989, lsl #16
	movk w15, #16177, lsl #16
	stp q3, q2, [sp, #192]
	dup v3.4s, w9
	mov w9, #1123942400
	str s8, [sp, #144]
	stp q1, q3, [sp, #160]
	dup v1.4s, w8
	mov w8, #-1023672320
	str q1, [sp, #128]
	b .LBB6_27
.LBB6_26:
	stp q28, q29, [x16]
	stp q30, q31, [x16, #32]
.LBB6_27:
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB6_31
	cbz x21, .LBB6_42
	mov x0, xzr
	mov x16, x21
	mov x21, xzr
	fcmp s8, #0.0
	ldp q24, q25, [x16]
	ldp q26, q27, [x16, #32]
	b.eq .LBB6_32
.LBB6_30:
	movi v1.4s, #128, lsl #16
	ldp q4, q3, [sp, #256]
	movi v5.4s, #127, msl #16
	ldr q6, [sp, #240]
	fmov v7.4s, #-1.00000000
	ldr q2, [sp, #128]
	dup v23.4s, w14
	fcmge v28.4s, v24.4s, v1.4s
	fcmge v30.4s, v25.4s, v1.4s
	fcmge v9.4s, v26.4s, v1.4s
	and v28.16b, v28.16b, v24.16b
	and v30.16b, v30.16b, v25.16b
	and v9.16b, v9.16b, v26.16b
	fminnm v29.4s, v28.4s, v0.4s
	fminnm v30.4s, v30.4s, v0.4s
	fminnm v11.4s, v9.4s, v0.4s
	cmeq v28.4s, v29.4s, #0
	add v29.4s, v29.4s, v3.4s
	mov v8.16b, v28.16b
	bsl v8.16b, v4.16b, v29.16b
	cmeq v29.4s, v30.4s, #0
	add v30.4s, v30.4s, v3.4s
	mov v10.16b, v29.16b
	and v31.16b, v8.16b, v5.16b
	bsl v10.16b, v4.16b, v30.16b
	add v31.4s, v31.4s, v6.4s
	fadd v30.4s, v31.4s, v7.4s
	fadd v31.4s, v31.4s, v0.4s
	and v12.16b, v10.16b, v5.16b
	add v12.4s, v12.4s, v6.4s
	fdiv v9.4s, v30.4s, v31.4s
	fcmge v31.4s, v27.4s, v1.4s
	cmeq v30.4s, v11.4s, #0
	add v11.4s, v11.4s, v3.4s
	mvni v1.4s, #126
	fadd v13.4s, v12.4s, v7.4s
	fadd v12.4s, v12.4s, v0.4s
	and v31.16b, v31.16b, v27.16b
	bit v11.16b, v4.16b, v30.16b
	ssra v1.4s, v8.4s, #23
	mov v8.16b, v2.16b
	fminnm v14.4s, v31.4s, v0.4s
	scvtf v1.4s, v1.4s
	and v15.16b, v11.16b, v5.16b
	fdiv v12.4s, v13.4s, v12.4s
	cmeq v31.4s, v14.4s, #0
	add v13.4s, v14.4s, v3.4s
	add v14.4s, v15.4s, v6.4s
	bit v13.16b, v4.16b, v31.16b
	fadd v15.4s, v14.4s, v7.4s
	fadd v14.4s, v14.4s, v0.4s
	ldp q21, q4, [sp, #176]
	and v3.16b, v13.16b, v5.16b
	mov v19.16b, v4.16b
	ldr q5, [sp, #224]
	mov v16.16b, v4.16b
	mov v20.16b, v21.16b
	fdiv v14.4s, v15.4s, v14.4s
	add v3.4s, v3.4s, v6.4s
	fmul v6.4s, v12.4s, v12.4s
	fadd v15.4s, v3.4s, v7.4s
	fadd v3.4s, v3.4s, v0.4s
	mvni v7.4s, #126
	fmla v16.4s, v5.4s, v6.4s
	ssra v7.4s, v10.4s, #23
	mvni v10.4s, #126
	fdiv v3.4s, v15.4s, v3.4s
	fmul v15.4s, v9.4s, v9.4s
	scvtf v7.4s, v7.4s
	ssra v10.4s, v11.4s, #23
	fmul v17.4s, v14.4s, v14.4s
	fmla v19.4s, v5.4s, v15.4s
	fmla v20.4s, v15.4s, v19.4s
	mov v19.16b, v21.16b
	fmla v19.4s, v6.4s, v16.4s
	mov v16.16b, v4.16b
	fmla v8.4s, v15.4s, v20.4s
	mov v20.16b, v2.16b
	mov v15.16b, v21.16b
	fmul v18.4s, v3.4s, v3.4s
	fmla v16.4s, v5.4s, v17.4s
	fmla v20.4s, v6.4s, v19.4s
	mov v6.16b, v21.16b
	mvni v19.4s, #126
	fmla v1.4s, v9.4s, v8.4s
	dup v8.4s, w8
	fmla v4.4s, v5.4s, v18.4s
	ldr q5, [sp, #288]
	fmla v6.4s, v17.4s, v16.4s
	ssra v19.4s, v13.4s, #23
	mov v16.16b, v2.16b
	fmla v7.4s, v12.4s, v20.4s
	scvtf v20.4s, v10.4s
	fmul v9.4s, v1.4s, v5.s[0]
	fmla v15.4s, v18.4s, v4.4s
	mov v4.16b, v2.16b
	scvtf v1.4s, v19.4s
	movi v2.4s, #67, lsl #24
	fmul v11.4s, v7.4s, v5.s[0]
	dup v7.4s, w9
	fmla v4.4s, v17.4s, v6.4s
	fmla v16.4s, v18.4s, v15.4s
	dup v15.4s, w12
	fmla v20.4s, v14.4s, v4.4s
	fmaxnm v4.4s, v9.4s, v8.4s
	dup v14.4s, w11
	fmla v1.4s, v3.4s, v16.4s
	fmaxnm v3.4s, v11.4s, v8.4s
	mov v21.16b, v15.16b
	mov v22.16b, v14.16b
	mov v18.16b, v14.16b
	fminnm v4.4s, v4.4s, v2.4s
	fmul v12.4s, v20.4s, v5.s[0]
	dup v20.4s, w10
	fmul v10.4s, v1.4s, v5.s[0]
	fminnm v1.4s, v3.4s, v2.4s
	dup v5.4s, w13
	frintn v6.4s, v4.4s
	fmaxnm v3.4s, v12.4s, v8.4s
	fmaxnm v16.4s, v10.4s, v8.4s
	frintn v17.4s, v1.4s
	fminnm v6.4s, v6.4s, v7.4s
	fminnm v3.4s, v3.4s, v2.4s
	fminnm v16.4s, v16.4s, v2.4s
	fminnm v17.4s, v17.4s, v7.4s
	fsub v4.4s, v4.4s, v6.4s
	frintn v19.4s, v3.4s
	fcvtns v6.4s, v6.4s
	frintn v13.4s, v16.4s
	fsub v1.4s, v1.4s, v17.4s
	fcvtns v17.4s, v17.4s
	fmla v22.4s, v20.4s, v4.4s
	fminnm v19.4s, v19.4s, v7.4s
	shl v6.4s, v6.4s, #23
	fminnm v7.4s, v13.4s, v7.4s
	fmla v18.4s, v20.4s, v1.4s
	shl v17.4s, v17.4s, #23
	fmla v21.4s, v4.4s, v22.4s
	mov v22.16b, v15.16b
	fsub v3.4s, v3.4s, v19.4s
	fsub v13.4s, v16.4s, v7.4s
	mov v16.16b, v5.16b
	fcvtns v19.4s, v19.4s
	fcvtns v7.4s, v7.4s
	fmla v22.4s, v1.4s, v18.4s
	mov v18.16b, v14.16b
	fmla v16.4s, v4.4s, v21.4s
	mov v21.16b, v5.16b
	fmla v14.4s, v20.4s, v13.4s
	shl v19.4s, v19.4s, #23
	fmla v18.4s, v20.4s, v3.4s
	mov v20.16b, v23.16b
	shl v7.4s, v7.4s, #23
	fmla v21.4s, v1.4s, v22.4s
	mov v22.16b, v15.16b
	fmla v20.4s, v4.4s, v16.4s
	mov v16.16b, v23.16b
	fmla v15.4s, v13.4s, v14.4s
	fmla v22.4s, v3.4s, v18.4s
	dup v18.4s, w15
	mov v14.16b, v5.16b
	add v7.4s, v7.4s, v0.4s
	fmla v16.4s, v1.4s, v21.4s
	mov v21.16b, v18.16b
	fmla v5.4s, v13.4s, v15.4s
	fmov v15.4s, #1.00000000
	fmla v14.4s, v3.4s, v22.4s
	fmov v22.4s, #1.00000000
	fmla v21.4s, v4.4s, v20.4s
	mov v20.16b, v18.16b
	fmla v20.4s, v1.4s, v16.4s
	mov v16.16b, v23.16b
	fmla v23.4s, v13.4s, v5.4s
	mov v5.16b, v18.16b
	fmla v22.4s, v4.4s, v21.4s
	fmov v4.4s, #1.00000000
	fmov v21.4s, #1.00000000
	fmla v16.4s, v3.4s, v14.4s
	fmla v18.4s, v13.4s, v23.4s
	fmla v15.4s, v1.4s, v20.4s
	fcmge v1.4s, v9.4s, v2.4s
	fcmgt v20.4s, v8.4s, v9.4s
	fcmge v9.4s, v11.4s, v2.4s
	fcmgt v23.4s, v8.4s, v12.4s
	fmla v5.4s, v3.4s, v16.4s
	fcmgt v16.4s, v8.4s, v11.4s
	fcmge v11.4s, v12.4s, v2.4s
	fcmge v12.4s, v10.4s, v2.4s
	fmla v21.4s, v13.4s, v18.4s
	add v18.4s, v19.4s, v0.4s
	fcmgt v8.4s, v8.4s, v10.4s
	fmla v4.4s, v3.4s, v5.4s
	add v5.4s, v6.4s, v0.4s
	add v6.4s, v17.4s, v0.4s
	mvn v17.16b, v9.16b
	mvn v19.16b, v11.16b
	mvn v3.16b, v1.16b
	mvn v10.16b, v12.16b
	fmul v7.4s, v21.4s, v7.4s
	fmul v5.4s, v22.4s, v5.4s
	fmul v6.4s, v15.4s, v6.4s
	fmul v4.4s, v4.4s, v18.4s
	bic v16.16b, v17.16b, v16.16b
	bic v17.16b, v19.16b, v23.16b
	bic v3.16b, v3.16b, v20.16b
	bic v18.16b, v10.16b, v8.16b
	ldr s8, [sp, #144]
	and v4.16b, v17.16b, v4.16b
	ldr q17, [sp, #208]
	and v3.16b, v3.16b, v5.16b
	and v5.16b, v16.16b, v6.16b
	and v6.16b, v18.16b, v7.16b
	and v1.16b, v1.16b, v17.16b
	and v7.16b, v9.16b, v17.16b
	and v16.16b, v11.16b, v17.16b
	and v17.16b, v12.16b, v17.16b
	orr v1.16b, v3.16b, v1.16b
	orr v3.16b, v5.16b, v7.16b
	orr v4.16b, v4.16b, v16.16b
	orr v5.16b, v6.16b, v17.16b
	ldr q6, [sp, #160]
	bsl v28.16b, v6.16b, v1.16b
	bsl v29.16b, v6.16b, v3.16b
	bsl v30.16b, v6.16b, v4.16b
	bsl v31.16b, v6.16b, v5.16b
	b .LBB6_33
.LBB6_31:
	add x17, x0, #64
	mov x16, x0
	mov x0, x17
	fcmp s8, #0.0
	ldp q24, q25, [x16]
	ldp q26, q27, [x16, #32]
	b.ne .LBB6_30
.LBB6_32:
	fcmgt v1.4s, v24.4s, #0.0
	fcmgt v3.4s, v25.4s, #0.0
	fcmgt v4.4s, v26.4s, #0.0
	fcmgt v5.4s, v27.4s, #0.0
	and v28.16b, v1.16b, v0.16b
	and v29.16b, v3.16b, v0.16b
	and v30.16b, v4.16b, v0.16b
	and v31.16b, v5.16b, v0.16b
.LBB6_33:
	fcmge v1.4s, v25.4s, v24.4s
	fcmgt v3.4s, v24.4s, v25.4s
	fcmge v4.4s, v27.4s, v26.4s
	fcmgt v5.4s, v26.4s, v27.4s
	orr v1.16b, v3.16b, v1.16b
	orr v3.16b, v5.16b, v4.16b
	mvn v1.16b, v1.16b
	orn v1.16b, v1.16b, v3.16b
	addp d1, v1.2d
	fmov x17, d1
	cbz x17, .LBB6_26
	fcmeq v1.4s, v24.4s, v24.4s
	fcmeq v3.4s, v27.4s, v27.4s
	fcmeq v4.4s, v26.4s, v26.4s
	fcmeq v5.4s, v25.4s, v25.4s
	bif v28.16b, v24.16b, v1.16b
	bif v30.16b, v26.16b, v4.16b
	bif v31.16b, v27.16b, v3.16b
	bif v29.16b, v25.16b, v5.16b
	b .LBB6_26
.LBB6_35:
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB6_2
.LBB6_36:
	fmov s0, s8
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
	b linear_srgb::simd::gamma_to_linear_slice_tier_scalar_t
.LBB6_37:
	.cfi_restore_state
	mov x24, x1
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB6_41
.LBB6_38:
	mov x1, x24
	cbz x21, .LBB6_42
	fmov s0, s8
	mov x24, x1
	mov x25, xzr
	mov x0, x21
	mov x21, xzr
.LBB6_40:
	ldp q1, q2, [x0]
	sub x8, x29, #192
	mov x26, x0
	fmov s8, s0
	stp q1, q2, [x23, #64]
	ldp q1, q3, [x0, #32]
	sub x0, x29, #128
	stp q1, q3, [x23, #96]
	bl linear_srgb::tf::math::gamma_outside_fit::<16>
	ldp q0, q1, [x23]
	mov x0, x25
	ldp q2, q3, [x23, #32]
	stp q0, q1, [x26]
	stp q2, q3, [x26, #32]
	cmp x25, x22
	ccmp x25, #0, #4, ne
	b.eq .LBB6_38
.LBB6_41:
	add x25, x0, #64
	fmov s0, s8
	b .LBB6_40
.LBB6_42:
	ldr w8, [sp, #316]
	cbz w8, .LBB6_44
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB6_44:
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
