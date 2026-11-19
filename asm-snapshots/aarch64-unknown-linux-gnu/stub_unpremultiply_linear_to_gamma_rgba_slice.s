.section .text.stub_unpremultiply_linear_to_gamma_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_gamma_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_gamma_rgba_slice,@function
stub_unpremultiply_linear_to_gamma_rgba_slice:
	.cfi_startproc
	stp d15, d14, [sp, #-160]!
	.cfi_def_cfa_offset 160
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	stp x29, x30, [sp, #64]
	stp x28, x27, [sp, #80]
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
	.cfi_offset w27, -72
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
	b.eq .LBB15_26
	cmp w8, #2
	b.ne .LBB15_25
.LBB15_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x19, x1, #0xc
	lsr x20, x1, #4
	add x22, x0, x8, lsl #2
	lsl x23, x19, #2
	stp x23, x22, [sp, #32]
	b.eq .LBB15_4
	mov w8, #64
	mov x21, x0
	mov w1, wzr
	sub x2, x8, x23
	add x8, sp, #368
	fmov s8, s0
	add x0, x8, x23
	bl memset
	add x0, sp, #368
	mov x1, x22
	mov x2, x23
	bl memcpy
	ldp q1, q2, [sp, #368]
	add x8, sp, #49
	fmov s0, s8
	mov x0, x21
	add x9, sp, #97
	stur q1, [x8, #255]
	ldp q1, q3, [sp, #400]
	add x8, sp, #65
	stur q2, [x8, #255]
	add x8, sp, #81
	stur q1, [x8, #255]
	mov w8, #1
	stur q3, [x9, #255]
	b .LBB15_5
.LBB15_4:
	mov w8, wzr
.LBB15_5:
	fmov s1, #1.00000000
	str w8, [sp, #300]
	add x8, sp, #300
	add x1, x8, #4
	mov w9, #8388607
	mov w12, #64269
	fmov s3, w9
	mov w14, #14483
	movk w12, #16330, lsl #16
	fdiv s8, s1, s0
	fmov s1, #4.00000000
	movi d0, #0000000000000000
	movk w14, #16246, lsl #16
	fmov v13.4s, #1.00000000
	mov w10, #981467136
	dup v6.4s, w12
	dup v4.4s, w10
	mov w11, #64269
	movk w11, #74, lsl #16
	mov w13, #46233
	add x22, x0, x20, lsl #6
	dup v5.4s, w11
	movk w13, #16147, lsl #16
	mov w16, #39322
	str q4, [sp, #208]
	mov w17, #43579
	mov w21, #63802
	mov w20, #38156
	mov w23, #22716
	mov w27, #65005
	mov w28, #29208
	mov w11, #21845
	sub x15, x29, #208
	movk w16, #16409, lsl #16
	movk w17, #16440, lsl #16
	mov w10, #1123942400
	movk w21, #14625, lsl #16
	movk w20, #15389, lsl #16
	movk w23, #15715, lsl #16
	movk w27, #15989, lsl #16
	movk w28, #16177, lsl #16
	movk w11, #16085, lsl #16
	str w16, [sp, #200]
	fcmp s8, s1
	str s8, [sp, #204]
	fcsel s1, s1, s8, gt
	fccmp s8, s0, #8, ls
	fmov w8, s1
	cset w24, ge
	cmp x19, #0
	csel x26, xzr, x1, eq
	mov w19, #50297
	movk w19, #15022, lsl #16
	sub w8, w8, #1
	cmp w8, w9
	fmov s2, w8
	mov w8, #1267
	fcsel s0, s0, s1, lo
	movk w8, #16181, lsl #16
	mov w9, #27455
	dup v4.4s, w8
	movk w9, #16094, lsl #16
	cmhi v2.4s, v3.4s, v2.4s
	mvni v3.4s, #127, msl #16
	str q0, [sp, #64]
	dup v0.4s, w14
	stp q4, q5, [sp, #160]
	dup v5.4s, w9
	dup v4.4s, w13
	and v2.16b, v2.16b, v13.16b
	mov w9, #-1023672320
	stp q6, q0, [sp, #96]
	fneg v0.4s, v3.4s
	stp q4, q5, [sp, #128]
	str q0, [sp, #80]
	dup v0.4s, v2.s[0]
	str q0, [sp, #48]
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB15_8
.LBB15_6:
	cbz x26, .LBB15_22
	mov x0, xzr
	mov x25, x26
	mov x26, xzr
	b .LBB15_9
.LBB15_8:
	add x8, x0, #64
	mov x25, x0
	mov x0, x8
.LBB15_9:
	ldp q20, q15, [x25]
	ldr q16, [sp, #208]
	ldp q22, q21, [x25, #32]
	dup v0.4s, v20.s[3]
	dup v2.4s, v15.s[3]
	dup v4.4s, v22.s[3]
	dup v6.4s, v21.s[3]
	stp q21, q20, [sp, #256]
	str q22, [sp, #240]
	fdiv v1.4s, v13.4s, v0.4s
	fcmgt v0.4s, v0.4s, v16.4s
	fdiv v3.4s, v13.4s, v2.4s
	fcmgt v2.4s, v2.4s, v16.4s
	and v0.16b, v0.16b, v1.16b
	fdiv v5.4s, v13.4s, v4.4s
	fcmgt v4.4s, v4.4s, v16.4s
	and v1.16b, v2.16b, v3.16b
	fmul v3.4s, v20.4s, v0.4s
	fmul v2.4s, v15.4s, v1.4s
	fdiv v7.4s, v13.4s, v6.4s
	fcmgt v6.4s, v6.4s, v16.4s
	and v4.16b, v4.16b, v5.16b
	fmul v1.4s, v22.4s, v4.4s
	and v5.16b, v6.16b, v7.16b
	fmul v0.4s, v21.4s, v5.4s
	tbz w24, #0, .LBB15_21
	ldr s4, [sp, #200]
	fcmp s8, s4
	b.ne .LBB15_12
	movi v20.4s, #128, lsl #16
	ldp q29, q27, [sp, #160]
	ldr q28, [sp, #96]
	movi v25.4s, #127, msl #16
	fmov v26.4s, #-1.00000000
	ldp q9, q8, [sp, #128]
	dup v31.4s, w17
	str q0, [sp, #224]
	fcmge v4.4s, v3.4s, v20.4s
	fcmge v6.4s, v2.4s, v20.4s
	fcmge v17.4s, v1.4s, v20.4s
	mov v12.16b, v31.16b
	and v4.16b, v4.16b, v3.16b
	and v6.16b, v6.16b, v2.16b
	and v17.16b, v17.16b, v1.16b
	fminnm v5.4s, v4.4s, v13.4s
	fminnm v6.4s, v6.4s, v13.4s
	fminnm v17.4s, v17.4s, v13.4s
	cmeq v4.4s, v5.4s, #0
	add v5.4s, v5.4s, v27.4s
	mov v16.16b, v4.16b
	bsl v16.16b, v28.16b, v5.16b
	cmeq v5.4s, v6.4s, #0
	add v6.4s, v6.4s, v27.4s
	mov v22.16b, v5.16b
	and v7.16b, v16.16b, v25.16b
	bsl v22.16b, v28.16b, v6.16b
	add v7.4s, v7.4s, v29.4s
	fadd v6.4s, v7.4s, v26.4s
	fadd v7.4s, v7.4s, v13.4s
	and v19.16b, v22.16b, v25.16b
	add v19.4s, v19.4s, v29.4s
	fdiv v18.4s, v6.4s, v7.4s
	fcmge v7.4s, v0.4s, v20.4s
	cmeq v6.4s, v17.4s, #0
	add v17.4s, v17.4s, v27.4s
	fadd v20.4s, v19.4s, v26.4s
	fadd v19.4s, v19.4s, v13.4s
	and v7.16b, v7.16b, v0.16b
	bit v17.16b, v28.16b, v6.16b
	mov v0.16b, v15.16b
	dup v15.4s, w23
	fminnm v23.4s, v7.4s, v13.4s
	and v24.16b, v17.16b, v25.16b
	fdiv v21.4s, v20.4s, v19.4s
	cmeq v7.4s, v23.4s, #0
	add v19.4s, v23.4s, v27.4s
	add v20.4s, v24.4s, v29.4s
	mov v27.16b, v9.16b
	mov v23.16b, v7.16b
	bsl v23.16b, v28.16b, v19.16b
	fadd v19.4s, v20.4s, v26.4s
	fadd v20.4s, v20.4s, v13.4s
	mvni v28.4s, #126
	and v24.16b, v23.16b, v25.16b
	fdiv v19.4s, v19.4s, v20.4s
	ssra v28.4s, v16.4s, #23
	mov v16.16b, v9.16b
	fmul v25.4s, v21.4s, v21.4s
	add v20.4s, v24.4s, v29.4s
	mvni v29.4s, #126
	fadd v24.4s, v20.4s, v26.4s
	fadd v20.4s, v20.4s, v13.4s
	mov v26.16b, v9.16b
	ldr q13, [sp, #112]
	fmla v27.4s, v8.4s, v25.4s
	ssra v29.4s, v22.4s, #23
	scvtf v22.4s, v28.4s
	mvni v28.4s, #126
	mov v30.16b, v13.16b
	fdiv v20.4s, v24.4s, v20.4s
	fmul v24.4s, v18.4s, v18.4s
	scvtf v29.4s, v29.4s
	ssra v28.4s, v17.4s, #23
	fmul v10.4s, v19.4s, v19.4s
	fmla v26.4s, v8.4s, v24.4s
	fmla v16.4s, v8.4s, v10.4s
	fmla v30.4s, v24.4s, v26.4s
	mov v26.16b, v13.16b
	fmla v26.4s, v25.4s, v27.4s
	mov v27.16b, v9.16b
	dup v9.4s, w28
	fmla v12.4s, v24.4s, v30.4s
	mov v24.16b, v31.16b
	mov v30.16b, v13.16b
	fmul v11.4s, v20.4s, v20.4s
	fmla v24.4s, v25.4s, v26.4s
	mov v26.16b, v13.16b
	mvni v25.4s, #126
	fmla v22.4s, v18.4s, v12.4s
	dup v12.4s, w16
	fmla v27.4s, v8.4s, v11.4s
	movi v8.4s, #67, lsl #24
	fmla v26.4s, v10.4s, v16.4s
	ssra v25.4s, v23.4s, #23
	scvtf v23.4s, v28.4s
	fmla v29.4s, v21.4s, v24.4s
	mov v21.16b, v31.16b
	dup v16.4s, w9
	fmul v17.4s, v22.4s, v12.4s
	fmla v30.4s, v11.4s, v27.4s
	scvtf v22.4s, v25.4s
	fmla v21.4s, v10.4s, v26.4s
	dup v26.4s, w10
	dup v10.4s, w21
	fmul v18.4s, v29.4s, v12.4s
	fmla v31.4s, v11.4s, v30.4s
	fmla v23.4s, v19.4s, v21.4s
	fmaxnm v21.4s, v17.4s, v16.4s
	fmaxnm v24.4s, v18.4s, v16.4s
	fmla v22.4s, v20.4s, v31.4s
	dup v31.4s, w19
	fmul v19.4s, v23.4s, v12.4s
	fminnm v23.4s, v21.4s, v8.4s
	fminnm v24.4s, v24.4s, v8.4s
	mov v11.16b, v31.16b
	mov v13.16b, v31.16b
	fmul v20.4s, v22.4s, v12.4s
	dup v12.4s, w20
	fmaxnm v21.4s, v19.4s, v16.4s
	frintn v22.4s, v23.4s
	frintn v27.4s, v24.4s
	mov v14.16b, v12.16b
	fmaxnm v25.4s, v20.4s, v16.4s
	fminnm v29.4s, v21.4s, v8.4s
	fminnm v21.4s, v22.4s, v26.4s
	fminnm v22.4s, v27.4s, v26.4s
	fminnm v30.4s, v25.4s, v8.4s
	fsub v28.4s, v23.4s, v21.4s
	frintn v25.4s, v29.4s
	fcvtns v21.4s, v21.4s
	fsub v27.4s, v24.4s, v22.4s
	fcvtns v22.4s, v22.4s
	frintn v23.4s, v30.4s
	fmla v11.4s, v10.4s, v28.4s
	fminnm v25.4s, v25.4s, v26.4s
	shl v21.4s, v21.4s, #23
	fmla v13.4s, v10.4s, v27.4s
	shl v22.4s, v22.4s, #23
	fminnm v26.4s, v23.4s, v26.4s
	fmla v14.4s, v28.4s, v11.4s
	mov v11.16b, v12.16b
	fsub v24.4s, v29.4s, v25.4s
	dup v29.4s, w27
	fcvtns v25.4s, v25.4s
	fmla v11.4s, v27.4s, v13.4s
	mov v13.16b, v15.16b
	fsub v23.4s, v30.4s, v26.4s
	mov v30.16b, v31.16b
	fcvtns v26.4s, v26.4s
	shl v25.4s, v25.4s, #23
	fmla v13.4s, v28.4s, v14.4s
	mov v14.16b, v15.16b
	fmla v30.4s, v10.4s, v24.4s
	fmla v31.4s, v10.4s, v23.4s
	mov v10.16b, v12.16b
	shl v26.4s, v26.4s, #23
	fmla v14.4s, v27.4s, v11.4s
	mov v11.16b, v29.16b
	fmla v10.4s, v24.4s, v30.4s
	fmla v12.4s, v23.4s, v31.4s
	mov v30.16b, v9.16b
	fmla v11.4s, v28.4s, v13.4s
	mov v13.16b, v29.16b
	fmov v31.4s, #1.00000000
	fmla v13.4s, v27.4s, v14.4s
	mov v14.16b, v15.16b
	fmla v15.4s, v23.4s, v12.4s
	mov v12.16b, v29.16b
	fmla v30.4s, v28.4s, v11.4s
	fmov v11.4s, #1.00000000
	fmla v14.4s, v24.4s, v10.4s
	mov v10.16b, v9.16b
	fmla v29.4s, v23.4s, v15.4s
	mov v15.16b, v0.16b
	fmla v31.4s, v28.4s, v30.4s
	fmov v28.4s, #1.00000000
	fmov v30.4s, #1.00000000
	fmla v10.4s, v27.4s, v13.4s
	mov v13.16b, v9.16b
	fmla v12.4s, v24.4s, v14.4s
	fmla v9.4s, v23.4s, v29.4s
	fcmge v29.4s, v20.4s, v8.4s
	fmla v11.4s, v27.4s, v10.4s
	fcmge v10.4s, v18.4s, v8.4s
	fcmge v27.4s, v17.4s, v8.4s
	fmla v13.4s, v24.4s, v12.4s
	fcmge v12.4s, v19.4s, v8.4s
	fcmgt v18.4s, v16.4s, v18.4s
	fmla v30.4s, v23.4s, v9.4s
	fcmgt v17.4s, v16.4s, v17.4s
	fcmgt v19.4s, v16.4s, v19.4s
	fcmgt v16.4s, v16.4s, v20.4s
	mvn v9.16b, v29.16b
	mvn v23.16b, v10.16b
	mvn v20.16b, v27.16b
	fmla v28.4s, v24.4s, v13.4s
	fmov v13.4s, #1.00000000
	bic v18.16b, v23.16b, v18.16b
	bic v16.16b, v9.16b, v16.16b
	bic v17.16b, v20.16b, v17.16b
	add v24.4s, v25.4s, v13.4s
	add v25.4s, v26.4s, v13.4s
	add v21.4s, v21.4s, v13.4s
	add v22.4s, v22.4s, v13.4s
	mvn v26.16b, v12.16b
	fmul v23.4s, v30.4s, v25.4s
	fmul v20.4s, v31.4s, v21.4s
	fmul v21.4s, v11.4s, v22.4s
	fmul v22.4s, v28.4s, v24.4s
	bic v19.16b, v26.16b, v19.16b
	and v16.16b, v16.16b, v23.16b
	ldr q23, [sp, #80]
	and v17.16b, v17.16b, v20.16b
	and v18.16b, v18.16b, v21.16b
	and v19.16b, v19.16b, v22.16b
	and v20.16b, v27.16b, v23.16b
	and v21.16b, v10.16b, v23.16b
	and v22.16b, v12.16b, v23.16b
	and v23.16b, v29.16b, v23.16b
	orr v17.16b, v17.16b, v20.16b
	orr v18.16b, v18.16b, v21.16b
	orr v19.16b, v19.16b, v22.16b
	orr v16.16b, v16.16b, v23.16b
	ldp q21, q20, [sp, #256]
	ldp q0, q22, [sp, #224]
	bic v4.16b, v17.16b, v4.16b
	bic v5.16b, v18.16b, v5.16b
	bic v6.16b, v19.16b, v6.16b
	bic v7.16b, v16.16b, v7.16b
	b .LBB15_17
.LBB15_12:
	fmov s4, w11
	fcmp s8, s4
	b.ne .LBB15_14
	movi v19.4s, #128, lsl #16
	ldp q27, q26, [sp, #160]
	movi v24.4s, #127, msl #16
	fmov v25.4s, #-1.00000000
	ldr q14, [sp, #112]
	ldp q10, q8, [sp, #128]
	dup v29.4s, w17
	mvni v31.4s, #126
	mvni v12.4s, #126
	fcmge v4.4s, v3.4s, v19.4s
	fcmge v5.4s, v2.4s, v19.4s
	fcmge v7.4s, v1.4s, v19.4s
	mov v11.16b, v29.16b
	mov v9.16b, v10.16b
	and v4.16b, v4.16b, v3.16b
	and v5.16b, v5.16b, v2.16b
	and v7.16b, v7.16b, v1.16b
	fminnm v4.4s, v4.4s, v13.4s
	fminnm v5.4s, v5.4s, v13.4s
	add v16.4s, v4.4s, v26.4s
	add v21.4s, v5.4s, v26.4s
	cmeq v4.4s, v4.4s, #0
	cmeq v5.4s, v5.4s, #0
	and v6.16b, v16.16b, v24.16b
	and v18.16b, v21.16b, v24.16b
	ssra v31.4s, v16.4s, #23
	mvni v16.4s, #126
	ssra v12.4s, v21.4s, #23
	mov v21.16b, v14.16b
	add v6.4s, v6.4s, v27.4s
	fadd v17.4s, v6.4s, v25.4s
	fadd v6.4s, v6.4s, v13.4s
	fdiv v17.4s, v17.4s, v6.4s
	fminnm v6.4s, v7.4s, v13.4s
	add v7.4s, v18.4s, v27.4s
	fcmge v18.4s, v0.4s, v19.4s
	fadd v19.4s, v7.4s, v25.4s
	fadd v7.4s, v7.4s, v13.4s
	add v22.4s, v6.4s, v26.4s
	cmeq v6.4s, v6.4s, #0
	and v20.16b, v18.16b, v0.16b
	and v23.16b, v22.16b, v24.16b
	fdiv v18.4s, v19.4s, v7.4s
	fminnm v7.4s, v20.4s, v13.4s
	add v19.4s, v23.4s, v27.4s
	fadd v20.4s, v19.4s, v25.4s
	fadd v19.4s, v19.4s, v13.4s
	add v23.4s, v7.4s, v26.4s
	cmeq v7.4s, v7.4s, #0
	and v24.16b, v23.16b, v24.16b
	ssra v16.4s, v23.4s, #23
	mov v23.16b, v29.16b
	fdiv v19.4s, v20.4s, v19.4s
	add v20.4s, v24.4s, v27.4s
	mov v27.16b, v14.16b
	scvtf v16.4s, v16.4s
	fadd v24.4s, v20.4s, v25.4s
	fadd v20.4s, v20.4s, v13.4s
	mov v25.16b, v10.16b
	fmul v26.4s, v18.4s, v18.4s
	fdiv v20.4s, v24.4s, v20.4s
	fmul v24.4s, v17.4s, v17.4s
	fmul v28.4s, v19.4s, v19.4s
	fmla v25.4s, v8.4s, v24.4s
	fmla v9.4s, v8.4s, v28.4s
	fmla v27.4s, v24.4s, v25.4s
	mov v25.16b, v10.16b
	fmla v25.4s, v8.4s, v26.4s
	fmla v11.4s, v24.4s, v27.4s
	mov v27.16b, v14.16b
	mvni v24.4s, #126
	fmul v30.4s, v20.4s, v20.4s
	fmla v27.4s, v26.4s, v25.4s
	mov v25.16b, v14.16b
	ssra v24.4s, v22.4s, #23
	scvtf v22.4s, v31.4s
	dup v31.4s, w23
	fmla v10.4s, v8.4s, v30.4s
	fmla v25.4s, v28.4s, v9.4s
	scvtf v24.4s, v24.4s
	fmla v23.4s, v26.4s, v27.4s
	mov v26.16b, v29.16b
	scvtf v27.4s, v12.4s
	fmla v22.4s, v17.4s, v11.4s
	dup v17.4s, w11
	fmla v21.4s, v30.4s, v10.4s
	fmla v26.4s, v28.4s, v25.4s
	fmla v27.4s, v18.4s, v23.4s
	fmul v18.4s, v22.4s, v17.4s
	fmla v29.4s, v30.4s, v21.4s
	fmla v24.4s, v19.4s, v26.4s
	dup v26.4s, w19
	fmul v22.4s, v27.4s, v17.4s
	dup v27.4s, w21
	frintn v19.4s, v18.4s
	fmla v16.4s, v20.4s, v29.4s
	mov v28.16b, v26.16b
	mov v29.16b, v26.16b
	fmul v24.4s, v24.4s, v17.4s
	frintn v20.4s, v22.4s
	fsub v18.4s, v18.4s, v19.4s
	fcvtns v19.4s, v19.4s
	fmul v25.4s, v16.4s, v17.4s
	frintn v21.4s, v24.4s
	fsub v16.4s, v22.4s, v20.4s
	fcvtns v20.4s, v20.4s
	fmla v28.4s, v27.4s, v18.4s
	shl v19.4s, v19.4s, #23
	frintn v23.4s, v25.4s
	fsub v17.4s, v24.4s, v21.4s
	dup v24.4s, w20
	fcvtns v21.4s, v21.4s
	shl v20.4s, v20.4s, #23
	fsub v22.4s, v25.4s, v23.4s
	mov v25.16b, v26.16b
	mov v30.16b, v24.16b
	fmla v29.4s, v27.4s, v17.4s
	fcvtns v23.4s, v23.4s
	fmla v25.4s, v27.4s, v16.4s
	fmla v30.4s, v18.4s, v28.4s
	mov v28.16b, v24.16b
	fmla v26.4s, v27.4s, v22.4s
	mov v27.16b, v24.16b
	fmla v28.4s, v17.4s, v29.4s
	dup v29.4s, w27
	fmla v27.4s, v16.4s, v25.4s
	mov v25.16b, v31.16b
	fmla v24.4s, v22.4s, v26.4s
	mov v26.16b, v31.16b
	fmla v25.4s, v18.4s, v30.4s
	mov v30.16b, v31.16b
	fmla v26.4s, v16.4s, v27.4s
	mov v27.16b, v29.16b
	fmla v31.4s, v22.4s, v24.4s
	mov v24.16b, v29.16b
	fmla v30.4s, v17.4s, v28.4s
	dup v28.4s, w28
	fmla v27.4s, v18.4s, v25.4s
	mov v25.16b, v29.16b
	fmla v24.4s, v16.4s, v26.4s
	fmov v26.4s, #1.00000000
	fmla v29.4s, v22.4s, v31.4s
	mov v31.16b, v28.16b
	fmla v25.4s, v17.4s, v30.4s
	mov v30.16b, v28.16b
	fmla v31.4s, v16.4s, v24.4s
	fmov v24.4s, #1.00000000
	fmla v30.4s, v18.4s, v27.4s
	mov v27.16b, v28.16b
	fmla v28.4s, v22.4s, v29.4s
	fmov v29.4s, #1.00000000
	fmla v27.4s, v17.4s, v25.4s
	fmov v25.4s, #1.00000000
	fmla v24.4s, v16.4s, v31.4s
	fmla v26.4s, v18.4s, v30.4s
	shl v18.4s, v21.4s, #23
	shl v21.4s, v23.4s, #23
	fmla v29.4s, v22.4s, v28.4s
	add v16.4s, v19.4s, v13.4s
	fmla v25.4s, v17.4s, v27.4s
	add v17.4s, v20.4s, v13.4s
	add v18.4s, v18.4s, v13.4s
	add v19.4s, v21.4s, v13.4s
	fmul v16.4s, v26.4s, v16.4s
	ldr q20, [sp, #272]
	ldp q22, q21, [sp, #240]
	fmul v17.4s, v24.4s, v17.4s
	fmul v18.4s, v25.4s, v18.4s
	fmul v19.4s, v29.4s, v19.4s
	bic v4.16b, v16.16b, v4.16b
	bic v5.16b, v17.16b, v5.16b
	bic v6.16b, v18.16b, v6.16b
	bic v7.16b, v19.16b, v7.16b
	b .LBB15_17
.LBB15_14:
	fcmp s8, #0.0
	b.ne .LBB15_16
	fcmgt v4.4s, v3.4s, #0.0
	fcmgt v5.4s, v2.4s, #0.0
	fcmgt v6.4s, v1.4s, #0.0
	fcmgt v7.4s, v0.4s, #0.0
	and v4.16b, v4.16b, v13.16b
	and v5.16b, v5.16b, v13.16b
	and v6.16b, v6.16b, v13.16b
	and v7.16b, v7.16b, v13.16b
	b .LBB15_17
.LBB15_16:
	movi v21.4s, #128, lsl #16
	ldp q29, q27, [sp, #160]
	ldp q28, q12, [sp, #96]
	movi v25.4s, #127, msl #16
	fmov v26.4s, #-1.00000000
	ldp q11, q8, [sp, #128]
	dup v31.4s, w17
	str q0, [sp, #224]
	fcmge v4.4s, v3.4s, v21.4s
	fcmge v6.4s, v2.4s, v21.4s
	fcmge v17.4s, v1.4s, v21.4s
	mov v30.16b, v12.16b
	and v4.16b, v4.16b, v3.16b
	and v6.16b, v6.16b, v2.16b
	and v17.16b, v17.16b, v1.16b
	fminnm v5.4s, v4.4s, v13.4s
	fminnm v6.4s, v6.4s, v13.4s
	fminnm v19.4s, v17.4s, v13.4s
	cmeq v4.4s, v5.4s, #0
	add v5.4s, v5.4s, v27.4s
	mov v16.16b, v4.16b
	bsl v16.16b, v28.16b, v5.16b
	cmeq v5.4s, v6.4s, #0
	add v6.4s, v6.4s, v27.4s
	mov v18.16b, v5.16b
	and v7.16b, v16.16b, v25.16b
	bsl v18.16b, v28.16b, v6.16b
	add v7.4s, v7.4s, v29.4s
	fadd v6.4s, v7.4s, v26.4s
	fadd v7.4s, v7.4s, v13.4s
	and v20.16b, v18.16b, v25.16b
	add v20.4s, v20.4s, v29.4s
	fdiv v17.4s, v6.4s, v7.4s
	fcmge v7.4s, v0.4s, v21.4s
	cmeq v6.4s, v19.4s, #0
	add v19.4s, v19.4s, v27.4s
	mov v21.16b, v6.16b
	and v7.16b, v7.16b, v0.16b
	mov v0.16b, v15.16b
	bsl v21.16b, v28.16b, v19.16b
	fadd v19.4s, v20.4s, v26.4s
	fadd v20.4s, v20.4s, v13.4s
	fminnm v23.4s, v7.4s, v13.4s
	and v24.16b, v21.16b, v25.16b
	fdiv v22.4s, v19.4s, v20.4s
	cmeq v7.4s, v23.4s, #0
	add v19.4s, v23.4s, v27.4s
	mov v27.16b, v11.16b
	add v20.4s, v24.4s, v29.4s
	mov v23.16b, v7.16b
	bsl v23.16b, v28.16b, v19.16b
	fadd v19.4s, v20.4s, v26.4s
	fadd v20.4s, v20.4s, v13.4s
	mvni v28.4s, #126
	and v24.16b, v23.16b, v25.16b
	fdiv v19.4s, v19.4s, v20.4s
	ssra v28.4s, v16.4s, #23
	mov v16.16b, v11.16b
	fmul v25.4s, v22.4s, v22.4s
	add v20.4s, v24.4s, v29.4s
	mvni v29.4s, #126
	fadd v24.4s, v20.4s, v26.4s
	fadd v20.4s, v20.4s, v13.4s
	mov v26.16b, v11.16b
	fmla v27.4s, v8.4s, v25.4s
	ssra v29.4s, v18.4s, #23
	scvtf v18.4s, v28.4s
	mvni v28.4s, #126
	dup v13.4s, w23
	fdiv v20.4s, v24.4s, v20.4s
	fmul v24.4s, v17.4s, v17.4s
	scvtf v29.4s, v29.4s
	ssra v28.4s, v21.4s, #23
	mov v21.16b, v31.16b
	fmul v9.4s, v19.4s, v19.4s
	fmla v26.4s, v8.4s, v24.4s
	fmla v16.4s, v8.4s, v9.4s
	fmla v30.4s, v24.4s, v26.4s
	mov v26.16b, v12.16b
	fmla v26.4s, v25.4s, v27.4s
	mov v27.16b, v11.16b
	mov v11.16b, v31.16b
	fmul v10.4s, v20.4s, v20.4s
	fmla v11.4s, v24.4s, v30.4s
	mov v24.16b, v31.16b
	mov v30.16b, v12.16b
	fmla v24.4s, v25.4s, v26.4s
	mov v26.16b, v12.16b
	fmla v27.4s, v8.4s, v10.4s
	mvni v25.4s, #126
	fmla v18.4s, v17.4s, v11.4s
	movi v8.4s, #67, lsl #24
	dup v11.4s, w20
	fmla v26.4s, v9.4s, v16.4s
	dup v16.4s, w9
	fmla v30.4s, v10.4s, v27.4s
	fmla v29.4s, v22.4s, v24.4s
	ssra v25.4s, v23.4s, #23
	scvtf v22.4s, v28.4s
	mov v14.16b, v11.16b
	fmla v21.4s, v9.4s, v26.4s
	ldr q26, [sp, #64]
	scvtf v23.4s, v25.4s
	fmla v31.4s, v10.4s, v30.4s
	dup v9.4s, w21
	fmul v17.4s, v18.4s, v26.s[0]
	fmul v18.4s, v29.4s, v26.s[0]
	fmla v22.4s, v19.4s, v21.4s
	fmla v23.4s, v20.4s, v31.4s
	dup v31.4s, w19
	fmaxnm v21.4s, v17.4s, v16.4s
	fmaxnm v24.4s, v18.4s, v16.4s
	fmul v19.4s, v22.4s, v26.s[0]
	mov v12.16b, v31.16b
	fmul v20.4s, v23.4s, v26.s[0]
	fminnm v25.4s, v21.4s, v8.4s
	fminnm v23.4s, v24.4s, v8.4s
	dup v24.4s, w10
	fmaxnm v21.4s, v19.4s, v16.4s
	fmaxnm v26.4s, v20.4s, v16.4s
	frintn v22.4s, v25.4s
	frintn v27.4s, v23.4s
	fminnm v29.4s, v21.4s, v8.4s
	fminnm v30.4s, v26.4s, v8.4s
	fminnm v21.4s, v22.4s, v24.4s
	fminnm v22.4s, v27.4s, v24.4s
	frintn v26.4s, v29.4s
	frintn v10.4s, v30.4s
	fsub v28.4s, v25.4s, v21.4s
	fsub v27.4s, v23.4s, v22.4s
	mov v23.16b, v31.16b
	fcvtns v21.4s, v21.4s
	fcvtns v22.4s, v22.4s
	fminnm v25.4s, v26.4s, v24.4s
	fminnm v26.4s, v10.4s, v24.4s
	mov v10.16b, v11.16b
	fmla v23.4s, v9.4s, v28.4s
	fmla v12.4s, v9.4s, v27.4s
	shl v21.4s, v21.4s, #23
	shl v22.4s, v22.4s, #23
	fsub v24.4s, v29.4s, v25.4s
	dup v29.4s, w27
	fcvtns v25.4s, v25.4s
	fmla v10.4s, v28.4s, v23.4s
	fmla v14.4s, v27.4s, v12.4s
	mov v12.16b, v13.16b
	fsub v23.4s, v30.4s, v26.4s
	mov v30.16b, v31.16b
	fcvtns v26.4s, v26.4s
	shl v25.4s, v25.4s, #23
	fmla v12.4s, v28.4s, v10.4s
	mov v10.16b, v13.16b
	fmla v30.4s, v9.4s, v24.4s
	fmla v31.4s, v9.4s, v23.4s
	mov v9.16b, v11.16b
	shl v26.4s, v26.4s, #23
	fmla v10.4s, v27.4s, v14.4s
	mov v14.16b, v29.16b
	fmla v9.4s, v24.4s, v30.4s
	dup v30.4s, w28
	fmla v11.4s, v23.4s, v31.4s
	fmov v31.4s, #1.00000000
	fmla v14.4s, v28.4s, v12.4s
	mov v12.16b, v29.16b
	mov v15.16b, v30.16b
	fmla v12.4s, v27.4s, v10.4s
	mov v10.16b, v13.16b
	fmla v13.4s, v23.4s, v11.4s
	fmla v15.4s, v28.4s, v14.4s
	fmov v11.4s, #1.00000000
	fmla v10.4s, v24.4s, v9.4s
	mov v9.16b, v30.16b
	fmla v31.4s, v28.4s, v15.4s
	fmov v28.4s, #1.00000000
	mov v15.16b, v0.16b
	fmla v9.4s, v27.4s, v12.4s
	mov v12.16b, v29.16b
	fmla v29.4s, v23.4s, v13.4s
	mov v13.16b, v30.16b
	ldr q0, [sp, #224]
	fmla v12.4s, v24.4s, v10.4s
	fmov v10.4s, #1.00000000
	fmla v30.4s, v23.4s, v29.4s
	fmla v11.4s, v27.4s, v9.4s
	fcmge v9.4s, v18.4s, v8.4s
	fcmge v29.4s, v20.4s, v8.4s
	fcmge v27.4s, v17.4s, v8.4s
	fcmgt v18.4s, v16.4s, v18.4s
	fcmgt v17.4s, v16.4s, v17.4s
	fmla v13.4s, v24.4s, v12.4s
	fcmge v12.4s, v19.4s, v8.4s
	fcmgt v19.4s, v16.4s, v19.4s
	fmla v10.4s, v23.4s, v30.4s
	fcmgt v16.4s, v16.4s, v20.4s
	mvn v23.16b, v9.16b
	mvn v30.16b, v29.16b
	mvn v20.16b, v27.16b
	fmla v28.4s, v24.4s, v13.4s
	fmov v13.4s, #1.00000000
	bic v18.16b, v23.16b, v18.16b
	bic v16.16b, v30.16b, v16.16b
	bic v17.16b, v20.16b, v17.16b
	add v24.4s, v25.4s, v13.4s
	add v25.4s, v26.4s, v13.4s
	add v21.4s, v21.4s, v13.4s
	add v22.4s, v22.4s, v13.4s
	mvn v26.16b, v12.16b
	fmul v23.4s, v10.4s, v25.4s
	fmul v20.4s, v31.4s, v21.4s
	fmul v21.4s, v11.4s, v22.4s
	fmul v22.4s, v28.4s, v24.4s
	bic v19.16b, v26.16b, v19.16b
	and v16.16b, v16.16b, v23.16b
	ldr q23, [sp, #80]
	and v17.16b, v17.16b, v20.16b
	and v18.16b, v18.16b, v21.16b
	and v19.16b, v19.16b, v22.16b
	and v20.16b, v27.16b, v23.16b
	and v21.16b, v9.16b, v23.16b
	and v22.16b, v12.16b, v23.16b
	and v23.16b, v29.16b, v23.16b
	orr v17.16b, v17.16b, v20.16b
	orr v18.16b, v18.16b, v21.16b
	orr v19.16b, v19.16b, v22.16b
	orr v16.16b, v16.16b, v23.16b
	ldr q20, [sp, #48]
	ldp q22, q21, [sp, #240]
	bsl v4.16b, v20.16b, v17.16b
	bsl v5.16b, v20.16b, v18.16b
	bsl v6.16b, v20.16b, v19.16b
	bsl v7.16b, v20.16b, v16.16b
	ldr q20, [sp, #272]
.LBB15_17:
	fcmge v16.4s, v2.4s, v3.4s
	fcmgt v17.4s, v3.4s, v2.4s
	fcmge v18.4s, v0.4s, v1.4s
	fcmgt v19.4s, v1.4s, v0.4s
	orr v16.16b, v17.16b, v16.16b
	orr v17.16b, v19.16b, v18.16b
	mvn v16.16b, v16.16b
	orn v16.16b, v16.16b, v17.16b
	addp d16, v16.2d
	fmov x8, d16
	cbz x8, .LBB15_19
	fcmeq v16.4s, v3.4s, v3.4s
	fcmeq v17.4s, v0.4s, v0.4s
	fcmeq v18.4s, v1.4s, v1.4s
	fcmeq v19.4s, v2.4s, v2.4s
	bif v4.16b, v3.16b, v16.16b
	bif v6.16b, v1.16b, v18.16b
	bif v7.16b, v0.16b, v17.16b
	bif v5.16b, v2.16b, v19.16b
.LBB15_19:
	ldr s8, [sp, #204]
.LBB15_20:
	mov v4.s[3], v20.s[3]
	mov v5.s[3], v15.s[3]
	mov v6.s[3], v22.s[3]
	mov v7.s[3], v21.s[3]
	stp q4, q5, [x25]
	stp q6, q7, [x25, #32]
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB15_8
	b .LBB15_6
.LBB15_21:
	sub x8, x29, #208
	str x0, [sp, #224]
	sub x0, x29, #144
	stp q3, q2, [x15, #64]
	stp q1, q0, [x15, #96]
	fmov s0, s8
	str x1, [sp, #24]
	str q15, [sp]
	bl linear_srgb::tf::math::gamma_outside_fit::<16>
	sub x15, x29, #208
	ldp q22, q21, [sp, #240]
	ldp q4, q5, [x15]
	mov w11, #21845
	ldp q6, q7, [x15, #32]
	mov w17, #43579
	ldr q20, [sp, #272]
	ldr q15, [sp]
	mov w16, #39322
	ldr x1, [sp, #24]
	ldr x0, [sp, #224]
	movk w11, #16085, lsl #16
	mov w10, #1123942400
	mov w9, #-1023672320
	movk w17, #16440, lsl #16
	movk w16, #16409, lsl #16
	fmov v13.4s, #1.00000000
	b .LBB15_20
.LBB15_22:
	ldr w8, [sp, #300]
	cbz w8, .LBB15_24
	ldp x2, x0, [sp, #32]
	bl memcpy
.LBB15_24:
	add sp, sp, #576
	.cfi_def_cfa wsp, 160
	ldp x20, x19, [sp, #144]
	ldp x22, x21, [sp, #128]
	ldp x24, x23, [sp, #112]
	ldp x26, x25, [sp, #96]
	ldp x28, x27, [sp, #80]
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
	.cfi_restore w27
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
.LBB15_25:
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
.LBB15_26:
	add sp, sp, #576
	.cfi_def_cfa wsp, 160
	ldp x20, x19, [sp, #144]
	ldp x22, x21, [sp, #128]
	ldp x24, x23, [sp, #112]
	ldp x26, x25, [sp, #96]
	ldp x28, x27, [sp, #80]
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
	.cfi_restore w27
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
