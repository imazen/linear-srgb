.section .text.stub_gamma_to_linear_premultiply_rgba_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_premultiply_rgba_slice
	.p2align	2
.type	stub_gamma_to_linear_premultiply_rgba_slice,@function
stub_gamma_to_linear_premultiply_rgba_slice:
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
	sub sp, sp, #640
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB5_22
	cmp w8, #2
	b.ne .LBB5_21
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
	sub x8, x29, #256
	fmov s8, s0
	add x0, x8, x20
	bl memset
	sub x0, x29, #256
	mov x1, x19
	mov x2, x20
	bl memcpy
	ldp q2, q1, [x29, #-256]
	add x8, sp, #129
	fmov s0, s8
	mov x0, x23
	add x9, sp, #177
	stur q2, [x8, #255]
	ldp q3, q2, [x29, #-224]
	add x8, sp, #145
	stur q1, [x8, #255]
	add x8, sp, #161
	stur q3, [x8, #255]
	mov w8, #1
	stur q2, [x9, #255]
	b .LBB5_5
.LBB5_4:
	mov w8, wzr
.LBB5_5:
	fmov s3, #4.00000000
	add x9, sp, #380
	cmp x21, #0
	add x1, x9, #4
	movi d2, #0000000000000000
	mov w10, #8388607
	csel x21, xzr, x1, eq
	add x22, x0, x22, lsl #6
	sub x23, x29, #192
	fcmp s0, s3
	str w8, [sp, #380]
	fcsel s1, s3, s0, gt
	fmov w9, s1
	sub w9, w9, #1
	cmp w9, w10
	fcsel s1, s2, s1, lo
	fcmp s0, #0.0
	str q1, [sp, #16]
	b.lt .LBB5_23
	fcmp s0, s3
	b.hi .LBB5_23
	fmov s3, w9
	fmov s1, w10
	mov w9, #64269
	mov w8, #64269
	movk w9, #16330, lsl #16
	fmov v2.4s, #1.00000000
	movk w8, #74, lsl #16
	str s0, [sp, #320]
	cmhi v1.4s, v1.4s, v3.4s
	dup v3.4s, w9
	dup v4.4s, w8
	mov w8, #1267
	mov w9, #27455
	movk w8, #16181, lsl #16
	movk w9, #16094, lsl #16
	str q3, [sp, #96]
	dup v3.4s, w8
	mov w8, #46233
	movk w8, #16147, lsl #16
	and v2.16b, v1.16b, v2.16b
	dup v1.4s, w8
	mov w8, #43579
	stp q3, q4, [sp, #272]
	dup v3.4s, w9
	mov w9, #14483
	movk w9, #16246, lsl #16
	movk w8, #16440, lsl #16
	stp q1, q3, [sp, #240]
	dup v3.4s, w9
	dup v1.4s, w8
	mov w8, #-1023672320
	mov w9, #39322
	movk w9, #16409, lsl #16
	stp q1, q3, [sp, #208]
	dup v3.4s, w8
	mov w8, #1123942400
	dup v1.4s, w8
	mov w8, #63802
	str w9, [sp, #304]
	movk w8, #14625, lsl #16
	stp q3, q1, [sp, #64]
	dup v3.4s, w8
	mov w8, #50297
	movk w8, #15022, lsl #16
	dup v1.4s, w8
	mov w8, #38156
	movk w8, #15389, lsl #16
	dup v4.4s, w8
	mov w8, #22716
	movk w8, #15715, lsl #16
	stp q1, q3, [sp, #176]
	mvni v1.4s, #127, msl #16
	dup v3.4s, w8
	mov w8, #65005
	movk w8, #15989, lsl #16
	stp q3, q4, [sp, #144]
	fneg v4.4s, v1.4s
	dup v1.4s, v2.s[0]
	dup v3.4s, w8
	mov w8, #29208
	movk w8, #16177, lsl #16
	str q1, [sp]
	dup v1.4s, w9
	stp q1, q4, [sp, #32]
	dup v1.4s, w8
	mov w8, #21845
	movk w8, #16085, lsl #16
	stp q1, q3, [sp, #112]
	b .LBB5_9
.LBB5_8:
	fmul v0.4s, v0.4s, v13.s[3]
	fmul v2.4s, v2.4s, v12.s[3]
	fmul v1.4s, v1.4s, v11.s[3]
	fmul v3.4s, v3.4s, v15.s[3]
	mov v0.s[3], v13.s[3]
	mov v2.s[3], v12.s[3]
	mov v1.s[3], v11.s[3]
	mov v3.s[3], v15.s[3]
	stp q0, q2, [x9]
	ldr s0, [sp, #320]
	stp q1, q3, [x9, #32]
.LBB5_9:
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB5_14
	cbz x21, .LBB5_28
	mov x0, xzr
	mov x9, x21
	mov x21, xzr
	ldr s1, [sp, #304]
	ldp q13, q12, [x9]
	ldp q11, q15, [x9, #32]
	fcmp s0, s1
	b.eq .LBB5_15
.LBB5_12:
	fmov s1, w8
	fcmp s0, s1
	b.ne .LBB5_18
	movi v7.4s, #128, lsl #16
	fmov v18.4s, #1.00000000
	movi v17.4s, #127, msl #16
	ldp q21, q6, [sp, #272]
	fmov v22.4s, #-1.00000000
	mvni v29.4s, #126
	fcmge v0.4s, v13.4s, v7.4s
	fcmge v2.4s, v12.4s, v7.4s
	and v0.16b, v0.16b, v13.16b
	and v2.16b, v2.16b, v12.16b
	fminnm v1.4s, v0.4s, v18.4s
	fminnm v30.4s, v2.4s, v18.4s
	add v0.4s, v1.4s, v6.4s
	add v27.4s, v30.4s, v6.4s
	cmeq v1.4s, v1.4s, #0
	and v3.16b, v0.16b, v17.16b
	and v5.16b, v27.16b, v17.16b
	ssra v29.4s, v0.4s, #23
	mvni v0.4s, #126
	add v2.4s, v3.4s, v21.4s
	fcmge v3.4s, v11.4s, v7.4s
	fadd v4.4s, v2.4s, v22.4s
	fadd v2.4s, v2.4s, v18.4s
	and v3.16b, v3.16b, v11.16b
	fdiv v2.4s, v4.4s, v2.4s
	fminnm v14.4s, v3.4s, v18.4s
	fcmge v4.4s, v15.4s, v7.4s
	add v3.4s, v5.4s, v21.4s
	add v7.4s, v14.4s, v6.4s
	and v4.16b, v4.16b, v15.16b
	fadd v5.4s, v3.4s, v22.4s
	fadd v3.4s, v3.4s, v18.4s
	and v16.16b, v7.16b, v17.16b
	fminnm v19.4s, v4.4s, v18.4s
	add v4.4s, v16.4s, v21.4s
	fdiv v3.4s, v5.4s, v3.4s
	add v16.4s, v19.4s, v6.4s
	ldp q9, q6, [sp, #240]
	fadd v5.4s, v4.4s, v22.4s
	fadd v4.4s, v4.4s, v18.4s
	and v20.16b, v16.16b, v17.16b
	ldp q23, q17, [sp, #208]
	mov v31.16b, v9.16b
	ssra v0.4s, v16.4s, #23
	mov v25.16b, v17.16b
	mov v8.16b, v23.16b
	mov v16.16b, v23.16b
	scvtf v0.4s, v0.4s
	fdiv v4.4s, v5.4s, v4.4s
	add v5.4s, v20.4s, v21.4s
	mov v21.16b, v9.16b
	fadd v20.4s, v5.4s, v22.4s
	fadd v5.4s, v5.4s, v18.4s
	fmul v24.4s, v3.4s, v3.4s
	mvni v22.4s, #126
	ssra v22.4s, v27.4s, #23
	fdiv v5.4s, v20.4s, v5.4s
	fmul v20.4s, v2.4s, v2.4s
	scvtf v22.4s, v22.4s
	fmul v26.4s, v4.4s, v4.4s
	fmla v21.4s, v6.4s, v20.4s
	fmla v31.4s, v6.4s, v26.4s
	fmla v25.4s, v20.4s, v21.4s
	mov v21.16b, v9.16b
	fmla v21.4s, v6.4s, v24.4s
	fmla v8.4s, v20.4s, v25.4s
	mov v25.16b, v17.16b
	mvni v20.4s, #126
	fmul v28.4s, v5.4s, v5.4s
	fmla v25.4s, v24.4s, v21.4s
	mov v21.16b, v17.16b
	ssra v20.4s, v7.4s, #23
	scvtf v7.4s, v29.4s
	fmla v9.4s, v6.4s, v28.4s
	fmla v21.4s, v26.4s, v31.4s
	scvtf v20.4s, v20.4s
	fmla v16.4s, v24.4s, v25.4s
	mov v24.16b, v23.16b
	fmla v7.4s, v2.4s, v8.4s
	dup v2.4s, w8
	fmla v17.4s, v28.4s, v9.4s
	fmla v24.4s, v26.4s, v21.4s
	ldr q26, [sp, #160]
	fmla v22.4s, v3.4s, v16.4s
	fmul v3.4s, v7.4s, v2.4s
	mov v25.16b, v26.16b
	fmla v23.4s, v28.4s, v17.4s
	fmla v20.4s, v4.4s, v24.4s
	fmla v0.4s, v5.4s, v23.4s
	fmul v5.4s, v22.4s, v2.4s
	mov v23.16b, v26.16b
	fmul v7.4s, v20.4s, v2.4s
	ldp q22, q6, [sp, #176]
	fmul v16.4s, v0.4s, v2.4s
	frintn v2.4s, v3.4s
	frintn v4.4s, v5.4s
	frintn v21.4s, v7.4s
	fsub v20.4s, v3.4s, v2.4s
	frintn v24.4s, v16.4s
	fsub v0.4s, v5.4s, v4.4s
	mov v5.16b, v22.16b
	fsub v3.4s, v7.4s, v21.4s
	mov v7.16b, v22.16b
	fcvtns v2.4s, v2.4s
	fcvtns v4.4s, v4.4s
	fcvtns v21.4s, v21.4s
	fmla v5.4s, v6.4s, v20.4s
	fsub v17.4s, v16.4s, v24.4s
	mov v16.16b, v22.16b
	fmla v7.4s, v6.4s, v0.4s
	fcvtns v24.4s, v24.4s
	shl v2.4s, v2.4s, #23
	shl v4.4s, v4.4s, #23
	fmla v16.4s, v6.4s, v3.4s
	fmla v23.4s, v20.4s, v5.4s
	mov v5.16b, v26.16b
	fmla v22.4s, v6.4s, v17.4s
	ldr q6, [sp, #144]
	fmla v5.4s, v0.4s, v7.4s
	mov v7.16b, v26.16b
	fmla v25.4s, v3.4s, v16.4s
	mov v16.16b, v6.16b
	mov v26.16b, v6.16b
	fmla v7.4s, v17.4s, v22.4s
	mov v22.16b, v6.16b
	fmla v16.4s, v0.4s, v5.4s
	mov v5.16b, v6.16b
	ldr q6, [sp, #128]
	fmla v26.4s, v20.4s, v23.4s
	mov v23.16b, v6.16b
	fmla v22.4s, v3.4s, v25.4s
	mov v25.16b, v6.16b
	fmla v5.4s, v17.4s, v7.4s
	mov v7.16b, v6.16b
	fmla v23.4s, v20.4s, v26.4s
	mov v26.16b, v6.16b
	ldr q6, [sp, #112]
	fmla v7.4s, v0.4s, v16.4s
	fmla v25.4s, v3.4s, v22.4s
	mov v28.16b, v6.16b
	mov v16.16b, v6.16b
	mov v22.16b, v6.16b
	fmla v26.4s, v17.4s, v5.4s
	mov v5.16b, v6.16b
	fmla v28.4s, v20.4s, v23.4s
	fmov v23.4s, #1.00000000
	fmla v16.4s, v0.4s, v7.4s
	fmov v7.4s, #1.00000000
	fmla v22.4s, v3.4s, v25.4s
	fmov v25.4s, #1.00000000
	fmla v5.4s, v17.4s, v26.4s
	fmov v26.4s, #1.00000000
	fmla v23.4s, v20.4s, v28.4s
	shl v20.4s, v21.4s, #23
	shl v21.4s, v24.4s, #23
	fmla v7.4s, v0.4s, v16.4s
	fmla v25.4s, v3.4s, v22.4s
	add v0.4s, v2.4s, v18.4s
	fmla v26.4s, v17.4s, v5.4s
	add v2.4s, v4.4s, v18.4s
	cmeq v5.4s, v30.4s, #0
	add v3.4s, v20.4s, v18.4s
	add v4.4s, v21.4s, v18.4s
	cmeq v16.4s, v19.4s, #0
	fmul v0.4s, v23.4s, v0.4s
	fmul v2.4s, v7.4s, v2.4s
	cmeq v7.4s, v14.4s, #0
	fmul v3.4s, v25.4s, v3.4s
	fmul v4.4s, v26.4s, v4.4s
	bic v0.16b, v0.16b, v1.16b
	bic v2.16b, v2.16b, v5.16b
	bic v1.16b, v3.16b, v7.16b
	bic v3.16b, v4.16b, v16.16b
	b .LBB5_16
.LBB5_14:
	add x10, x0, #64
	mov x9, x0
	mov x0, x10
	ldr s1, [sp, #304]
	ldp q13, q12, [x9]
	ldp q11, q15, [x9, #32]
	fcmp s0, s1
	b.ne .LBB5_12
.LBB5_15:
	movi v4.4s, #128, lsl #16
	fmov v30.4s, #1.00000000
	ldr q21, [sp, #96]
	ldp q22, q6, [sp, #272]
	movi v5.4s, #127, msl #16
	fmov v23.4s, #-1.00000000
	ldr q19, [sp, #64]
	fcmge v0.4s, v13.4s, v4.4s
	fcmge v1.4s, v12.4s, v4.4s
	fcmge v3.4s, v11.4s, v4.4s
	and v0.16b, v0.16b, v13.16b
	and v1.16b, v1.16b, v12.16b
	and v3.16b, v3.16b, v11.16b
	fminnm v0.4s, v0.4s, v30.4s
	fminnm v1.4s, v1.4s, v30.4s
	fminnm v3.4s, v3.4s, v30.4s
	cmeq v2.4s, v0.4s, #0
	add v0.4s, v0.4s, v6.4s
	cmeq v7.4s, v1.4s, #0
	add v1.4s, v1.4s, v6.4s
	bit v0.16b, v21.16b, v2.16b
	stp q7, q2, [sp, #336]
	mov v8.16b, v7.16b
	bsl v8.16b, v21.16b, v1.16b
	and v2.16b, v0.16b, v5.16b
	add v2.4s, v2.4s, v22.4s
	and v7.16b, v8.16b, v5.16b
	fadd v1.4s, v2.4s, v23.4s
	fadd v2.4s, v2.4s, v30.4s
	add v7.4s, v7.4s, v22.4s
	fdiv v27.4s, v1.4s, v2.4s
	fcmge v2.4s, v15.4s, v4.4s
	cmeq v1.4s, v3.4s, #0
	add v3.4s, v3.4s, v6.4s
	mov v17.16b, v1.16b
	and v2.16b, v2.16b, v15.16b
	bsl v17.16b, v21.16b, v3.16b
	fadd v3.4s, v7.4s, v23.4s
	fadd v7.4s, v7.4s, v30.4s
	fminnm v2.4s, v2.4s, v30.4s
	and v16.16b, v17.16b, v5.16b
	fdiv v20.4s, v3.4s, v7.4s
	cmeq v14.4s, v2.4s, #0
	add v2.4s, v2.4s, v6.4s
	ldp q9, q6, [sp, #240]
	add v3.4s, v16.4s, v22.4s
	mov v7.16b, v14.16b
	mov v24.16b, v9.16b
	mov v29.16b, v9.16b
	bsl v7.16b, v21.16b, v2.16b
	fadd v2.4s, v3.4s, v23.4s
	fadd v3.4s, v3.4s, v30.4s
	mov v21.16b, v9.16b
	and v16.16b, v7.16b, v5.16b
	fdiv v2.4s, v2.4s, v3.4s
	ldp q5, q4, [sp, #208]
	add v3.4s, v16.4s, v22.4s
	fmul v22.4s, v20.4s, v20.4s
	mov v25.16b, v4.16b
	mov v31.16b, v5.16b
	mov v26.16b, v4.16b
	fadd v16.4s, v3.4s, v23.4s
	fadd v3.4s, v3.4s, v30.4s
	mvni v23.4s, #126
	fmla v24.4s, v6.4s, v22.4s
	ssra v23.4s, v0.4s, #23
	fdiv v3.4s, v16.4s, v3.4s
	fmul v16.4s, v27.4s, v27.4s
	fmla v26.4s, v22.4s, v24.4s
	mvni v24.4s, #126
	fmul v0.4s, v2.4s, v2.4s
	fmla v21.4s, v6.4s, v16.4s
	ssra v24.4s, v17.4s, #23
	fmla v29.4s, v6.4s, v0.4s
	fmla v25.4s, v16.4s, v21.4s
	mvni v21.4s, #126
	ssra v21.4s, v8.4s, #23
	scvtf v8.4s, v23.4s
	mov v23.16b, v5.16b
	fmla v31.4s, v16.4s, v25.4s
	mov v16.16b, v9.16b
	mov v25.16b, v4.16b
	fmul v28.4s, v3.4s, v3.4s
	fmla v23.4s, v22.4s, v26.4s
	mvni v22.4s, #126
	scvtf v21.4s, v21.4s
	fmla v25.4s, v0.4s, v29.4s
	fmla v8.4s, v27.4s, v31.4s
	fmla v16.4s, v6.4s, v28.4s
	ssra v22.4s, v7.4s, #23
	mov v7.16b, v5.16b
	ldr q6, [sp, #32]
	fmla v21.4s, v20.4s, v23.4s
	fmul v9.4s, v8.4s, v6.4s
	fmla v7.4s, v0.4s, v25.4s
	scvtf v0.4s, v22.4s
	fmla v4.4s, v28.4s, v16.4s
	scvtf v16.4s, v24.4s
	movi v8.4s, #67, lsl #24
	fmul v17.4s, v21.4s, v6.4s
	ldp q27, q25, [sp, #176]
	fmla v5.4s, v28.4s, v4.4s
	fmaxnm v4.4s, v9.4s, v19.4s
	fmla v16.4s, v2.4s, v7.4s
	fmaxnm v2.4s, v17.4s, v19.4s
	mov v22.16b, v27.16b
	mov v23.16b, v27.16b
	fmla v0.4s, v3.4s, v5.4s
	fminnm v3.4s, v4.4s, v8.4s
	fmul v21.4s, v16.4s, v6.4s
	fminnm v2.4s, v2.4s, v8.4s
	fmul v0.4s, v0.4s, v6.4s
	frintn v4.4s, v3.4s
	fmaxnm v5.4s, v21.4s, v19.4s
	ldr q6, [sp, #80]
	frintn v16.4s, v2.4s
	fmaxnm v7.4s, v0.4s, v19.4s
	fminnm v26.4s, v4.4s, v6.4s
	fminnm v5.4s, v5.4s, v8.4s
	fminnm v29.4s, v16.4s, v6.4s
	fminnm v7.4s, v7.4s, v8.4s
	fsub v4.4s, v3.4s, v26.4s
	frintn v3.4s, v5.4s
	fsub v2.4s, v2.4s, v29.4s
	fcvtns v26.4s, v26.4s
	fcvtns v29.4s, v29.4s
	frintn v16.4s, v7.4s
	fminnm v3.4s, v3.4s, v6.4s
	fmla v22.4s, v25.4s, v4.4s
	fmla v23.4s, v25.4s, v2.4s
	shl v26.4s, v26.4s, #23
	shl v29.4s, v29.4s, #23
	fminnm v20.4s, v16.4s, v6.4s
	ldp q18, q6, [sp, #144]
	fsub v31.4s, v5.4s, v3.4s
	fcvtns v3.4s, v3.4s
	add v26.4s, v26.4s, v30.4s
	add v29.4s, v29.4s, v30.4s
	mov v16.16b, v6.16b
	mov v5.16b, v6.16b
	mov v28.16b, v6.16b
	fsub v24.4s, v7.4s, v20.4s
	mov v7.16b, v18.16b
	fcvtns v20.4s, v20.4s
	shl v3.4s, v3.4s, #23
	fmla v16.4s, v4.4s, v22.4s
	mov v22.16b, v27.16b
	fmla v5.4s, v2.4s, v23.4s
	mov v23.16b, v18.16b
	shl v20.4s, v20.4s, #23
	add v3.4s, v3.4s, v30.4s
	fmla v22.4s, v25.4s, v31.4s
	fmla v7.4s, v4.4s, v16.4s
	mov v16.16b, v27.16b
	ldr q27, [sp, #128]
	fmla v23.4s, v2.4s, v5.4s
	fmov v5.4s, #1.00000000
	add v20.4s, v20.4s, v30.4s
	fmla v16.4s, v25.4s, v24.4s
	mov v25.16b, v27.16b
	fmla v28.4s, v31.4s, v22.4s
	mov v22.16b, v27.16b
	fmla v25.4s, v4.4s, v7.4s
	mov v7.16b, v6.16b
	mov v6.16b, v27.16b
	fmla v22.4s, v2.4s, v23.4s
	mov v23.16b, v18.16b
	fmla v7.4s, v24.4s, v16.4s
	ldr q16, [sp, #112]
	fmla v23.4s, v31.4s, v28.4s
	mov v10.16b, v16.16b
	mov v28.16b, v16.16b
	fmla v18.4s, v24.4s, v7.4s
	fmov v7.4s, #1.00000000
	fmla v10.4s, v4.4s, v25.4s
	fmla v28.4s, v2.4s, v22.4s
	mov v22.16b, v27.16b
	fmov v25.4s, #1.00000000
	fmla v6.4s, v24.4s, v18.4s
	fmla v22.4s, v31.4s, v23.4s
	fmov v23.4s, #1.00000000
	fmla v5.4s, v4.4s, v10.4s
	mov v4.16b, v16.16b
	fmla v7.4s, v2.4s, v28.4s
	fcmge v28.4s, v17.4s, v8.4s
	fcmge v2.4s, v9.4s, v8.4s
	fcmgt v17.4s, v19.4s, v17.4s
	fcmgt v9.4s, v19.4s, v9.4s
	fmla v16.4s, v24.4s, v6.4s
	fmla v4.4s, v31.4s, v22.4s
	fcmge v6.4s, v0.4s, v8.4s
	fcmgt v0.4s, v19.4s, v0.4s
	fcmge v22.4s, v21.4s, v8.4s
	fcmgt v21.4s, v19.4s, v21.4s
	fmul v5.4s, v5.4s, v26.4s
	fmul v7.4s, v7.4s, v29.4s
	fmla v25.4s, v24.4s, v16.4s
	mvn v16.16b, v28.16b
	fmla v23.4s, v31.4s, v4.4s
	mvn v31.16b, v6.16b
	mvn v4.16b, v2.16b
	mvn v24.16b, v22.16b
	bic v16.16b, v16.16b, v17.16b
	fmul v17.4s, v25.4s, v20.4s
	bic v0.16b, v31.16b, v0.16b
	bic v4.16b, v4.16b, v9.16b
	fmul v3.4s, v23.4s, v3.4s
	bic v20.16b, v24.16b, v21.16b
	and v4.16b, v4.16b, v5.16b
	and v5.16b, v16.16b, v7.16b
	and v0.16b, v0.16b, v17.16b
	ldr q17, [sp, #48]
	and v3.16b, v20.16b, v3.16b
	and v2.16b, v2.16b, v17.16b
	and v7.16b, v28.16b, v17.16b
	and v6.16b, v6.16b, v17.16b
	and v16.16b, v22.16b, v17.16b
	orr v2.16b, v4.16b, v2.16b
	orr v4.16b, v5.16b, v7.16b
	orr v5.16b, v0.16b, v6.16b
	ldr q0, [sp, #352]
	orr v3.16b, v3.16b, v16.16b
	bic v0.16b, v2.16b, v0.16b
	ldr q2, [sp, #336]
	bic v1.16b, v3.16b, v1.16b
	bic v3.16b, v5.16b, v14.16b
	bic v2.16b, v4.16b, v2.16b
.LBB5_16:
	fcmge v4.4s, v12.4s, v13.4s
	fcmgt v5.4s, v13.4s, v12.4s
	fcmge v6.4s, v15.4s, v11.4s
	fcmgt v7.4s, v11.4s, v15.4s
	orr v4.16b, v5.16b, v4.16b
	orr v5.16b, v7.16b, v6.16b
	mvn v4.16b, v4.16b
	orn v4.16b, v4.16b, v5.16b
	addp d4, v4.2d
	fmov x10, d4
	cbz x10, .LBB5_8
	fcmeq v4.4s, v13.4s, v13.4s
	fcmeq v5.4s, v15.4s, v15.4s
	fcmeq v6.4s, v11.4s, v11.4s
	fcmeq v7.4s, v12.4s, v12.4s
	bif v0.16b, v13.16b, v4.16b
	bif v1.16b, v11.16b, v6.16b
	bif v3.16b, v15.16b, v5.16b
	bif v2.16b, v12.16b, v7.16b
	b .LBB5_8
.LBB5_18:
	fcmp s0, #0.0
	b.ne .LBB5_20
	fcmgt v0.4s, v13.4s, #0.0
	fcmgt v1.4s, v12.4s, #0.0
	fcmgt v3.4s, v11.4s, #0.0
	fcmgt v4.4s, v15.4s, #0.0
	fmov v5.4s, #1.00000000
	and v0.16b, v0.16b, v5.16b
	and v2.16b, v1.16b, v5.16b
	and v1.16b, v3.16b, v5.16b
	and v3.16b, v4.16b, v5.16b
	b .LBB5_16
.LBB5_20:
	movi v17.4s, #128, lsl #16
	fmov v19.4s, #1.00000000
	ldr q7, [sp, #96]
	ldp q16, q6, [sp, #272]
	movi v21.4s, #127, msl #16
	fmov v22.4s, #-1.00000000
	ldp q25, q23, [sp, #208]
	mvni v29.4s, #126
	ldr q18, [sp, #64]
	movi v27.4s, #67, lsl #24
	fcmge v0.4s, v13.4s, v17.4s
	fcmge v1.4s, v12.4s, v17.4s
	fcmge v3.4s, v11.4s, v17.4s
	mov v9.16b, v23.16b
	and v0.16b, v0.16b, v13.16b
	and v1.16b, v1.16b, v12.16b
	and v3.16b, v3.16b, v11.16b
	fminnm v0.4s, v0.4s, v19.4s
	fminnm v1.4s, v1.4s, v19.4s
	fminnm v3.4s, v3.4s, v19.4s
	cmeq v10.4s, v0.4s, #0
	add v0.4s, v0.4s, v6.4s
	cmeq v4.4s, v1.4s, #0
	add v1.4s, v1.4s, v6.4s
	bit v0.16b, v7.16b, v10.16b
	str q4, [sp, #352]
	bsl v4.16b, v7.16b, v1.16b
	and v2.16b, v0.16b, v21.16b
	ssra v29.4s, v0.4s, #23
	and v5.16b, v4.16b, v21.16b
	add v2.4s, v2.4s, v16.4s
	fadd v1.4s, v2.4s, v22.4s
	fadd v2.4s, v2.4s, v19.4s
	fdiv v30.4s, v1.4s, v2.4s
	fcmge v2.4s, v15.4s, v17.4s
	cmeq v1.4s, v3.4s, #0
	add v3.4s, v3.4s, v6.4s
	add v17.4s, v5.4s, v16.4s
	mov v5.16b, v1.16b
	str q1, [sp, #336]
	and v2.16b, v2.16b, v15.16b
	bsl v5.16b, v7.16b, v3.16b
	fadd v3.4s, v17.4s, v22.4s
	fadd v17.4s, v17.4s, v19.4s
	fminnm v2.4s, v2.4s, v19.4s
	and v20.16b, v5.16b, v21.16b
	fdiv v17.4s, v3.4s, v17.4s
	cmeq v8.4s, v2.4s, #0
	add v2.4s, v2.4s, v6.4s
	add v3.4s, v20.4s, v16.4s
	mov v20.16b, v8.16b
	bsl v20.16b, v7.16b, v2.16b
	fadd v2.4s, v3.4s, v22.4s
	fadd v3.4s, v3.4s, v19.4s
	mov v7.16b, v23.16b
	and v21.16b, v20.16b, v21.16b
	fdiv v14.4s, v2.4s, v3.4s
	fmul v26.4s, v17.4s, v17.4s
	add v3.4s, v21.4s, v16.4s
	mov v16.16b, v25.16b
	fadd v21.4s, v3.4s, v22.4s
	fadd v3.4s, v3.4s, v19.4s
	ldp q22, q6, [sp, #240]
	fdiv v3.4s, v21.4s, v3.4s
	fmul v21.4s, v30.4s, v30.4s
	mov v24.16b, v22.16b
	mov v31.16b, v22.16b
	mov v0.16b, v22.16b
	fmul v2.4s, v14.4s, v14.4s
	fmla v24.4s, v6.4s, v21.4s
	fmla v31.4s, v6.4s, v26.4s
	fmla v0.4s, v6.4s, v2.4s
	fmla v9.4s, v21.4s, v24.4s
	mvni v24.4s, #126
	fmla v7.4s, v26.4s, v31.4s
	mvni v31.4s, #126
	ssra v24.4s, v4.4s, #23
	scvtf v4.4s, v29.4s
	mov v29.16b, v25.16b
	fmla v16.4s, v21.4s, v9.4s
	mov v21.16b, v22.16b
	ssra v31.4s, v5.4s, #23
	fmul v1.4s, v3.4s, v3.4s
	fmla v29.4s, v26.4s, v7.4s
	mov v26.16b, v23.16b
	mvni v7.4s, #126
	scvtf v24.4s, v24.4s
	fmla v4.4s, v30.4s, v16.4s
	mov v16.16b, v23.16b
	fmla v21.4s, v6.4s, v1.4s
	ldr q6, [sp, #16]
	fmla v26.4s, v2.4s, v0.4s
	ssra v7.4s, v20.4s, #23
	mov v0.16b, v25.16b
	fmla v24.4s, v17.4s, v29.4s
	scvtf v17.4s, v31.4s
	fmul v5.4s, v4.4s, v6.s[0]
	mov v4.16b, v25.16b
	fmla v16.4s, v1.4s, v21.4s
	fmla v0.4s, v2.4s, v26.4s
	scvtf v2.4s, v7.4s
	ldp q23, q22, [sp, #176]
	fmul v9.4s, v24.4s, v6.s[0]
	fmla v4.4s, v1.4s, v16.4s
	fmaxnm v1.4s, v5.4s, v18.4s
	fmla v17.4s, v14.4s, v0.4s
	mov v21.16b, v23.16b
	fmla v2.4s, v3.4s, v4.4s
	fmaxnm v3.4s, v9.4s, v18.4s
	fminnm v1.4s, v1.4s, v27.4s
	fmul v0.4s, v17.4s, v6.s[0]
	fmul v17.4s, v2.4s, v6.s[0]
	fminnm v3.4s, v3.4s, v27.4s
	frintn v2.4s, v1.4s
	fmaxnm v4.4s, v0.4s, v18.4s
	ldr q6, [sp, #80]
	fmaxnm v7.4s, v17.4s, v18.4s
	frintn v16.4s, v3.4s
	fminnm v20.4s, v2.4s, v6.4s
	fminnm v4.4s, v4.4s, v27.4s
	fminnm v7.4s, v7.4s, v27.4s
	fminnm v26.4s, v16.4s, v6.4s
	fsub v2.4s, v1.4s, v20.4s
	frintn v1.4s, v4.4s
	fcvtns v20.4s, v20.4s
	frintn v16.4s, v7.4s
	fsub v29.4s, v3.4s, v26.4s
	fmla v21.4s, v22.4s, v2.4s
	fminnm v31.4s, v1.4s, v6.4s
	mov v1.16b, v23.16b
	fcvtns v26.4s, v26.4s
	shl v20.4s, v20.4s, #23
	fminnm v3.4s, v16.4s, v6.4s
	ldp q28, q6, [sp, #144]
	fmla v1.4s, v22.4s, v29.4s
	fsub v24.4s, v4.4s, v31.4s
	shl v26.4s, v26.4s, #23
	add v20.4s, v20.4s, v19.4s
	mov v16.16b, v6.16b
	mov v4.16b, v6.16b
	mov v30.16b, v28.16b
	mov v25.16b, v6.16b
	fmla v16.4s, v2.4s, v21.4s
	fsub v21.4s, v7.4s, v3.4s
	mov v7.16b, v28.16b
	fmla v4.4s, v29.4s, v1.4s
	mov v1.16b, v23.16b
	fcvtns v3.4s, v3.4s
	fmla v7.4s, v2.4s, v16.4s
	mov v16.16b, v23.16b
	fmla v1.4s, v22.4s, v24.4s
	fmla v30.4s, v29.4s, v4.4s
	fmov v4.4s, #1.00000000
	shl v3.4s, v3.4s, #23
	fmla v16.4s, v22.4s, v21.4s
	ldr q22, [sp, #128]
	fmla v25.4s, v24.4s, v1.4s
	add v3.4s, v3.4s, v19.4s
	mov v14.16b, v22.16b
	mov v1.16b, v22.16b
	fmla v14.4s, v2.4s, v7.4s
	mov v7.16b, v6.16b
	ldr q6, [sp, #112]
	fmla v1.4s, v29.4s, v30.4s
	mov v30.16b, v28.16b
	fmla v7.4s, v21.4s, v16.4s
	mov v16.16b, v6.16b
	fmla v30.4s, v24.4s, v25.4s
	mov v25.16b, v6.16b
	fmla v16.4s, v2.4s, v14.4s
	mov v14.16b, v28.16b
	mov v28.16b, v22.16b
	fmla v25.4s, v29.4s, v1.4s
	mov v1.16b, v22.16b
	fmla v14.4s, v21.4s, v7.4s
	fmov v7.4s, #1.00000000
	fmla v1.4s, v24.4s, v30.4s
	fmla v4.4s, v2.4s, v16.4s
	mov v2.16b, v6.16b
	mov v16.16b, v6.16b
	fmov v30.4s, #1.00000000
	ldr q6, [sp, #48]
	fmla v28.4s, v21.4s, v14.4s
	fmla v7.4s, v29.4s, v25.4s
	fcvtns v29.4s, v31.4s
	fmla v2.4s, v24.4s, v1.4s
	fmov v14.4s, #1.00000000
	fcmge v25.4s, v5.4s, v27.4s
	fcmge v31.4s, v9.4s, v27.4s
	fcmge v1.4s, v0.4s, v27.4s
	fcmgt v5.4s, v18.4s, v5.4s
	fcmgt v9.4s, v18.4s, v9.4s
	fcmgt v0.4s, v18.4s, v0.4s
	fmul v4.4s, v4.4s, v20.4s
	fmla v16.4s, v21.4s, v28.4s
	shl v29.4s, v29.4s, #23
	fcmge v28.4s, v17.4s, v27.4s
	fmla v30.4s, v24.4s, v2.4s
	mvn v2.16b, v25.16b
	add v24.4s, v26.4s, v19.4s
	mvn v26.16b, v1.16b
	fcmgt v17.4s, v18.4s, v17.4s
	and v1.16b, v1.16b, v6.16b
	fmla v14.4s, v21.4s, v16.4s
	mvn v16.16b, v31.16b
	add v21.4s, v29.4s, v19.4s
	bic v2.16b, v2.16b, v5.16b
	fmul v5.4s, v7.4s, v24.4s
	bic v0.16b, v26.16b, v0.16b
	mvn v29.16b, v28.16b
	bic v7.16b, v16.16b, v9.16b
	fmul v16.4s, v30.4s, v21.4s
	fmul v3.4s, v14.4s, v3.4s
	and v2.16b, v2.16b, v4.16b
	bic v17.16b, v29.16b, v17.16b
	and v4.16b, v7.16b, v5.16b
	and v5.16b, v25.16b, v6.16b
	and v7.16b, v31.16b, v6.16b
	and v0.16b, v0.16b, v16.16b
	and v16.16b, v28.16b, v6.16b
	and v3.16b, v17.16b, v3.16b
	orr v2.16b, v2.16b, v5.16b
	ldr q5, [sp]
	orr v4.16b, v4.16b, v7.16b
	orr v1.16b, v0.16b, v1.16b
	mov v0.16b, v10.16b
	orr v3.16b, v3.16b, v16.16b
	bsl v0.16b, v5.16b, v2.16b
	ldr q2, [sp, #352]
	bit v3.16b, v5.16b, v8.16b
	bsl v2.16b, v5.16b, v4.16b
	ldr q4, [sp, #336]
	bit v1.16b, v5.16b, v4.16b
	b .LBB5_16
.LBB5_21:
	fmov s8, s0
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	fmov s0, s8
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB5_2
.LBB5_22:
	add sp, sp, #640
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
	b linear_srgb::simd::gamma_to_linear_premultiply_rgba_slice_tier_scalar
.LBB5_23:
	.cfi_restore_state
	fmov s8, s0
	cmp x0, x22
	ccmp x0, #0, #4, ne
	b.ne .LBB5_27
.LBB5_24:
	cbz x21, .LBB5_28
	fmov s0, s8
	mov x24, x1
	mov x25, xzr
	mov x0, x21
	mov x21, xzr
.LBB5_26:
	ldp q1, q3, [x0]
	sub x8, x29, #192
	ldr q2, [x0, #32]
	mov x26, x0
	fmov s8, s0
	stp q1, q3, [x23, #64]
	stp q2, q1, [sp, #320]
	ldr q1, [x0, #48]
	sub x0, x29, #128
	str q3, [sp, #304]
	str q1, [sp, #352]
	stp q2, q1, [x23, #96]
	bl linear_srgb::tf::math::gamma_outside_fit::<16>
	ldp q0, q1, [x23]
	mov x0, x25
	ldp q7, q6, [sp, #320]
	mov x1, x24
	ldp q2, q3, [x23, #32]
	ldr q4, [sp, #304]
	ldr q5, [sp, #352]
	fmul v0.4s, v0.4s, v6.s[3]
	fmul v1.4s, v1.4s, v4.s[3]
	fmul v2.4s, v2.4s, v7.s[3]
	fmul v3.4s, v3.4s, v5.s[3]
	mov v0.s[3], v6.s[3]
	mov v1.s[3], v4.s[3]
	mov v2.s[3], v7.s[3]
	mov v3.s[3], v5.s[3]
	stp q0, q1, [x26]
	stp q2, q3, [x26, #32]
	cmp x25, x22
	ccmp x25, #0, #4, ne
	b.eq .LBB5_24
.LBB5_27:
	mov x24, x1
	add x25, x0, #64
	fmov s0, s8
	b .LBB5_26
.LBB5_28:
	ldr w8, [sp, #380]
	cbz w8, .LBB5_30
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB5_30:
	add sp, sp, #640
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
