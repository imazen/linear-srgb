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
	sub sp, sp, #656
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	mov x21, x0
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB12_26
	cmp w8, #2
	b.ne .LBB12_25
.LBB12_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x19, x1, #0xf
	lsr x20, x1, #4
	add x22, x21, x8, lsl #2
	lsl x23, x19, #2
	stp x23, x22, [sp, #16]
	b.eq .LBB12_4
	mov w8, #64
	mov w1, wzr
	fmov s8, s0
	sub x2, x8, x23
	add x8, sp, #412
	add x0, x8, x23
	bl memset
	add x0, sp, #412
	mov x1, x22
	mov x2, x23
	bl memcpy
	add x8, sp, #157
	fmov s0, s8
	add x9, sp, #141
	ldur q1, [x8, #255]
	add x8, sp, #173
	ldur q2, [x8, #255]
	add x8, sp, #93
	stur q1, [x8, #255]
	add x8, sp, #189
	ldur q1, [x8, #255]
	add x8, sp, #205
	ldur q3, [x8, #255]
	add x8, sp, #109
	stur q2, [x8, #255]
	add x8, sp, #125
	stur q1, [x8, #255]
	mov w8, #1
	stur q3, [x9, #255]
	b .LBB12_5
.LBB12_4:
	mov w8, wzr
.LBB12_5:
	fmov s1, #1.00000000
	mov w10, #39322
	str w8, [sp, #344]
	movk w10, #16409, lsl #16
	add x8, sp, #344
	cmp x19, #0
	dup v2.4s, w10
	mov w11, #931135488
	add x8, x8, #4
	fdiv s10, s1, s0
	movi d1, #0000000000000000
	fmov s0, #4.00000000
	csel x27, xzr, x8, eq
	mov w12, #64269
	mov w13, #1267
	str q2, [sp, #64]
	dup v2.4s, w11
	fmov v11.4s, #1.00000000
	movk w12, #74, lsl #16
	movk w13, #16181, lsl #16
	fmov s8, w10
	mov w14, #27455
	mov w15, #46233
	str x8, [sp, #8]
	str q2, [sp, #192]
	dup v2.4s, w12
	movk w14, #16094, lsl #16
	movk w15, #16147, lsl #16
	mov w16, #14483
	mov w17, #43579
	movk w16, #16246, lsl #16
	movk w17, #16440, lsl #16
	mov w9, #8388607
	add x24, x21, x20, lsl #6
	mov w22, #29208
	mov w20, #21845
	sub x23, x29, #240
	movk w22, #16177, lsl #16
	sub x25, x29, #81
	sub x26, x29, #240
	sub x28, x29, #244
	movk w20, #16085, lsl #16
	fcmp s10, #0.0
	fcmeq v1.4s, v10.4s, v1.4s
	fmov w8, s10
	str q10, [sp, #176]
	cset w10, lt
	fcmp s10, s0
	dup v0.4s, w13
	ubfx w11, w8, #23, #8
	and w8, w8, #0x7fffffff
	sub w8, w8, #1
	stp q0, q2, [sp, #144]
	dup v2.4s, w14
	and v0.16b, v1.16b, v11.16b
	dup v1.4s, w15
	csinc w10, w10, wzr, ls
	cmp w11, #254
	ccmp w8, w9, #0, ls
	dup v0.4s, v0.s[0]
	csinc w8, w10, wzr, hs
	stp q1, q2, [sp, #112]
	dup v2.4s, w16
	dup v1.4s, w17
	str w8, [sp, #60]
	str q0, [sp, #32]
	stp q1, q2, [sp, #80]
	b .LBB12_8
.LBB12_6:
	sub x0, x29, #176
	sub x1, x29, #112
	stp x25, x26, [x29, #-112]
	stur x28, [x29, #-96]
	stp q14, q13, [sp, #224]
	str q15, [sp, #208]
	bl linear_srgb::tf::math::gamma_x16::<archmage::tokens::generated::arm::NeonToken>::{closure#0}
	ldp q15, q14, [sp, #208]
	fmov v11.4s, #1.00000000
	ldp q1, q0, [x23, #64]
	ldr q13, [sp, #240]
	ldp q2, q3, [x23, #96]
	ldr q10, [sp, #176]
.LBB12_7:
	ldp q16, q20, [sp, #304]
	fcmge v4.4s, v13.4s, v11.4s
	fcmge v6.4s, v14.4s, v11.4s
	fcmge v7.4s, v15.4s, v11.4s
	ldr q18, [sp, #288]
	ldp q17, q19, [sp, #256]
	fcmge v5.4s, v20.4s, v11.4s
	bic v16.16b, v16.16b, v4.16b
	and v4.16b, v4.16b, v11.16b
	bic v18.16b, v18.16b, v6.16b
	bic v19.16b, v19.16b, v7.16b
	and v6.16b, v6.16b, v11.16b
	and v7.16b, v7.16b, v11.16b
	bic v17.16b, v17.16b, v5.16b
	and v1.16b, v16.16b, v1.16b
	and v5.16b, v5.16b, v11.16b
	and v0.16b, v18.16b, v0.16b
	and v2.16b, v19.16b, v2.16b
	fcmeq v16.4s, v13.4s, v13.4s
	and v3.16b, v17.16b, v3.16b
	orr v1.16b, v1.16b, v4.16b
	fcmeq v4.4s, v14.4s, v14.4s
	orr v0.16b, v0.16b, v6.16b
	fcmeq v6.4s, v15.4s, v15.4s
	orr v2.16b, v2.16b, v7.16b
	fcmeq v7.4s, v20.4s, v20.4s
	orr v3.16b, v3.16b, v5.16b
	bif v1.16b, v13.16b, v16.16b
	bif v0.16b, v14.16b, v4.16b
	bif v2.16b, v15.16b, v6.16b
	bif v3.16b, v20.16b, v7.16b
	stp q1, q0, [x19]
	stp q2, q3, [x19, #32]
.LBB12_8:
	cmp x21, x24
	ccmp x21, #0, #4, ne
	b.ne .LBB12_11
	cbz x27, .LBB12_22
	mov x21, xzr
	mov x19, x27
	mov x27, xzr
	b .LBB12_12
.LBB12_11:
	add x8, x21, #64
	mov x19, x21
	mov x21, x8
.LBB12_12:
	ldp q13, q14, [x19]
	fcmp s10, s8
	ldp q15, q5, [x19, #32]
	stur s10, [x29, #-244]
	fcmgt v4.4s, v13.4s, #0.0
	fcmgt v0.4s, v11.4s, v13.4s
	fcmgt v6.4s, v14.4s, #0.0
	fcmgt v1.4s, v11.4s, v14.4s
	fcmgt v7.4s, v15.4s, #0.0
	fcmgt v2.4s, v11.4s, v15.4s
	fcmgt v16.4s, v5.4s, #0.0
	fcmgt v3.4s, v11.4s, v5.4s
	str q5, [sp, #320]
	stp q6, q4, [sp, #288]
	and v0.16b, v0.16b, v4.16b
	and v1.16b, v1.16b, v6.16b
	and v4.16b, v2.16b, v7.16b
	and v3.16b, v3.16b, v16.16b
	stp q16, q7, [sp, #256]
	mov v7.16b, v0.16b
	mov v2.16b, v1.16b
	mov v1.16b, v4.16b
	mov v0.16b, v3.16b
	bsl v7.16b, v13.16b, v11.16b
	bsl v2.16b, v14.16b, v11.16b
	bsl v1.16b, v15.16b, v11.16b
	bsl v0.16b, v5.16b, v11.16b
	stp q7, q2, [x23]
	stp q1, q0, [x23, #32]
	b.ne .LBB12_15
	fcmge v3.4s, v11.4s, v7.4s
	fcmge v4.4s, v11.4s, v2.4s
	ldr q19, [sp, #192]
	fcmge v5.4s, v11.4s, v1.4s
	fcmge v6.4s, v11.4s, v0.4s
	cmeq v20.4s, v7.4s, #0
	fcmge v16.4s, v7.4s, v19.4s
	fcmge v17.4s, v2.4s, v19.4s
	fcmge v18.4s, v1.4s, v19.4s
	fcmge v19.4s, v0.4s, v19.4s
	cmeq v21.4s, v2.4s, #0
	cmeq v22.4s, v1.4s, #0
	mvn v3.16b, v3.16b
	mvn v4.16b, v4.16b
	cmeq v23.4s, v0.4s, #0
	mvn v5.16b, v5.16b
	mvn v6.16b, v6.16b
	orn v3.16b, v3.16b, v16.16b
	orn v4.16b, v4.16b, v17.16b
	orn v5.16b, v5.16b, v18.16b
	orn v6.16b, v6.16b, v19.16b
	bic v3.16b, v3.16b, v20.16b
	bic v4.16b, v4.16b, v21.16b
	bic v5.16b, v5.16b, v22.16b
	bic v6.16b, v6.16b, v23.16b
	orr v3.16b, v4.16b, v3.16b
	orr v4.16b, v5.16b, v6.16b
	orr v3.16b, v3.16b, v4.16b
	xtn v3.4h, v3.4s
	shl v3.4h, v3.4h, #15
	cmlt v3.4h, v3.4h, #0
	fmov x8, d3
	cbnz x8, .LBB12_6
	cmtst v4.4s, v2.4s, v2.4s
	cmtst v16.4s, v1.4s, v1.4s
	cmtst v3.4s, v7.4s, v7.4s
	cmtst v17.4s, v0.4s, v0.4s
	xtn v5.4h, v4.4s
	xtn v4.4h, v16.4s
	ldr q16, [sp, #64]
	xtn v6.4h, v3.4s
	xtn v3.4h, v17.4s
	b .LBB12_18
.LBB12_15:
	fmov s3, w20
	fcmp s10, s3
	b.ne .LBB12_19
	fcmge v3.4s, v11.4s, v7.4s
	fcmge v4.4s, v11.4s, v2.4s
	ldr q19, [sp, #192]
	fcmge v5.4s, v11.4s, v1.4s
	fcmge v6.4s, v11.4s, v0.4s
	cmeq v20.4s, v7.4s, #0
	fcmge v16.4s, v7.4s, v19.4s
	fcmge v17.4s, v2.4s, v19.4s
	fcmge v18.4s, v1.4s, v19.4s
	fcmge v19.4s, v0.4s, v19.4s
	cmeq v21.4s, v2.4s, #0
	cmeq v22.4s, v1.4s, #0
	mvn v3.16b, v3.16b
	mvn v4.16b, v4.16b
	cmeq v23.4s, v0.4s, #0
	mvn v5.16b, v5.16b
	mvn v6.16b, v6.16b
	orn v3.16b, v3.16b, v16.16b
	orn v4.16b, v4.16b, v17.16b
	orn v5.16b, v5.16b, v18.16b
	orn v6.16b, v6.16b, v19.16b
	bic v3.16b, v3.16b, v20.16b
	bic v4.16b, v4.16b, v21.16b
	bic v5.16b, v5.16b, v22.16b
	bic v6.16b, v6.16b, v23.16b
	orr v3.16b, v4.16b, v3.16b
	orr v4.16b, v5.16b, v6.16b
	orr v3.16b, v3.16b, v4.16b
	xtn v3.4h, v3.4s
	shl v3.4h, v3.4h, #15
	cmlt v3.4h, v3.4h, #0
	fmov x8, d3
	cbnz x8, .LBB12_6
	cmtst v3.4s, v7.4s, v7.4s
	cmtst v4.4s, v2.4s, v2.4s
	cmtst v16.4s, v1.4s, v1.4s
	cmtst v17.4s, v0.4s, v0.4s
	xtn v6.4h, v3.4s
	xtn v5.4h, v4.4s
	xtn v4.4h, v16.4s
	xtn v3.4h, v17.4s
	dup v16.4s, w20
.LBB12_18:
	ldp q24, q23, [sp, #144]
	movi v21.4s, #127, msl #16
	fmov v22.4s, #-1.00000000
	ldp q29, q31, [sp, #112]
	ldr q12, [sp, #96]
	mvni v9.4s, #126
	mvni v28.4s, #126
	add v7.4s, v7.4s, v23.4s
	add v2.4s, v2.4s, v23.4s
	add v1.4s, v1.4s, v23.4s
	add v0.4s, v0.4s, v23.4s
	mvni v23.4s, #126
	mov v26.16b, v29.16b
	mov v27.16b, v29.16b
	mvni v30.4s, #126
	mov w8, #50297
	and v17.16b, v7.16b, v21.16b
	and v19.16b, v2.16b, v21.16b
	movk w8, #15022, lsl #16
	and v20.16b, v1.16b, v21.16b
	and v21.16b, v0.16b, v21.16b
	ssra v23.4s, v7.4s, #23
	mov v7.16b, v29.16b
	ssra v9.4s, v0.4s, #23
	ssra v28.4s, v2.4s, #23
	add v17.4s, v17.4s, v24.4s
	mov v2.16b, v12.16b
	ssra v30.4s, v1.4s, #23
	mov v1.16b, v12.16b
	ushll v6.4s, v6.4h, #0
	ushll v5.4s, v5.4h, #0
	ushll v4.4s, v4.4h, #0
	ushll v3.4s, v3.4h, #0
	fadd v18.4s, v17.4s, v22.4s
	fadd v17.4s, v17.4s, v11.4s
	shl v6.4s, v6.4s, #31
	shl v5.4s, v5.4s, #31
	shl v4.4s, v4.4s, #31
	shl v3.4s, v3.4s, #31
	fdiv v17.4s, v18.4s, v17.4s
	add v18.4s, v19.4s, v24.4s
	cmlt v6.4s, v6.4s, #0
	cmlt v5.4s, v5.4s, #0
	cmlt v4.4s, v4.4s, #0
	cmlt v3.4s, v3.4s, #0
	fadd v19.4s, v18.4s, v22.4s
	fadd v18.4s, v18.4s, v11.4s
	fdiv v18.4s, v19.4s, v18.4s
	add v19.4s, v20.4s, v24.4s
	fadd v20.4s, v19.4s, v22.4s
	fadd v19.4s, v19.4s, v11.4s
	fdiv v19.4s, v20.4s, v19.4s
	add v20.4s, v21.4s, v24.4s
	fadd v21.4s, v20.4s, v22.4s
	fadd v20.4s, v20.4s, v11.4s
	fmul v22.4s, v18.4s, v18.4s
	fmla v27.4s, v31.4s, v22.4s
	fdiv v20.4s, v21.4s, v20.4s
	fmul v21.4s, v17.4s, v17.4s
	fmul v24.4s, v19.4s, v19.4s
	fmla v26.4s, v31.4s, v21.4s
	fmla v7.4s, v31.4s, v24.4s
	fmla v2.4s, v24.4s, v7.4s
	scvtf v7.4s, v23.4s
	fmul v25.4s, v20.4s, v20.4s
	fmla v29.4s, v31.4s, v25.4s
	mov v31.16b, v12.16b
	fmla v31.4s, v21.4s, v26.4s
	mov v26.16b, v12.16b
	fmla v1.4s, v25.4s, v29.4s
	fmla v26.4s, v22.4s, v27.4s
	ldr q27, [sp, #80]
	mov v0.16b, v27.16b
	mov v23.16b, v27.16b
	fmla v0.4s, v21.4s, v31.4s
	mov v21.16b, v27.16b
	fmla v23.4s, v24.4s, v2.4s
	scvtf v2.4s, v9.4s
	fmla v21.4s, v22.4s, v26.4s
	mov v22.16b, v27.16b
	scvtf v26.4s, v28.4s
	scvtf v27.4s, v30.4s
	fmla v7.4s, v17.4s, v0.4s
	fmla v22.4s, v25.4s, v1.4s
	fmla v26.4s, v18.4s, v21.4s
	dup v21.4s, w8
	mov w8, #63802
	fmla v27.4s, v19.4s, v23.4s
	fmul v0.4s, v7.4s, v16.4s
	movk w8, #14625, lsl #16
	fmla v2.4s, v20.4s, v22.4s
	dup v22.4s, w8
	mov w8, #38156
	fmul v1.4s, v26.4s, v16.4s
	mov v23.16b, v21.16b
	movk w8, #15389, lsl #16
	fmul v7.4s, v27.4s, v16.4s
	mov v24.16b, v21.16b
	mov v25.16b, v21.16b
	fmul v20.4s, v2.4s, v16.4s
	frintn v16.4s, v0.4s
	frintn v17.4s, v1.4s
	frintn v18.4s, v7.4s
	frintn v19.4s, v20.4s
	fsub v0.4s, v0.4s, v16.4s
	fcvtns v16.4s, v16.4s
	fsub v2.4s, v1.4s, v17.4s
	fcvtns v17.4s, v17.4s
	fsub v1.4s, v7.4s, v18.4s
	fcvtns v18.4s, v18.4s
	fsub v7.4s, v20.4s, v19.4s
	dup v20.4s, w8
	fmla v23.4s, v22.4s, v0.4s
	mov w8, #22716
	fmla v24.4s, v22.4s, v2.4s
	fcvtns v19.4s, v19.4s
	movk w8, #15715, lsl #16
	fmla v25.4s, v22.4s, v1.4s
	shl v16.4s, v16.4s, #23
	mov v26.16b, v20.16b
	dup v27.4s, w8
	mov v28.16b, v20.16b
	fmla v21.4s, v22.4s, v7.4s
	mov w8, #65005
	shl v17.4s, v17.4s, #23
	movk w8, #15989, lsl #16
	shl v18.4s, v18.4s, #23
	shl v19.4s, v19.4s, #23
	fmla v26.4s, v0.4s, v23.4s
	mov v23.16b, v20.16b
	fmla v28.4s, v1.4s, v25.4s
	dup v22.4s, w8
	mov v25.16b, v27.16b
	fmla v20.4s, v7.4s, v21.4s
	fmov v21.4s, #1.00000000
	fmla v23.4s, v2.4s, v24.4s
	mov v24.16b, v27.16b
	mov v29.16b, v22.16b
	fmla v24.4s, v0.4s, v26.4s
	mov v26.16b, v27.16b
	fmla v27.4s, v7.4s, v20.4s
	dup v20.4s, w22
	fmla v25.4s, v2.4s, v23.4s
	fmov v23.4s, #1.00000000
	fmla v26.4s, v1.4s, v28.4s
	mov v28.16b, v22.16b
	fmla v28.4s, v0.4s, v24.4s
	mov v24.16b, v22.16b
	fmla v22.4s, v7.4s, v27.4s
	mov v27.16b, v20.16b
	fmla v29.4s, v1.4s, v26.4s
	fmov v26.4s, #1.00000000
	fmla v24.4s, v2.4s, v25.4s
	fmov v25.4s, #1.00000000
	fmla v27.4s, v0.4s, v28.4s
	mov v28.16b, v20.16b
	fmla v28.4s, v2.4s, v24.4s
	mov v24.16b, v20.16b
	fmla v20.4s, v7.4s, v22.4s
	fmla v21.4s, v0.4s, v27.4s
	add v0.4s, v16.4s, v11.4s
	fmla v24.4s, v1.4s, v29.4s
	fmla v23.4s, v2.4s, v28.4s
	fmla v26.4s, v7.4s, v20.4s
	add v2.4s, v18.4s, v11.4s
	add v7.4s, v19.4s, v11.4s
	fmul v0.4s, v21.4s, v0.4s
	fmla v25.4s, v1.4s, v24.4s
	add v1.4s, v17.4s, v11.4s
	fmul v7.4s, v26.4s, v7.4s
	fmul v16.4s, v23.4s, v1.4s
	and v1.16b, v6.16b, v0.16b
	fmul v2.4s, v25.4s, v2.4s
	and v3.16b, v3.16b, v7.16b
	and v0.16b, v5.16b, v16.16b
	and v2.16b, v4.16b, v2.16b
	b .LBB12_7
.LBB12_19:
	ldr w8, [sp, #60]
	tbnz w8, #0, .LBB12_6
	fcmge v3.4s, v11.4s, v7.4s
	fcmge v4.4s, v11.4s, v2.4s
	ldr q19, [sp, #192]
	fcmge v5.4s, v11.4s, v1.4s
	fcmge v6.4s, v11.4s, v0.4s
	fcmge v16.4s, v7.4s, v19.4s
	fcmge v17.4s, v2.4s, v19.4s
	fcmge v18.4s, v1.4s, v19.4s
	fcmge v19.4s, v0.4s, v19.4s
	mvn v20.16b, v3.16b
	mvn v21.16b, v4.16b
	cmeq v3.4s, v7.4s, #0
	mvn v22.16b, v5.16b
	mvn v23.16b, v6.16b
	cmeq v4.4s, v2.4s, #0
	cmeq v5.4s, v1.4s, #0
	cmeq v6.4s, v0.4s, #0
	orn v16.16b, v20.16b, v16.16b
	orn v17.16b, v21.16b, v17.16b
	orn v18.16b, v22.16b, v18.16b
	orn v19.16b, v23.16b, v19.16b
	bic v16.16b, v16.16b, v3.16b
	bic v17.16b, v17.16b, v4.16b
	bic v18.16b, v18.16b, v5.16b
	bic v19.16b, v19.16b, v6.16b
	orr v16.16b, v17.16b, v16.16b
	orr v17.16b, v18.16b, v19.16b
	orr v16.16b, v16.16b, v17.16b
	xtn v16.4h, v16.4s
	shl v16.4h, v16.4h, #15
	cmlt v16.4h, v16.4h, #0
	fmov x8, d16
	cbnz x8, .LBB12_6
	ldp q23, q22, [sp, #144]
	movi v20.4s, #127, msl #16
	fmov v21.4s, #-1.00000000
	ldp q30, q31, [sp, #112]
	mvni v28.4s, #126
	mvni v26.4s, #126
	ldr q9, [sp, #96]
	add v7.4s, v7.4s, v22.4s
	add v2.4s, v2.4s, v22.4s
	add v1.4s, v1.4s, v22.4s
	add v0.4s, v0.4s, v22.4s
	mov v29.16b, v30.16b
	mov v24.16b, v30.16b
	mov v25.16b, v30.16b
	mvni v27.4s, #126
	mov w8, #50297
	and v16.16b, v7.16b, v20.16b
	and v18.16b, v2.16b, v20.16b
	movk w8, #15022, lsl #16
	and v19.16b, v1.16b, v20.16b
	and v20.16b, v0.16b, v20.16b
	ssra v28.4s, v1.4s, #23
	mov v1.16b, v9.16b
	ssra v26.4s, v7.4s, #23
	mov v7.16b, v9.16b
	add v16.4s, v16.4s, v23.4s
	ssra v27.4s, v2.4s, #23
	mvni v2.4s, #126
	fadd v17.4s, v16.4s, v21.4s
	fadd v16.4s, v16.4s, v11.4s
	ssra v2.4s, v0.4s, #23
	scvtf v2.4s, v2.4s
	fdiv v16.4s, v17.4s, v16.4s
	add v17.4s, v18.4s, v23.4s
	fadd v18.4s, v17.4s, v21.4s
	fadd v17.4s, v17.4s, v11.4s
	fdiv v17.4s, v18.4s, v17.4s
	add v18.4s, v19.4s, v23.4s
	fadd v19.4s, v18.4s, v21.4s
	fadd v18.4s, v18.4s, v11.4s
	fdiv v18.4s, v19.4s, v18.4s
	add v19.4s, v20.4s, v23.4s
	fadd v20.4s, v19.4s, v21.4s
	fadd v19.4s, v19.4s, v11.4s
	fmul v21.4s, v17.4s, v17.4s
	fmla v25.4s, v31.4s, v21.4s
	fdiv v19.4s, v20.4s, v19.4s
	fmul v20.4s, v16.4s, v16.4s
	fmla v7.4s, v21.4s, v25.4s
	fmul v22.4s, v18.4s, v18.4s
	fmla v24.4s, v31.4s, v20.4s
	fmla v29.4s, v31.4s, v22.4s
	fmla v1.4s, v22.4s, v29.4s
	ldr q29, [sp, #80]
	fmul v23.4s, v19.4s, v19.4s
	mov v0.16b, v29.16b
	mov v25.16b, v29.16b
	fmla v0.4s, v21.4s, v7.4s
	mov v7.16b, v29.16b
	mov v21.16b, v29.16b
	fmla v30.4s, v31.4s, v23.4s
	mov v31.16b, v9.16b
	fmla v7.4s, v22.4s, v1.4s
	fmla v31.4s, v20.4s, v24.4s
	mov v24.16b, v9.16b
	fmla v24.4s, v23.4s, v30.4s
	fmla v25.4s, v20.4s, v31.4s
	scvtf v20.4s, v26.4s
	scvtf v26.4s, v27.4s
	scvtf v27.4s, v28.4s
	fmla v21.4s, v23.4s, v24.4s
	fmla v20.4s, v16.4s, v25.4s
	fmla v26.4s, v17.4s, v0.4s
	fmla v27.4s, v18.4s, v7.4s
	fmla v2.4s, v19.4s, v21.4s
	dup v21.4s, w8
	mov w8, #63802
	fmul v0.4s, v20.4s, v10.s[0]
	fmul v1.4s, v26.4s, v10.s[0]
	movk w8, #14625, lsl #16
	fmul v20.4s, v27.4s, v10.s[0]
	dup v22.4s, w8
	mov w8, #38156
	movk w8, #15389, lsl #16
	mov v24.16b, v21.16b
	mov v25.16b, v21.16b
	fmul v2.4s, v2.4s, v10.s[0]
	dup v23.4s, w8
	mov w8, #22716
	frintn v16.4s, v0.4s
	frintn v17.4s, v1.4s
	movk w8, #15715, lsl #16
	frintn v18.4s, v20.4s
	dup v26.4s, w8
	mov w8, #65005
	mov v27.16b, v23.16b
	movk w8, #15989, lsl #16
	frintn v19.4s, v2.4s
	fsub v0.4s, v0.4s, v16.4s
	fsub v7.4s, v1.4s, v17.4s
	fcvtns v16.4s, v16.4s
	fsub v1.4s, v20.4s, v18.4s
	mov v20.16b, v21.16b
	fcvtns v17.4s, v17.4s
	fcvtns v18.4s, v18.4s
	fsub v2.4s, v2.4s, v19.4s
	fcvtns v19.4s, v19.4s
	fmla v20.4s, v22.4s, v0.4s
	fmla v24.4s, v22.4s, v7.4s
	shl v16.4s, v16.4s, #23
	fmla v25.4s, v22.4s, v1.4s
	shl v17.4s, v17.4s, #23
	shl v18.4s, v18.4s, #23
	fmla v21.4s, v22.4s, v2.4s
	mov v22.16b, v23.16b
	shl v19.4s, v19.4s, #23
	fmla v27.4s, v7.4s, v24.4s
	dup v24.4s, w8
	fmla v22.4s, v0.4s, v20.4s
	mov v20.16b, v23.16b
	fmla v23.4s, v2.4s, v21.4s
	mov v21.16b, v26.16b
	mov v28.16b, v24.16b
	fmla v20.4s, v1.4s, v25.4s
	mov v25.16b, v26.16b
	fmla v21.4s, v0.4s, v22.4s
	mov v22.16b, v26.16b
	fmla v26.4s, v2.4s, v23.4s
	mov v23.16b, v24.16b
	fmla v25.4s, v7.4s, v27.4s
	fmov v27.4s, #1.00000000
	fmla v22.4s, v1.4s, v20.4s
	dup v20.4s, w22
	fmla v23.4s, v0.4s, v21.4s
	fmov v21.4s, #1.00000000
	fmla v28.4s, v7.4s, v25.4s
	mov v25.16b, v24.16b
	fmla v24.4s, v2.4s, v26.4s
	mov v26.16b, v20.16b
	fmla v25.4s, v1.4s, v22.4s
	mov v22.16b, v20.16b
	fmla v26.4s, v7.4s, v28.4s
	mov v28.16b, v20.16b
	fmla v20.4s, v2.4s, v24.4s
	fmla v22.4s, v0.4s, v23.4s
	fmov v23.4s, #1.00000000
	fmla v28.4s, v1.4s, v25.4s
	fmov v25.4s, #1.00000000
	fmla v21.4s, v7.4s, v26.4s
	add v7.4s, v19.4s, v11.4s
	fmla v27.4s, v0.4s, v22.4s
	add v0.4s, v16.4s, v11.4s
	fmla v23.4s, v1.4s, v28.4s
	add v1.4s, v17.4s, v11.4s
	fmla v25.4s, v2.4s, v20.4s
	add v2.4s, v18.4s, v11.4s
	ldr q17, [sp, #32]
	fmul v0.4s, v27.4s, v0.4s
	fmul v16.4s, v21.4s, v1.4s
	mov v1.16b, v3.16b
	fmul v2.4s, v23.4s, v2.4s
	fmul v7.4s, v25.4s, v7.4s
	mov v3.16b, v6.16b
	bsl v1.16b, v17.16b, v0.16b
	mov v0.16b, v4.16b
	bit v2.16b, v17.16b, v5.16b
	bsl v3.16b, v17.16b, v7.16b
	bsl v0.16b, v17.16b, v16.16b
	b .LBB12_7
.LBB12_22:
	ldr w8, [sp, #344]
	cbz w8, .LBB12_24
	ldp x2, x0, [sp, #16]
	ldr x1, [sp, #8]
	bl memcpy
.LBB12_24:
	add sp, sp, #656
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
.LBB12_25:
	.cfi_restore_state
	fmov s8, s0
	mov x19, x1
	bl archmage::tokens::generated::arm::neon_detect
	fmov s0, s8
	mov x1, x19
	cbnz w0, .LBB12_2
.LBB12_26:
	mov x0, x21
	add sp, sp, #656
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
	b linear_srgb::simd::linear_to_gamma_slice_tier_scalar_t
