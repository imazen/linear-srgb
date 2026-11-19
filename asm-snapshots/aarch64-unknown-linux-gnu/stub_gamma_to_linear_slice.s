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
	sub sp, sp, #672
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	fmov s10, s0
	mov x22, x0
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB11_26
	cmp w8, #2
	str q10, [sp, #192]
	b.ne .LBB11_25
.LBB11_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x19, x1, #0xf
	lsr x20, x1, #4
	add x21, x22, x8, lsl #2
	lsl x23, x19, #2
	stp x23, x21, [sp, #16]
	b.eq .LBB11_4
	mov w8, #64
	mov w1, wzr
	sub x2, x8, x23
	add x8, sp, #428
	add x0, x8, x23
	bl memset
	add x0, sp, #428
	mov x1, x21
	mov x2, x23
	bl memcpy
	add x8, sp, #173
	ldr q10, [sp, #192]
	add x9, sp, #157
	ldur q0, [x8, #255]
	add x8, sp, #189
	ldur q1, [x8, #255]
	add x8, sp, #109
	stur q0, [x8, #255]
	add x8, sp, #205
	ldur q0, [x8, #255]
	add x8, sp, #221
	ldur q2, [x8, #255]
	add x8, sp, #125
	stur q1, [x8, #255]
	add x8, sp, #141
	stur q0, [x8, #255]
	mov w8, #1
	stur q2, [x9, #255]
	b .LBB11_5
.LBB11_4:
	mov w8, wzr
.LBB11_5:
	movi d0, #0000000000000000
	str w8, [sp, #360]
	add x8, sp, #360
	cmp x19, #0
	add x10, x8, #4
	fmov s1, #4.00000000
	csel x25, xzr, x10, eq
	fcmp s10, #0.0
	fmov w9, s10
	fmov v11.4s, #1.00000000
	str x10, [sp, #8]
	mov w10, #39322
	fcmeq v0.4s, v10.4s, v0.4s
	movk w10, #16409, lsl #16
	mov w11, #931135488
	cset w13, lt
	fcmp s10, s1
	and w12, w9, #0x7fffffff
	ubfx w9, w9, #23, #8
	dup v2.4s, w10
	dup v1.4s, w11
	mov w8, #8388607
	mov w14, #64269
	fmov s8, w10
	and v0.16b, v0.16b, v11.16b
	sub w10, w12, #1
	csinc w11, w13, wzr, ls
	cmp w9, #254
	mov w9, #1267
	movk w14, #74, lsl #16
	movk w9, #16181, lsl #16
	ccmp w10, w8, #0, ls
	str q2, [sp, #64]
	str q1, [sp, #208]
	dup v2.4s, w14
	dup v1.4s, w9
	csinc w9, w11, wzr, hs
	dup v0.4s, v0.s[0]
	mov w8, #27455
	str w9, [sp, #60]
	mov w9, #46233
	movk w8, #16094, lsl #16
	movk w9, #16147, lsl #16
	stp q1, q2, [sp, #160]
	dup v1.4s, w8
	str q0, [sp, #32]
	dup v0.4s, w9
	mov w8, #14483
	mov w9, #43579
	movk w8, #16246, lsl #16
	mov w10, #63802
	movk w9, #16440, lsl #16
	movk w10, #14625, lsl #16
	add x24, x22, x20, lsl #6
	stp q0, q1, [sp, #128]
	dup v1.4s, w8
	dup v0.4s, w9
	mov w21, #29208
	mov w20, #21845
	sub x23, x29, #240
	movk w21, #16177, lsl #16
	sub x26, x29, #81
	sub x27, x29, #240
	stp q0, q1, [sp, #96]
	dup v0.4s, w10
	sub x28, x29, #244
	movk w20, #16085, lsl #16
	str q0, [sp, #80]
	b .LBB11_8
.LBB11_6:
	sub x0, x29, #176
	sub x1, x29, #112
	stp x26, x27, [x29, #-112]
	stur x28, [x29, #-96]
	stp q15, q14, [sp, #224]
	bl linear_srgb::tf::math::gamma_x16::<archmage::tokens::generated::arm::NeonToken>::{closure#0}
	ldp q15, q14, [sp, #224]
	fmov v11.4s, #1.00000000
	ldp q1, q0, [x23, #64]
	ldr q10, [sp, #192]
	ldp q2, q3, [x23, #96]
.LBB11_7:
	ldp q21, q20, [sp, #320]
	fcmge v4.4s, v14.4s, v11.4s
	fcmge v6.4s, v15.4s, v11.4s
	ldp q18, q16, [sp, #288]
	ldp q17, q19, [sp, #256]
	fcmge v7.4s, v20.4s, v11.4s
	fcmge v5.4s, v21.4s, v11.4s
	bic v16.16b, v16.16b, v4.16b
	and v4.16b, v4.16b, v11.16b
	bic v18.16b, v18.16b, v6.16b
	and v6.16b, v6.16b, v11.16b
	bic v19.16b, v19.16b, v7.16b
	bic v17.16b, v17.16b, v5.16b
	and v1.16b, v16.16b, v1.16b
	and v0.16b, v18.16b, v0.16b
	and v7.16b, v7.16b, v11.16b
	and v5.16b, v5.16b, v11.16b
	fcmeq v16.4s, v14.4s, v14.4s
	and v2.16b, v19.16b, v2.16b
	and v3.16b, v17.16b, v3.16b
	orr v1.16b, v1.16b, v4.16b
	fcmeq v4.4s, v15.4s, v15.4s
	orr v0.16b, v0.16b, v6.16b
	fcmeq v6.4s, v20.4s, v20.4s
	orr v2.16b, v2.16b, v7.16b
	fcmeq v7.4s, v21.4s, v21.4s
	orr v3.16b, v3.16b, v5.16b
	bif v1.16b, v14.16b, v16.16b
	bif v0.16b, v15.16b, v4.16b
	bif v2.16b, v20.16b, v6.16b
	bif v3.16b, v21.16b, v7.16b
	stp q1, q0, [x19]
	stp q2, q3, [x19, #32]
.LBB11_8:
	cmp x22, x24
	ccmp x22, #0, #4, ne
	b.ne .LBB11_11
	cbz x25, .LBB11_22
	mov x22, xzr
	mov x19, x25
	mov x25, xzr
	b .LBB11_12
.LBB11_11:
	add x8, x22, #64
	mov x19, x22
	mov x22, x8
.LBB11_12:
	ldp q14, q15, [x19]
	fcmp s10, s8
	ldp q5, q6, [x19, #32]
	stur s10, [x29, #-244]
	fcmgt v4.4s, v14.4s, #0.0
	fcmgt v0.4s, v11.4s, v14.4s
	fcmgt v7.4s, v15.4s, #0.0
	fcmgt v1.4s, v11.4s, v15.4s
	fcmgt v16.4s, v5.4s, #0.0
	fcmgt v2.4s, v11.4s, v5.4s
	fcmgt v17.4s, v6.4s, #0.0
	fcmgt v3.4s, v11.4s, v6.4s
	stp q6, q5, [sp, #320]
	stp q7, q4, [sp, #288]
	and v0.16b, v0.16b, v4.16b
	and v1.16b, v1.16b, v7.16b
	and v4.16b, v2.16b, v16.16b
	and v3.16b, v3.16b, v17.16b
	stp q17, q16, [sp, #256]
	mov v7.16b, v0.16b
	mov v2.16b, v1.16b
	mov v1.16b, v4.16b
	mov v0.16b, v3.16b
	bsl v7.16b, v14.16b, v11.16b
	bsl v2.16b, v15.16b, v11.16b
	bsl v1.16b, v5.16b, v11.16b
	bsl v0.16b, v6.16b, v11.16b
	stp q7, q2, [x23]
	stp q1, q0, [x23, #32]
	b.ne .LBB11_15
	fcmge v3.4s, v11.4s, v7.4s
	fcmge v4.4s, v11.4s, v2.4s
	ldr q19, [sp, #208]
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
	cbnz x8, .LBB11_6
	cmtst v4.4s, v2.4s, v2.4s
	cmtst v16.4s, v1.4s, v1.4s
	cmtst v3.4s, v7.4s, v7.4s
	cmtst v17.4s, v0.4s, v0.4s
	xtn v5.4h, v4.4s
	xtn v4.4h, v16.4s
	ldr q16, [sp, #64]
	xtn v6.4h, v3.4s
	xtn v3.4h, v17.4s
	b .LBB11_18
.LBB11_15:
	fmov s3, w20
	fcmp s10, s3
	b.ne .LBB11_19
	fcmge v3.4s, v11.4s, v7.4s
	fcmge v4.4s, v11.4s, v2.4s
	ldr q19, [sp, #208]
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
	cbnz x8, .LBB11_6
	cmtst v3.4s, v7.4s, v7.4s
	cmtst v4.4s, v2.4s, v2.4s
	cmtst v16.4s, v1.4s, v1.4s
	cmtst v17.4s, v0.4s, v0.4s
	xtn v6.4h, v3.4s
	xtn v5.4h, v4.4s
	xtn v4.4h, v16.4s
	xtn v3.4h, v17.4s
	dup v16.4s, w20
.LBB11_18:
	ldp q23, q21, [sp, #160]
	movi v22.4s, #127, msl #16
	fmov v24.4s, #-1.00000000
	ldp q29, q31, [sp, #128]
	ldr q12, [sp, #112]
	mvni v9.4s, #126
	mvni v28.4s, #126
	add v7.4s, v7.4s, v21.4s
	add v2.4s, v2.4s, v21.4s
	add v1.4s, v1.4s, v21.4s
	add v0.4s, v0.4s, v21.4s
	mov v26.16b, v29.16b
	mov v27.16b, v29.16b
	mvni v30.4s, #126
	mov w8, #50297
	ushll v6.4s, v6.4h, #0
	and v17.16b, v7.16b, v22.16b
	and v19.16b, v2.16b, v22.16b
	movk w8, #15022, lsl #16
	and v20.16b, v1.16b, v22.16b
	and v21.16b, v0.16b, v22.16b
	ssra v9.4s, v0.4s, #23
	ssra v28.4s, v2.4s, #23
	mov v2.16b, v12.16b
	ssra v30.4s, v1.4s, #23
	add v17.4s, v17.4s, v23.4s
	mov v1.16b, v12.16b
	ushll v5.4s, v5.4h, #0
	ushll v4.4s, v4.4h, #0
	ushll v3.4s, v3.4h, #0
	shl v6.4s, v6.4s, #31
	fadd v18.4s, v17.4s, v24.4s
	fadd v17.4s, v17.4s, v11.4s
	shl v5.4s, v5.4s, #31
	shl v4.4s, v4.4s, #31
	shl v3.4s, v3.4s, #31
	cmlt v6.4s, v6.4s, #0
	cmlt v5.4s, v5.4s, #0
	fdiv v17.4s, v18.4s, v17.4s
	add v18.4s, v19.4s, v23.4s
	cmlt v4.4s, v4.4s, #0
	cmlt v3.4s, v3.4s, #0
	fadd v19.4s, v18.4s, v24.4s
	fadd v18.4s, v18.4s, v11.4s
	fdiv v18.4s, v19.4s, v18.4s
	add v19.4s, v20.4s, v23.4s
	fadd v20.4s, v19.4s, v24.4s
	fadd v19.4s, v19.4s, v11.4s
	fdiv v19.4s, v20.4s, v19.4s
	add v20.4s, v21.4s, v23.4s
	mvni v23.4s, #126
	fadd v21.4s, v20.4s, v24.4s
	fadd v20.4s, v20.4s, v11.4s
	fmul v22.4s, v18.4s, v18.4s
	ssra v23.4s, v7.4s, #23
	mov v7.16b, v29.16b
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
	ldr q27, [sp, #96]
	mov v0.16b, v27.16b
	mov v23.16b, v27.16b
	fmla v0.4s, v21.4s, v31.4s
	mov v21.16b, v27.16b
	fmla v23.4s, v24.4s, v2.4s
	scvtf v2.4s, v9.4s
	fmla v21.4s, v22.4s, v26.4s
	mov v22.16b, v27.16b
	scvtf v27.4s, v30.4s
	scvtf v26.4s, v28.4s
	fmla v7.4s, v17.4s, v0.4s
	fmla v22.4s, v25.4s, v1.4s
	fmla v27.4s, v19.4s, v23.4s
	fmla v26.4s, v18.4s, v21.4s
	fmul v0.4s, v7.4s, v16.4s
	fmla v2.4s, v20.4s, v22.4s
	dup v22.4s, w8
	mov w8, #38156
	fmul v20.4s, v27.4s, v16.4s
	movk w8, #15389, lsl #16
	fmul v7.4s, v26.4s, v16.4s
	ldr q26, [sp, #80]
	mov v23.16b, v22.16b
	mov v24.16b, v22.16b
	fmul v21.4s, v2.4s, v16.4s
	frintn v16.4s, v0.4s
	frintn v18.4s, v20.4s
	frintn v17.4s, v7.4s
	frintn v19.4s, v21.4s
	fsub v1.4s, v0.4s, v16.4s
	fcvtns v16.4s, v16.4s
	fsub v0.4s, v20.4s, v18.4s
	mov v20.16b, v22.16b
	fcvtns v18.4s, v18.4s
	fsub v2.4s, v7.4s, v17.4s
	fcvtns v17.4s, v17.4s
	fsub v7.4s, v21.4s, v19.4s
	dup v21.4s, w8
	fmla v20.4s, v26.4s, v1.4s
	mov w8, #22716
	fmla v24.4s, v26.4s, v0.4s
	fcvtns v19.4s, v19.4s
	fmla v23.4s, v26.4s, v2.4s
	movk w8, #15715, lsl #16
	shl v16.4s, v16.4s, #23
	mov v25.16b, v21.16b
	mov v27.16b, v21.16b
	shl v17.4s, v17.4s, #23
	fmla v22.4s, v26.4s, v7.4s
	dup v26.4s, w8
	mov w8, #65005
	movk w8, #15989, lsl #16
	shl v18.4s, v18.4s, #23
	shl v19.4s, v19.4s, #23
	fmla v25.4s, v1.4s, v20.4s
	mov v20.16b, v21.16b
	fmla v27.4s, v0.4s, v24.4s
	mov v24.16b, v26.16b
	fmla v21.4s, v7.4s, v22.4s
	dup v22.4s, w8
	fmla v20.4s, v2.4s, v23.4s
	mov v23.16b, v26.16b
	mov v28.16b, v22.16b
	mov v29.16b, v22.16b
	fmla v23.4s, v1.4s, v25.4s
	mov v25.16b, v26.16b
	fmla v26.4s, v7.4s, v21.4s
	fmla v24.4s, v2.4s, v20.4s
	dup v20.4s, w21
	fmov v21.4s, #1.00000000
	fmla v25.4s, v0.4s, v27.4s
	mov v27.16b, v22.16b
	fmla v22.4s, v7.4s, v26.4s
	mov v26.16b, v20.16b
	fmla v28.4s, v2.4s, v24.4s
	fmov v24.4s, #1.00000000
	fmla v27.4s, v1.4s, v23.4s
	fmov v23.4s, #1.00000000
	fmla v29.4s, v0.4s, v25.4s
	fmov v25.4s, #1.00000000
	fmla v26.4s, v1.4s, v27.4s
	mov v27.16b, v20.16b
	fmla v27.4s, v2.4s, v28.4s
	mov v28.16b, v20.16b
	fmla v20.4s, v7.4s, v22.4s
	fmla v21.4s, v1.4s, v26.4s
	add v1.4s, v16.4s, v11.4s
	fmla v28.4s, v0.4s, v29.4s
	fmla v23.4s, v2.4s, v27.4s
	fmla v25.4s, v7.4s, v20.4s
	add v2.4s, v18.4s, v11.4s
	add v7.4s, v19.4s, v11.4s
	fmul v1.4s, v21.4s, v1.4s
	fmla v24.4s, v0.4s, v28.4s
	add v0.4s, v17.4s, v11.4s
	fmul v7.4s, v25.4s, v7.4s
	and v1.16b, v6.16b, v1.16b
	fmul v0.4s, v23.4s, v0.4s
	fmul v2.4s, v24.4s, v2.4s
	and v3.16b, v3.16b, v7.16b
	and v0.16b, v5.16b, v0.16b
	and v2.16b, v4.16b, v2.16b
	b .LBB11_7
.LBB11_19:
	ldr w8, [sp, #60]
	tbnz w8, #0, .LBB11_6
	fcmge v3.4s, v11.4s, v7.4s
	fcmge v4.4s, v11.4s, v2.4s
	ldr q19, [sp, #208]
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
	cbnz x8, .LBB11_6
	ldp q22, q20, [sp, #160]
	movi v21.4s, #127, msl #16
	fmov v23.4s, #-1.00000000
	ldp q29, q30, [sp, #128]
	ldp q13, q12, [sp, #96]
	mvni v25.4s, #126
	add v7.4s, v7.4s, v20.4s
	add v18.4s, v2.4s, v20.4s
	add v1.4s, v1.4s, v20.4s
	add v0.4s, v0.4s, v20.4s
	mov v26.16b, v29.16b
	mov v27.16b, v29.16b
	mov v28.16b, v12.16b
	mov v31.16b, v12.16b
	mov v9.16b, v13.16b
	and v16.16b, v7.16b, v21.16b
	and v19.16b, v18.16b, v21.16b
	mov w8, #50297
	and v20.16b, v0.16b, v21.16b
	ssra v25.4s, v7.4s, #23
	movk w8, #15022, lsl #16
	mvni v7.4s, #126
	add v16.4s, v16.4s, v22.4s
	ssra v7.4s, v18.4s, #23
	mov v18.16b, v13.16b
	fadd v17.4s, v16.4s, v23.4s
	fadd v16.4s, v16.4s, v11.4s
	scvtf v7.4s, v7.4s
	fdiv v2.4s, v17.4s, v16.4s
	add v16.4s, v19.4s, v22.4s
	and v19.16b, v1.16b, v21.16b
	fadd v17.4s, v16.4s, v23.4s
	fadd v16.4s, v16.4s, v11.4s
	fdiv v16.4s, v17.4s, v16.4s
	add v17.4s, v19.4s, v22.4s
	fadd v19.4s, v17.4s, v23.4s
	fadd v17.4s, v17.4s, v11.4s
	fdiv v17.4s, v19.4s, v17.4s
	add v19.4s, v20.4s, v22.4s
	mov v22.16b, v29.16b
	fadd v20.4s, v19.4s, v23.4s
	fadd v19.4s, v19.4s, v11.4s
	fmul v21.4s, v16.4s, v16.4s
	fmla v26.4s, v30.4s, v21.4s
	fdiv v19.4s, v20.4s, v19.4s
	fmul v20.4s, v2.4s, v2.4s
	fmla v31.4s, v21.4s, v26.4s
	mov v26.16b, v12.16b
	fmul v23.4s, v17.4s, v17.4s
	fmla v22.4s, v30.4s, v20.4s
	fmla v27.4s, v30.4s, v23.4s
	fmla v28.4s, v20.4s, v22.4s
	mvni v22.4s, #126
	ssra v22.4s, v1.4s, #23
	mov v1.16b, v13.16b
	fmla v26.4s, v23.4s, v27.4s
	fmla v9.4s, v20.4s, v28.4s
	mov v20.16b, v12.16b
	fmul v24.4s, v19.4s, v19.4s
	fmla v1.4s, v21.4s, v31.4s
	mov v21.16b, v13.16b
	scvtf v22.4s, v22.4s
	fmla v18.4s, v23.4s, v26.4s
	ldr q26, [sp, #80]
	fmla v29.4s, v30.4s, v24.4s
	mvni v30.4s, #126
	fmla v7.4s, v16.4s, v1.4s
	fmla v22.4s, v17.4s, v18.4s
	ssra v30.4s, v0.4s, #23
	scvtf v0.4s, v25.4s
	fmla v20.4s, v24.4s, v29.4s
	fmul v1.4s, v7.4s, v10.s[0]
	scvtf v23.4s, v30.4s
	fmla v0.4s, v2.4s, v9.4s
	fmul v2.4s, v22.4s, v10.s[0]
	fmla v21.4s, v24.4s, v20.4s
	dup v20.4s, w8
	mov w8, #38156
	frintn v17.4s, v1.4s
	movk w8, #15389, lsl #16
	dup v22.4s, w8
	mov w8, #22716
	fmul v0.4s, v0.4s, v10.s[0]
	frintn v18.4s, v2.4s
	mov v24.16b, v20.16b
	fmla v23.4s, v19.4s, v21.4s
	mov v21.16b, v20.16b
	movk w8, #15715, lsl #16
	fsub v1.4s, v1.4s, v17.4s
	mov v25.16b, v22.16b
	mov v27.16b, v22.16b
	fcvtns v17.4s, v17.4s
	frintn v16.4s, v0.4s
	fsub v2.4s, v2.4s, v18.4s
	fcvtns v18.4s, v18.4s
	fmul v7.4s, v23.4s, v10.s[0]
	mov v23.16b, v20.16b
	shl v17.4s, v17.4s, #23
	fsub v0.4s, v0.4s, v16.4s
	fmla v23.4s, v26.4s, v1.4s
	fmla v24.4s, v26.4s, v2.4s
	frintn v19.4s, v7.4s
	fcvtns v16.4s, v16.4s
	shl v18.4s, v18.4s, #23
	fmla v21.4s, v26.4s, v0.4s
	fmla v27.4s, v2.4s, v24.4s
	fsub v7.4s, v7.4s, v19.4s
	fcvtns v19.4s, v19.4s
	shl v16.4s, v16.4s, #23
	fmla v25.4s, v0.4s, v21.4s
	mov v21.16b, v22.16b
	fmla v20.4s, v26.4s, v7.4s
	dup v26.4s, w8
	mov w8, #65005
	movk w8, #15989, lsl #16
	shl v19.4s, v19.4s, #23
	fmla v21.4s, v1.4s, v23.4s
	mov v23.16b, v26.16b
	mov v24.16b, v26.16b
	fmla v22.4s, v7.4s, v20.4s
	dup v20.4s, w8
	fmla v23.4s, v0.4s, v25.4s
	mov v25.16b, v26.16b
	fmla v24.4s, v1.4s, v21.4s
	mov v21.16b, v20.16b
	fmla v26.4s, v7.4s, v22.4s
	dup v22.4s, w21
	fmla v25.4s, v2.4s, v27.4s
	mov v27.16b, v20.16b
	fmla v21.4s, v0.4s, v23.4s
	mov v23.16b, v20.16b
	fmla v20.4s, v7.4s, v26.4s
	mov v26.16b, v22.16b
	fmla v23.4s, v1.4s, v24.4s
	fmla v27.4s, v2.4s, v25.4s
	mov v25.16b, v22.16b
	fmov v24.4s, #1.00000000
	fmla v25.4s, v0.4s, v21.4s
	mov v21.16b, v22.16b
	fmla v22.4s, v7.4s, v20.4s
	fmla v26.4s, v1.4s, v23.4s
	fmov v23.4s, #1.00000000
	fmov v20.4s, #1.00000000
	fmla v21.4s, v2.4s, v27.4s
	fmov v27.4s, #1.00000000
	fmla v24.4s, v0.4s, v25.4s
	add v0.4s, v16.4s, v11.4s
	fmla v23.4s, v1.4s, v26.4s
	add v1.4s, v17.4s, v11.4s
	fmla v20.4s, v7.4s, v22.4s
	add v7.4s, v19.4s, v11.4s
	ldr q17, [sp, #32]
	fmla v27.4s, v2.4s, v21.4s
	add v2.4s, v18.4s, v11.4s
	fmul v0.4s, v24.4s, v0.4s
	fmul v16.4s, v23.4s, v1.4s
	mov v1.16b, v3.16b
	fmul v7.4s, v20.4s, v7.4s
	mov v3.16b, v6.16b
	fmul v2.4s, v27.4s, v2.4s
	bsl v1.16b, v17.16b, v0.16b
	mov v0.16b, v4.16b
	bsl v3.16b, v17.16b, v7.16b
	bsl v0.16b, v17.16b, v16.16b
	bit v2.16b, v17.16b, v5.16b
	b .LBB11_7
.LBB11_22:
	ldr w8, [sp, #360]
	cbz w8, .LBB11_24
	ldp x2, x0, [sp, #16]
	ldr x1, [sp, #8]
	bl memcpy
.LBB11_24:
	add sp, sp, #672
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
.LBB11_25:
	.cfi_restore_state
	mov x19, x1
	bl archmage::tokens::generated::arm::neon_detect
	ldr q10, [sp, #192]
	mov x1, x19
	cbnz w0, .LBB11_2
.LBB11_26:
	mov x0, x22
	fmov s0, s10
	add sp, sp, #672
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
	b linear_srgb::simd::gamma_to_linear_slice_tier_scalar_t
