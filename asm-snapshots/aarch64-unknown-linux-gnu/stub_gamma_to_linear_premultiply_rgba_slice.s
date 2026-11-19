.section .text.stub_gamma_to_linear_premultiply_rgba_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_premultiply_rgba_slice
	.p2align	2
.type	stub_gamma_to_linear_premultiply_rgba_slice,@function
stub_gamma_to_linear_premultiply_rgba_slice:
	.cfi_startproc
	stp d13, d12, [sp, #-144]!
	.cfi_def_cfa_offset 144
	stp d11, d10, [sp, #16]
	stp d9, d8, [sp, #32]
	stp x29, x30, [sp, #48]
	stp x28, x27, [sp, #64]
	stp x26, x25, [sp, #80]
	stp x24, x23, [sp, #96]
	stp x22, x21, [sp, #112]
	stp x20, x19, [sp, #128]
	add x29, sp, #48
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
	.cfi_remember_state
	sub sp, sp, #736
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	fmov s10, s0
	mov x22, x0
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB10_27
	cmp w8, #2
	str q10, [sp, #192]
	b.ne .LBB10_26
.LBB10_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x19, x1, #0xc
	lsr x20, x1, #4
	add x21, x22, x8, lsl #2
	lsl x23, x19, #2
	stp x23, x21, [sp, #16]
	b.eq .LBB10_4
	mov w8, #64
	mov w1, wzr
	sub x2, x8, x23
	add x8, sp, #492
	add x0, x8, x23
	bl memset
	add x0, sp, #492
	mov x1, x21
	mov x2, x23
	bl memcpy
	add x8, sp, #237
	ldr q10, [sp, #192]
	add x9, sp, #221
	ldur q0, [x8, #255]
	add x8, sp, #253
	ldur q1, [x8, #255]
	add x8, sp, #173
	stur q0, [x8, #255]
	add x8, sp, #269
	ldur q0, [x8, #255]
	add x8, sp, #285
	ldur q2, [x8, #255]
	add x8, sp, #189
	stur q1, [x8, #255]
	add x8, sp, #205
	stur q0, [x8, #255]
	mov w8, #1
	stur q2, [x9, #255]
	b .LBB10_5
.LBB10_4:
	mov w8, wzr
.LBB10_5:
	str w8, [sp, #424]
	movi d0, #0000000000000000
	add x8, sp, #424
	cmp x19, #0
	add x10, x8, #4
	fmov s1, #4.00000000
	csel x25, xzr, x10, eq
	fcmp s10, #0.0
	str x10, [sp, #8]
	mov w10, #39322
	fmov w9, s10
	fmov v11.4s, #1.00000000
	movk w10, #16409, lsl #16
	fcmeq v0.4s, v10.4s, v0.4s
	mov w11, #931135488
	cset w14, lt
	fcmp s10, s1
	dup v1.4s, w10
	and w12, w9, #0x7fffffff
	ubfx w9, w9, #23, #8
	mov w8, #8388607
	mov w13, #64269
	fmov s8, w10
	sub w10, w12, #1
	str q1, [sp, #64]
	dup v1.4s, w11
	and v0.16b, v0.16b, v11.16b
	csinc w11, w14, wzr, ls
	cmp w9, #254
	mov w9, #1267
	movk w13, #74, lsl #16
	movk w9, #16181, lsl #16
	ccmp w10, w8, #0, ls
	str q1, [sp, #208]
	dup v2.4s, w13
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
	sub x23, x29, #224
	movk w21, #16177, lsl #16
	sub x26, x29, #65
	sub x27, x29, #224
	stp q0, q1, [sp, #96]
	dup v0.4s, w10
	sub x28, x29, #228
	movk w20, #16085, lsl #16
	str q0, [sp, #80]
	b .LBB10_8
.LBB10_6:
	sub x0, x29, #160
	sub x1, x29, #96
	stp x26, x27, [x29, #-96]
	stur x28, [x29, #-80]
	bl linear_srgb::tf::math::gamma_x16::<archmage::tokens::generated::arm::NeonToken>::{closure#0}
	ldp q6, q1, [x23, #64]
	fmov v11.4s, #1.00000000
	ldp q0, q2, [x23, #96]
	ldr q10, [sp, #192]
.LBB10_7:
	ldp q27, q26, [sp, #368]
	fmov v5.4s, #1.00000000
	ldp q17, q22, [sp, #224]
	fmov v16.4s, #1.00000000
	ldp q29, q28, [sp, #336]
	fmov v19.4s, #1.00000000
	fcmge v4.4s, v26.4s, v11.4s
	fcmge v3.4s, v27.4s, v11.4s
	ldr q25, [sp, #400]
	fmov v21.4s, #1.00000000
	ldp q23, q24, [sp, #256]
	fcmge v7.4s, v25.4s, v11.4s
	fcmge v18.4s, v28.4s, v11.4s
	mov v5.s[0], v29.s[0]
	ldp q31, q30, [sp, #304]
	fcmeq v20.4s, v26.4s, v26.4s
	bic v22.16b, v22.16b, v4.16b
	bic v17.16b, v17.16b, v3.16b
	and v3.16b, v3.16b, v11.16b
	and v4.16b, v4.16b, v11.16b
	bic v23.16b, v23.16b, v7.16b
	bic v24.16b, v24.16b, v18.16b
	mov v16.s[0], v30.s[0]
	and v1.16b, v22.16b, v1.16b
	ldr q22, [sp, #288]
	and v6.16b, v17.16b, v6.16b
	mov v19.s[0], v31.s[0]
	fcmeq v17.4s, v27.4s, v27.4s
	mov v5.s[1], v29.s[0]
	mov v21.s[0], v22.s[0]
	and v0.16b, v23.16b, v0.16b
	and v2.16b, v24.16b, v2.16b
	orr v3.16b, v6.16b, v3.16b
	and v6.16b, v7.16b, v11.16b
	and v7.16b, v18.16b, v11.16b
	mov v16.s[1], v30.s[0]
	fcmeq v18.4s, v25.4s, v25.4s
	orr v1.16b, v1.16b, v4.16b
	mov v19.s[1], v31.s[0]
	fcmeq v4.4s, v28.4s, v28.4s
	mov v5.s[2], v29.s[0]
	mov v21.s[1], v22.s[0]
	orr v0.16b, v0.16b, v6.16b
	orr v2.16b, v2.16b, v7.16b
	bif v3.16b, v27.16b, v17.16b
	bif v1.16b, v26.16b, v20.16b
	mov v16.s[2], v30.s[0]
	bif v0.16b, v25.16b, v18.16b
	mov v19.s[2], v31.s[0]
	bif v2.16b, v28.16b, v4.16b
	mov v21.s[2], v22.s[0]
	fmul v3.4s, v5.4s, v3.4s
	fmul v1.4s, v16.4s, v1.4s
	fmul v0.4s, v19.4s, v0.4s
	fmul v2.4s, v21.4s, v2.4s
	stp q3, q1, [x19]
	str s29, [x19, #12]
	stp q0, q2, [x19, #32]
	str s30, [x19, #28]
	str s31, [x19, #44]
	str s22, [x19, #60]
.LBB10_8:
	cmp x22, x24
	ccmp x22, #0, #4, ne
	b.ne .LBB10_11
	cbz x25, .LBB10_23
	mov x22, xzr
	mov x19, x25
	mov x25, xzr
	b .LBB10_12
.LBB10_11:
	add x8, x22, #64
	mov x19, x22
	mov x22, x8
.LBB10_12:
	ldp q6, q5, [x19]
	fcmp s10, s8
	ldp q4, q16, [x19, #32]
	stur s10, [x29, #-228]
	fcmgt v19.4s, v6.4s, #0.0
	fcmgt v0.4s, v11.4s, v6.4s
	fcmgt v18.4s, v5.4s, #0.0
	fcmgt v1.4s, v11.4s, v5.4s
	fcmgt v17.4s, v4.4s, #0.0
	fcmgt v2.4s, v11.4s, v4.4s
	fcmgt v7.4s, v16.4s, #0.0
	fcmgt v3.4s, v11.4s, v16.4s
	stp q6, q5, [sp, #368]
	str q4, [sp, #400]
	and v0.16b, v0.16b, v19.16b
	stp q19, q18, [sp, #224]
	and v1.16b, v1.16b, v18.16b
	and v2.16b, v2.16b, v17.16b
	stp q17, q7, [sp, #256]
	and v3.16b, v3.16b, v7.16b
	mov v7.16b, v0.16b
	ldr s0, [x19, #12]
	stp q0, q16, [sp, #336]
	mov v0.16b, v3.16b
	bsl v7.16b, v6.16b, v11.16b
	mov v6.16b, v1.16b
	mov v1.16b, v2.16b
	ldr s2, [x19, #28]
	bsl v0.16b, v16.16b, v11.16b
	bsl v6.16b, v5.16b, v11.16b
	bsl v1.16b, v4.16b, v11.16b
	str q2, [sp, #320]
	ldr s2, [x19, #44]
	str q2, [sp, #304]
	ldr s2, [x19, #60]
	stp q7, q6, [x23]
	str q2, [sp, #288]
	stp q1, q0, [x23, #32]
	b.ne .LBB10_15
	fcmge v2.4s, v11.4s, v7.4s
	fcmge v3.4s, v11.4s, v6.4s
	ldr q19, [sp, #208]
	fcmge v4.4s, v11.4s, v1.4s
	fcmge v5.4s, v11.4s, v0.4s
	fcmge v16.4s, v7.4s, v19.4s
	fcmge v17.4s, v6.4s, v19.4s
	fcmge v18.4s, v1.4s, v19.4s
	fcmge v19.4s, v0.4s, v19.4s
	mvn v20.16b, v2.16b
	mvn v21.16b, v3.16b
	cmeq v2.4s, v7.4s, #0
	mvn v22.16b, v4.16b
	mvn v23.16b, v5.16b
	cmeq v3.4s, v6.4s, #0
	cmeq v4.4s, v1.4s, #0
	cmeq v5.4s, v0.4s, #0
	orn v16.16b, v20.16b, v16.16b
	orn v17.16b, v21.16b, v17.16b
	orn v18.16b, v22.16b, v18.16b
	orn v19.16b, v23.16b, v19.16b
	bic v16.16b, v16.16b, v2.16b
	bic v17.16b, v17.16b, v3.16b
	bic v18.16b, v18.16b, v4.16b
	bic v19.16b, v19.16b, v5.16b
	orr v16.16b, v17.16b, v16.16b
	orr v17.16b, v18.16b, v19.16b
	orr v16.16b, v16.16b, v17.16b
	xtn v16.4h, v16.4s
	shl v16.4h, v16.4h, #15
	cmlt v16.4h, v16.4h, #0
	fmov x8, d16
	cbnz x8, .LBB10_6
	ldp q22, q20, [sp, #160]
	movi v21.4s, #127, msl #16
	fmov v23.4s, #-1.00000000
	ldp q30, q31, [sp, #128]
	mvni v28.4s, #126
	mvni v26.4s, #126
	ldr q9, [sp, #112]
	add v7.4s, v7.4s, v20.4s
	add v6.4s, v6.4s, v20.4s
	add v1.4s, v1.4s, v20.4s
	add v0.4s, v0.4s, v20.4s
	mov v29.16b, v30.16b
	mov v24.16b, v30.16b
	mov v25.16b, v30.16b
	mvni v27.4s, #126
	mov w8, #50297
	and v16.16b, v7.16b, v21.16b
	and v18.16b, v6.16b, v21.16b
	movk w8, #15022, lsl #16
	and v19.16b, v1.16b, v21.16b
	and v20.16b, v0.16b, v21.16b
	ssra v28.4s, v1.4s, #23
	mov v1.16b, v9.16b
	ssra v26.4s, v7.4s, #23
	mov v7.16b, v9.16b
	add v16.4s, v16.4s, v22.4s
	ssra v27.4s, v6.4s, #23
	mvni v6.4s, #126
	fadd v17.4s, v16.4s, v23.4s
	fadd v16.4s, v16.4s, v11.4s
	ssra v6.4s, v0.4s, #23
	scvtf v6.4s, v6.4s
	fdiv v16.4s, v17.4s, v16.4s
	add v17.4s, v18.4s, v22.4s
	fadd v18.4s, v17.4s, v23.4s
	fadd v17.4s, v17.4s, v11.4s
	fdiv v17.4s, v18.4s, v17.4s
	add v18.4s, v19.4s, v22.4s
	fadd v19.4s, v18.4s, v23.4s
	fadd v18.4s, v18.4s, v11.4s
	fdiv v18.4s, v19.4s, v18.4s
	add v19.4s, v20.4s, v22.4s
	fadd v20.4s, v19.4s, v23.4s
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
	ldr q29, [sp, #96]
	fmul v23.4s, v19.4s, v19.4s
	mov v0.16b, v29.16b
	mov v25.16b, v29.16b
	fmla v0.4s, v21.4s, v7.4s
	mov v7.16b, v29.16b
	mov v21.16b, v29.16b
	fmla v30.4s, v31.4s, v23.4s
	mov v31.16b, v9.16b
	fmla v7.4s, v22.4s, v1.4s
	dup v22.4s, w8
	mov w8, #38156
	fmla v31.4s, v20.4s, v24.4s
	mov v24.16b, v9.16b
	movk w8, #15389, lsl #16
	fmla v24.4s, v23.4s, v30.4s
	fmla v25.4s, v20.4s, v31.4s
	scvtf v20.4s, v26.4s
	scvtf v26.4s, v27.4s
	scvtf v27.4s, v28.4s
	fmla v21.4s, v23.4s, v24.4s
	mov v23.16b, v22.16b
	mov v24.16b, v22.16b
	fmla v20.4s, v16.4s, v25.4s
	fmla v26.4s, v17.4s, v0.4s
	fmla v27.4s, v18.4s, v7.4s
	ldp q7, q25, [sp, #64]
	fmla v6.4s, v19.4s, v21.4s
	fmul v0.4s, v20.4s, v7.4s
	fmul v1.4s, v26.4s, v7.4s
	fmul v20.4s, v27.4s, v7.4s
	fmul v21.4s, v6.4s, v7.4s
	frintn v16.4s, v0.4s
	frintn v17.4s, v1.4s
	frintn v18.4s, v20.4s
	frintn v19.4s, v21.4s
	fsub v6.4s, v0.4s, v16.4s
	fsub v7.4s, v1.4s, v17.4s
	fcvtns v16.4s, v16.4s
	fsub v0.4s, v20.4s, v18.4s
	mov v20.16b, v22.16b
	fcvtns v17.4s, v17.4s
	fcvtns v18.4s, v18.4s
	fsub v1.4s, v21.4s, v19.4s
	dup v21.4s, w8
	mov w8, #22716
	fmla v20.4s, v25.4s, v6.4s
	fmla v23.4s, v25.4s, v7.4s
	movk w8, #15715, lsl #16
	fmla v24.4s, v25.4s, v0.4s
	dup v26.4s, w8
	mov w8, #65005
	mov v27.16b, v21.16b
	movk w8, #15989, lsl #16
	fcvtns v19.4s, v19.4s
	fmla v22.4s, v25.4s, v1.4s
	mov v25.16b, v21.16b
	shl v16.4s, v16.4s, #23
	shl v17.4s, v17.4s, #23
	shl v18.4s, v18.4s, #23
	fmla v27.4s, v7.4s, v23.4s
	dup v23.4s, w8
	fmla v25.4s, v6.4s, v20.4s
	mov v20.16b, v21.16b
	shl v19.4s, v19.4s, #23
	fmla v21.4s, v1.4s, v22.4s
	mov v22.16b, v26.16b
	mov v28.16b, v23.16b
	fmla v20.4s, v0.4s, v24.4s
	mov v24.16b, v26.16b
	fmla v22.4s, v6.4s, v25.4s
	mov v25.16b, v26.16b
	fmla v26.4s, v1.4s, v21.4s
	mov v21.16b, v23.16b
	fmla v24.4s, v7.4s, v27.4s
	fmov v27.4s, #1.00000000
	fmla v25.4s, v0.4s, v20.4s
	dup v20.4s, w21
	fmla v21.4s, v6.4s, v22.4s
	fmov v22.4s, #1.00000000
	fmla v28.4s, v7.4s, v24.4s
	mov v24.16b, v23.16b
	fmla v23.4s, v1.4s, v26.4s
	mov v26.16b, v20.16b
	fmla v24.4s, v0.4s, v25.4s
	mov v25.16b, v20.16b
	fmla v26.4s, v7.4s, v28.4s
	mov v28.16b, v20.16b
	fmla v20.4s, v1.4s, v23.4s
	fmla v25.4s, v6.4s, v21.4s
	fmov v21.4s, #1.00000000
	fmla v28.4s, v0.4s, v24.4s
	fmov v24.4s, #1.00000000
	fmla v22.4s, v7.4s, v26.4s
	add v7.4s, v19.4s, v11.4s
	fmla v27.4s, v6.4s, v25.4s
	add v6.4s, v16.4s, v11.4s
	fmla v21.4s, v0.4s, v28.4s
	fmla v24.4s, v1.4s, v20.4s
	add v0.4s, v17.4s, v11.4s
	add v1.4s, v18.4s, v11.4s
	fmul v6.4s, v27.4s, v6.4s
	fmul v0.4s, v22.4s, v0.4s
	fmul v16.4s, v21.4s, v1.4s
	b .LBB10_18
.LBB10_15:
	fmov s2, w20
	fcmp s10, s2
	b.ne .LBB10_20
	fcmge v2.4s, v11.4s, v7.4s
	fcmge v3.4s, v11.4s, v6.4s
	ldr q19, [sp, #208]
	fcmge v4.4s, v11.4s, v1.4s
	fcmge v5.4s, v11.4s, v0.4s
	fcmge v16.4s, v7.4s, v19.4s
	fcmge v17.4s, v6.4s, v19.4s
	fcmge v18.4s, v1.4s, v19.4s
	fcmge v19.4s, v0.4s, v19.4s
	mvn v20.16b, v2.16b
	mvn v21.16b, v3.16b
	cmeq v2.4s, v7.4s, #0
	mvn v22.16b, v4.16b
	mvn v23.16b, v5.16b
	cmeq v3.4s, v6.4s, #0
	cmeq v4.4s, v1.4s, #0
	cmeq v5.4s, v0.4s, #0
	orn v16.16b, v20.16b, v16.16b
	orn v17.16b, v21.16b, v17.16b
	orn v18.16b, v22.16b, v18.16b
	orn v19.16b, v23.16b, v19.16b
	bic v16.16b, v16.16b, v2.16b
	bic v17.16b, v17.16b, v3.16b
	bic v18.16b, v18.16b, v4.16b
	bic v19.16b, v19.16b, v5.16b
	orr v16.16b, v17.16b, v16.16b
	orr v17.16b, v18.16b, v19.16b
	orr v16.16b, v16.16b, v17.16b
	xtn v16.4h, v16.4s
	shl v16.4h, v16.4h, #15
	cmlt v16.4h, v16.4h, #0
	fmov x8, d16
	cbnz x8, .LBB10_6
	ldp q22, q20, [sp, #160]
	movi v21.4s, #127, msl #16
	fmov v23.4s, #-1.00000000
	ldp q30, q31, [sp, #128]
	mvni v28.4s, #126
	mvni v26.4s, #126
	ldr q9, [sp, #112]
	add v7.4s, v7.4s, v20.4s
	add v6.4s, v6.4s, v20.4s
	add v1.4s, v1.4s, v20.4s
	add v0.4s, v0.4s, v20.4s
	mov v29.16b, v30.16b
	mov v24.16b, v30.16b
	mov v25.16b, v30.16b
	mvni v27.4s, #126
	mov w8, #50297
	and v16.16b, v7.16b, v21.16b
	and v18.16b, v6.16b, v21.16b
	movk w8, #15022, lsl #16
	and v19.16b, v1.16b, v21.16b
	and v20.16b, v0.16b, v21.16b
	ssra v28.4s, v1.4s, #23
	mov v1.16b, v9.16b
	ssra v26.4s, v7.4s, #23
	mov v7.16b, v9.16b
	add v16.4s, v16.4s, v22.4s
	ssra v27.4s, v6.4s, #23
	mvni v6.4s, #126
	fadd v17.4s, v16.4s, v23.4s
	fadd v16.4s, v16.4s, v11.4s
	ssra v6.4s, v0.4s, #23
	scvtf v6.4s, v6.4s
	fdiv v16.4s, v17.4s, v16.4s
	add v17.4s, v18.4s, v22.4s
	fadd v18.4s, v17.4s, v23.4s
	fadd v17.4s, v17.4s, v11.4s
	fdiv v17.4s, v18.4s, v17.4s
	add v18.4s, v19.4s, v22.4s
	fadd v19.4s, v18.4s, v23.4s
	fadd v18.4s, v18.4s, v11.4s
	fdiv v18.4s, v19.4s, v18.4s
	add v19.4s, v20.4s, v22.4s
	fadd v20.4s, v19.4s, v23.4s
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
	ldr q29, [sp, #96]
	fmul v23.4s, v19.4s, v19.4s
	mov v0.16b, v29.16b
	mov v25.16b, v29.16b
	fmla v0.4s, v21.4s, v7.4s
	mov v7.16b, v29.16b
	mov v21.16b, v29.16b
	fmla v30.4s, v31.4s, v23.4s
	mov v31.16b, v9.16b
	fmla v7.4s, v22.4s, v1.4s
	dup v1.4s, w20
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
	ldr q25, [sp, #80]
	fmla v27.4s, v18.4s, v7.4s
	fmla v6.4s, v19.4s, v21.4s
	dup v21.4s, w8
	mov w8, #38156
	fmul v0.4s, v20.4s, v1.4s
	fmul v7.4s, v26.4s, v1.4s
	movk w8, #15389, lsl #16
	fmul v20.4s, v27.4s, v1.4s
	dup v22.4s, w8
	mov w8, #22716
	mov v23.16b, v21.16b
	mov v24.16b, v21.16b
	movk w8, #15715, lsl #16
	fmul v1.4s, v6.4s, v1.4s
	dup v26.4s, w8
	mov w8, #65005
	frintn v16.4s, v0.4s
	frintn v17.4s, v7.4s
	mov v27.16b, v22.16b
	frintn v18.4s, v20.4s
	movk w8, #15989, lsl #16
	frintn v19.4s, v1.4s
	fsub v6.4s, v0.4s, v16.4s
	fsub v7.4s, v7.4s, v17.4s
	fcvtns v16.4s, v16.4s
	fsub v0.4s, v20.4s, v18.4s
	mov v20.16b, v21.16b
	fcvtns v17.4s, v17.4s
	fcvtns v18.4s, v18.4s
	fsub v1.4s, v1.4s, v19.4s
	fcvtns v19.4s, v19.4s
	fmla v20.4s, v25.4s, v6.4s
	fmla v23.4s, v25.4s, v7.4s
	shl v16.4s, v16.4s, #23
	fmla v24.4s, v25.4s, v0.4s
	shl v17.4s, v17.4s, #23
	shl v18.4s, v18.4s, #23
	fmla v21.4s, v25.4s, v1.4s
	mov v25.16b, v22.16b
	shl v19.4s, v19.4s, #23
	fmla v27.4s, v7.4s, v23.4s
	dup v23.4s, w8
	fmla v25.4s, v6.4s, v20.4s
	mov v20.16b, v22.16b
	fmla v22.4s, v1.4s, v21.4s
	mov v21.16b, v26.16b
	mov v28.16b, v23.16b
	fmla v20.4s, v0.4s, v24.4s
	mov v24.16b, v26.16b
	fmla v21.4s, v6.4s, v25.4s
	mov v25.16b, v26.16b
	fmla v26.4s, v1.4s, v22.4s
	mov v22.16b, v23.16b
	fmla v24.4s, v7.4s, v27.4s
	fmov v27.4s, #1.00000000
	fmla v25.4s, v0.4s, v20.4s
	dup v20.4s, w21
	fmla v22.4s, v6.4s, v21.4s
	fmov v21.4s, #1.00000000
	fmla v28.4s, v7.4s, v24.4s
	mov v24.16b, v23.16b
	fmla v23.4s, v1.4s, v26.4s
	mov v26.16b, v20.16b
	fmla v24.4s, v0.4s, v25.4s
	mov v25.16b, v20.16b
	fmla v26.4s, v7.4s, v28.4s
	mov v28.16b, v20.16b
	fmla v20.4s, v1.4s, v23.4s
	fmla v25.4s, v6.4s, v22.4s
	fmov v22.4s, #1.00000000
	fmla v28.4s, v0.4s, v24.4s
	fmov v24.4s, #1.00000000
	fmla v21.4s, v7.4s, v26.4s
	add v7.4s, v19.4s, v11.4s
	fmla v27.4s, v6.4s, v25.4s
	add v6.4s, v16.4s, v11.4s
	fmla v22.4s, v0.4s, v28.4s
	fmla v24.4s, v1.4s, v20.4s
	add v0.4s, v17.4s, v11.4s
	add v1.4s, v18.4s, v11.4s
	fmul v6.4s, v27.4s, v6.4s
	fmul v0.4s, v21.4s, v0.4s
	fmul v16.4s, v22.4s, v1.4s
.LBB10_18:
	fmul v7.4s, v24.4s, v7.4s
	bic v6.16b, v6.16b, v2.16b
	bic v1.16b, v0.16b, v3.16b
	bic v0.16b, v16.16b, v4.16b
	bic v2.16b, v7.16b, v5.16b
.LBB10_19:
	str q6, [x23, #64]
	b .LBB10_7
.LBB10_20:
	ldr w8, [sp, #60]
	tbnz w8, #0, .LBB10_6
	fcmge v2.4s, v11.4s, v7.4s
	fcmge v3.4s, v11.4s, v6.4s
	ldr q19, [sp, #208]
	fcmge v4.4s, v11.4s, v1.4s
	fcmge v5.4s, v11.4s, v0.4s
	fcmge v16.4s, v7.4s, v19.4s
	fcmge v17.4s, v6.4s, v19.4s
	fcmge v18.4s, v1.4s, v19.4s
	fcmge v19.4s, v0.4s, v19.4s
	mvn v20.16b, v2.16b
	mvn v21.16b, v3.16b
	cmeq v2.4s, v7.4s, #0
	mvn v22.16b, v4.16b
	mvn v23.16b, v5.16b
	cmeq v3.4s, v6.4s, #0
	cmeq v4.4s, v1.4s, #0
	cmeq v5.4s, v0.4s, #0
	orn v16.16b, v20.16b, v16.16b
	orn v17.16b, v21.16b, v17.16b
	orn v18.16b, v22.16b, v18.16b
	orn v19.16b, v23.16b, v19.16b
	bic v16.16b, v16.16b, v2.16b
	bic v17.16b, v17.16b, v3.16b
	bic v18.16b, v18.16b, v4.16b
	bic v19.16b, v19.16b, v5.16b
	orr v16.16b, v17.16b, v16.16b
	orr v17.16b, v18.16b, v19.16b
	orr v16.16b, v16.16b, v17.16b
	xtn v16.4h, v16.4s
	shl v16.4h, v16.4h, #15
	cmlt v16.4h, v16.4h, #0
	fmov x8, d16
	cbnz x8, .LBB10_6
	ldp q22, q20, [sp, #160]
	movi v21.4s, #127, msl #16
	fmov v23.4s, #-1.00000000
	ldp q29, q30, [sp, #128]
	ldp q13, q12, [sp, #96]
	mvni v25.4s, #126
	add v7.4s, v7.4s, v20.4s
	add v18.4s, v6.4s, v20.4s
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
	fdiv v6.4s, v17.4s, v16.4s
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
	fmul v20.4s, v6.4s, v6.4s
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
	fmla v0.4s, v6.4s, v9.4s
	fmul v6.4s, v22.4s, v10.s[0]
	fmla v21.4s, v24.4s, v20.4s
	dup v20.4s, w8
	mov w8, #38156
	frintn v17.4s, v1.4s
	movk w8, #15389, lsl #16
	dup v22.4s, w8
	mov w8, #22716
	fmul v0.4s, v0.4s, v10.s[0]
	frintn v18.4s, v6.4s
	mov v24.16b, v20.16b
	fmla v23.4s, v19.4s, v21.4s
	mov v21.16b, v20.16b
	movk w8, #15715, lsl #16
	fsub v1.4s, v1.4s, v17.4s
	mov v25.16b, v22.16b
	mov v27.16b, v22.16b
	fcvtns v17.4s, v17.4s
	frintn v16.4s, v0.4s
	fsub v6.4s, v6.4s, v18.4s
	fcvtns v18.4s, v18.4s
	fmul v7.4s, v23.4s, v10.s[0]
	mov v23.16b, v20.16b
	shl v17.4s, v17.4s, #23
	fsub v0.4s, v0.4s, v16.4s
	fmla v23.4s, v26.4s, v1.4s
	fmla v24.4s, v26.4s, v6.4s
	frintn v19.4s, v7.4s
	fcvtns v16.4s, v16.4s
	shl v18.4s, v18.4s, #23
	fmla v21.4s, v26.4s, v0.4s
	fmla v27.4s, v6.4s, v24.4s
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
	fmla v25.4s, v6.4s, v27.4s
	mov v27.16b, v20.16b
	fmla v21.4s, v0.4s, v23.4s
	mov v23.16b, v20.16b
	fmla v20.4s, v7.4s, v26.4s
	mov v26.16b, v22.16b
	fmla v27.4s, v6.4s, v25.4s
	mov v25.16b, v22.16b
	fmla v23.4s, v1.4s, v24.4s
	fmov v24.4s, #1.00000000
	fmla v25.4s, v0.4s, v21.4s
	mov v21.16b, v22.16b
	fmla v22.4s, v7.4s, v20.4s
	fmla v26.4s, v1.4s, v23.4s
	fmov v23.4s, #1.00000000
	fmov v20.4s, #1.00000000
	fmla v21.4s, v6.4s, v27.4s
	fmov v27.4s, #1.00000000
	fmla v24.4s, v0.4s, v25.4s
	add v0.4s, v16.4s, v11.4s
	fmla v23.4s, v1.4s, v26.4s
	fmla v20.4s, v7.4s, v22.4s
	add v1.4s, v17.4s, v11.4s
	add v7.4s, v19.4s, v11.4s
	ldr q17, [sp, #32]
	fmla v27.4s, v6.4s, v21.4s
	add v6.4s, v18.4s, v11.4s
	fmul v0.4s, v24.4s, v0.4s
	fmul v1.4s, v23.4s, v1.4s
	fmul v7.4s, v20.4s, v7.4s
	fmul v16.4s, v27.4s, v6.4s
	mov v6.16b, v2.16b
	mov v2.16b, v5.16b
	bit v1.16b, v17.16b, v3.16b
	bsl v6.16b, v17.16b, v0.16b
	mov v0.16b, v4.16b
	bsl v2.16b, v17.16b, v7.16b
	bsl v0.16b, v17.16b, v16.16b
	b .LBB10_19
.LBB10_23:
	ldr w8, [sp, #424]
	cbz w8, .LBB10_25
	ldp x2, x0, [sp, #16]
	ldr x1, [sp, #8]
	bl memcpy
.LBB10_25:
	add sp, sp, #736
	.cfi_def_cfa wsp, 144
	ldp x20, x19, [sp, #128]
	ldp x22, x21, [sp, #112]
	ldp x24, x23, [sp, #96]
	ldp x26, x25, [sp, #80]
	ldp x28, x27, [sp, #64]
	ldp x29, x30, [sp, #48]
	ldp d9, d8, [sp, #32]
	ldp d11, d10, [sp, #16]
	ldp d13, d12, [sp], #144
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
	ret
.LBB10_26:
	.cfi_restore_state
	mov x19, x1
	bl archmage::tokens::generated::arm::neon_detect
	ldr q10, [sp, #192]
	mov x1, x19
	cbnz w0, .LBB10_2
.LBB10_27:
	mov x0, x22
	fmov s0, s10
	add sp, sp, #736
	.cfi_def_cfa wsp, 144
	ldp x20, x19, [sp, #128]
	ldp x22, x21, [sp, #112]
	ldp x24, x23, [sp, #96]
	ldp x26, x25, [sp, #80]
	ldp x28, x27, [sp, #64]
	ldp x29, x30, [sp, #48]
	ldp d9, d8, [sp, #32]
	ldp d11, d10, [sp, #16]
	ldp d13, d12, [sp], #144
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
	b linear_srgb::simd::gamma_to_linear_premultiply_rgba_slice_tier_scalar
