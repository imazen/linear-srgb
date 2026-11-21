.section .text.stub_gamma_to_linear_premultiply_rgba_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_premultiply_rgba_slice
	.p2align	2
.type	stub_gamma_to_linear_premultiply_rgba_slice,@function
stub_gamma_to_linear_premultiply_rgba_slice:
	.cfi_startproc
	stp d15, d14, [sp, #-128]!
	.cfi_def_cfa_offset 128
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	stp x29, x30, [sp, #64]
	str x28, [sp, #80]
	stp x22, x21, [sp, #96]
	stp x20, x19, [sp, #112]
	add x29, sp, #64
	.cfi_def_cfa w29, 64
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w28, -48
	.cfi_offset w30, -56
	.cfi_offset w29, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	.cfi_offset b10, -88
	.cfi_offset b11, -96
	.cfi_offset b12, -104
	.cfi_offset b13, -112
	.cfi_offset b14, -120
	.cfi_offset b15, -128
	.cfi_remember_state
	sub sp, sp, #464
	str s0, [x29, #28]
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB8_24
	cmp w8, #2
	b.ne .LBB8_23
.LBB8_2:
	ldr s0, [x29, #28]
	fcmp s0, #0.0
	b.lt .LBB8_10
	fmov s1, #4.00000000
	ldr s0, [x29, #28]
	fcmp s0, s1
	b.hi .LBB8_10
	ldr s0, [x29, #28]
	lsl x8, x1, #2
	and x9, x1, #0x1ffffffffffffff0
	add x19, x0, x9, lsl #2
	fcmp s0, #0.0
	and x9, x8, #0x7fffffffffffffc0
	add x8, x0, x9
	b.ne .LBB8_15
	cbz x9, .LBB8_8
	adrp x9, .LCPI8_1
	ldr q0, [x9, :lo12:.LCPI8_1]
.LBB8_7:
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
	bic v4.16b, v1.16b, v4.16b
	and v3.16b, v3.16b, v0.16b
	and v7.16b, v7.16b, v0.16b
	bic v16.16b, v2.16b, v16.16b
	bic v18.16b, v5.16b, v18.16b
	and v17.16b, v17.16b, v0.16b
	and v19.16b, v19.16b, v0.16b
	orr v3.16b, v3.16b, v4.16b
	bic v4.16b, v6.16b, v20.16b
	orr v7.16b, v7.16b, v16.16b
	orr v16.16b, v17.16b, v18.16b
	orr v4.16b, v19.16b, v4.16b
	fmul v3.4s, v3.4s, v1.s[3]
	fmul v7.4s, v7.4s, v2.s[3]
	fmul v16.4s, v16.4s, v5.s[3]
	fmul v4.4s, v4.4s, v6.s[3]
	mov v3.s[3], v1.s[3]
	mov v7.s[3], v2.s[3]
	mov v16.s[3], v5.s[3]
	mov v4.s[3], v6.s[3]
	stp q3, q7, [x0]
	stp q16, q4, [x0, #32]
	add x0, x0, #64
	cmp x0, x8
	b.ne .LBB8_7
.LBB8_8:
	ands x8, x1, #0xc
	b.eq .LBB8_22
	lsl x20, x8, #2
	mov w8, #64
	mov w1, wzr
	sub x2, x8, x20
	sub x8, x29, #128
	add x0, x8, x20
	bl memset
	sub x0, x29, #128
	mov x1, x19
	mov x2, x20
	bl memcpy
	ldp q0, q1, [x29, #-128]
	adrp x8, .LCPI8_1
	ldp q2, q3, [x29, #-96]
	ldr q16, [x8, :lo12:.LCPI8_1]
	fcmgt v4.4s, v0.4s, #0.0
	fcmeq v5.4s, v0.4s, v0.4s
	fcmgt v6.4s, v1.4s, #0.0
	fcmeq v7.4s, v1.4s, v1.4s
	fcmgt v17.4s, v2.4s, #0.0
	fcmeq v18.4s, v2.4s, v2.4s
	fcmgt v19.4s, v3.4s, #0.0
	fcmeq v20.4s, v3.4s, v3.4s
	bic v5.16b, v0.16b, v5.16b
	and v4.16b, v4.16b, v16.16b
	and v6.16b, v6.16b, v16.16b
	bic v7.16b, v1.16b, v7.16b
	bic v18.16b, v2.16b, v18.16b
	and v17.16b, v17.16b, v16.16b
	and v16.16b, v19.16b, v16.16b
	orr v4.16b, v4.16b, v5.16b
	bic v5.16b, v3.16b, v20.16b
	orr v6.16b, v6.16b, v7.16b
	orr v7.16b, v17.16b, v18.16b
	orr v5.16b, v16.16b, v5.16b
	fmul v4.4s, v4.4s, v0.s[3]
	fmul v6.4s, v6.4s, v1.s[3]
	fmul v7.4s, v7.4s, v2.s[3]
	fmul v5.4s, v5.4s, v3.s[3]
	mov v4.s[3], v0.s[3]
	mov v6.s[3], v1.s[3]
	mov v7.s[3], v2.s[3]
	mov v5.s[3], v3.s[3]
	stp q4, q6, [x29, #-128]
	stp q7, q5, [x29, #-96]
	b .LBB8_20
.LBB8_10:
	lsl x8, x1, #2
	mov x20, x0
	mov x19, x1
	ands x8, x8, #0x7fffffffffffffc0
	b.eq .LBB8_13
	mov x22, x20
	add x21, x20, x8
.LBB8_12:
	ldp q0, q2, [x22]
	sub x8, x29, #192
	ldr q1, [x22, #32]
	sub x0, x29, #128
	stp q0, q2, [x29, #-128]
	stp q1, q0, [sp, #224]
	ldr q0, [x22, #48]
	str q2, [sp, #208]
	str q0, [sp, #256]
	stp q1, q0, [x29, #-96]
	ldr s0, [x29, #28]
	bl linear_srgb::tf::math::gamma_outside_fit::<16>
	ldp q0, q1, [x29, #-192]
	ldr q4, [sp, #208]
	ldp q7, q6, [sp, #224]
	ldr q5, [sp, #256]
	ldp q2, q3, [x29, #-160]
	fmul v1.4s, v1.4s, v4.s[3]
	fmul v0.4s, v0.4s, v6.s[3]
	fmul v2.4s, v2.4s, v7.s[3]
	fmul v3.4s, v3.4s, v5.s[3]
	mov v1.s[3], v4.s[3]
	mov v0.s[3], v6.s[3]
	mov v2.s[3], v7.s[3]
	mov v3.s[3], v5.s[3]
	stp q0, q1, [x22]
	stp q2, q3, [x22, #32]
	add x22, x22, #64
	cmp x22, x21
	b.ne .LBB8_12
.LBB8_13:
	ands x8, x19, #0xc
	b.eq .LBB8_22
	and x9, x19, #0x1ffffffffffffff0
	lsl x19, x8, #2
	mov w8, #64
	mov w1, wzr
	add x20, x20, x9, lsl #2
	sub x2, x8, x19
	sub x8, x29, #256
	add x0, x8, x19
	bl memset
	sub x0, x29, #256
	mov x1, x20
	mov x2, x19
	bl memcpy
	ldp q0, q2, [x29, #-256]
	sub x8, x29, #192
	ldur q1, [x29, #-224]
	sub x0, x29, #128
	stp q0, q2, [x29, #-128]
	stp q1, q0, [sp, #224]
	ldur q0, [x29, #-208]
	str q2, [sp, #208]
	str q0, [sp, #256]
	stp q1, q0, [x29, #-96]
	ldr s0, [x29, #28]
	bl linear_srgb::tf::math::gamma_outside_fit::<16>
	ldp q0, q1, [x29, #-192]
	sub x1, x29, #256
	ldp q7, q6, [sp, #224]
	mov x0, x20
	ldp q2, q3, [x29, #-160]
	mov x2, x19
	ldr q4, [sp, #208]
	ldr q5, [sp, #256]
	fmul v0.4s, v0.4s, v6.s[3]
	fmul v1.4s, v1.4s, v4.s[3]
	fmul v2.4s, v2.4s, v7.s[3]
	fmul v3.4s, v3.4s, v5.s[3]
	mov v0.s[3], v6.s[3]
	mov v1.s[3], v4.s[3]
	mov v2.s[3], v7.s[3]
	mov v3.s[3], v5.s[3]
	stp q0, q1, [x29, #-256]
	stp q2, q3, [x29, #-224]
	b .LBB8_21
.LBB8_15:
	adrp x21, .LCPI8_0
	cbz x9, .LBB8_18
	fmov s0, #4.00000000
	ldr s1, [x29, #28]
	mov w9, #64269
	mov w10, #64269
	movk w9, #74, lsl #16
	fmov v5.4s, #1.00000000
	movk w10, #16330, lsl #16
	dup v2.4s, w9
	mov w9, #1267
	fcmp s1, s0
	movk w9, #16181, lsl #16
	fcsel s0, s0, s1, gt
	dup v1.4s, w10
	mov w10, #27455
	movk w10, #16094, lsl #16
	stp q1, q2, [sp, #208]
	dup v1.4s, w10
	mov w10, #46233
	movk w10, #16147, lsl #16
	dup v2.4s, w9
	mov w9, #8388607
	dup v6.4s, w10
	fmov w10, s0
	fmov s3, w9
	stp q1, q2, [sp, #176]
	movi d1, #0000000000000000
	sub w10, w10, #1
	cmp w10, w9
	fmov s2, w10
	mov w10, #14483
	movk w10, #16246, lsl #16
	mov w9, #43579
	dup v4.4s, w10
	movk w9, #16440, lsl #16
	mov w10, #-1023672320
	cmhi v2.4s, v3.4s, v2.4s
	dup v21.4s, w10
	mov w10, #63802
	movk w10, #14625, lsl #16
	fcsel s0, s1, s0, lo
	stp q4, q6, [sp, #144]
	dup v4.4s, w9
	mov w9, #1123942400
	dup v3.4s, w9
	mov w9, #50297
	and v2.16b, v2.16b, v5.16b
	movk w9, #15022, lsl #16
	stp q3, q4, [sp, #112]
	dup v4.4s, w10
	dup v3.4s, w9
	mov w10, #38156
	mov w9, #22716
	dup v2.4s, v2.s[0]
	movk w10, #15389, lsl #16
	movk w9, #15715, lsl #16
	stp q3, q4, [sp, #80]
	dup v4.4s, w10
	mov w10, #65005
	dup v3.4s, w9
	mov w9, #29208
	movk w10, #15989, lsl #16
	movk w9, #16177, lsl #16
	dup v29.4s, w10
	stp q0, q2, [sp, #16]
	dup v30.4s, w9
	ldr q0, [x21, :lo12:.LCPI8_0]
	stp q3, q4, [sp, #48]
	str q0, [sp]
.LBB8_17:
	movi v7.4s, #128, lsl #16
	ldp q9, q8, [x0]
	ldp q19, q18, [sp, #208]
	movi v22.4s, #127, msl #16
	ldp q10, q11, [x0, #32]
	fmov v23.4s, #-1.00000000
	ldr q20, [sp, #192]
	mvni v13.4s, #126
	ldr q27, [sp, #144]
	fcmge v0.4s, v9.4s, v7.4s
	fcmge v1.4s, v8.4s, v7.4s
	mvni v25.4s, #126
	fcmge v3.4s, v10.4s, v7.4s
	mov v31.16b, v27.16b
	and v0.16b, v0.16b, v9.16b
	and v1.16b, v1.16b, v8.16b
	and v4.16b, v3.16b, v10.16b
	fminnm v0.4s, v0.4s, v5.4s
	fminnm v1.4s, v1.4s, v5.4s
	fminnm v4.4s, v4.4s, v5.4s
	cmeq v2.4s, v0.4s, #0
	add v0.4s, v0.4s, v18.4s
	cmeq v6.4s, v1.4s, #0
	add v1.4s, v1.4s, v18.4s
	cmeq v14.4s, v4.4s, #0
	add v4.4s, v4.4s, v18.4s
	bit v0.16b, v19.16b, v2.16b
	stp q6, q2, [sp, #240]
	mov v3.16b, v6.16b
	bit v4.16b, v19.16b, v14.16b
	bsl v3.16b, v19.16b, v1.16b
	and v2.16b, v0.16b, v22.16b
	ssra v13.4s, v0.4s, #23
	mov v0.16b, v27.16b
	and v17.16b, v4.16b, v22.16b
	ssra v25.4s, v4.4s, #23
	add v2.4s, v2.4s, v20.4s
	and v6.16b, v3.16b, v22.16b
	scvtf v13.4s, v13.4s
	fadd v1.4s, v2.4s, v23.4s
	fadd v2.4s, v2.4s, v5.4s
	add v6.4s, v6.4s, v20.4s
	fdiv v1.4s, v1.4s, v2.4s
	fcmge v2.4s, v11.4s, v7.4s
	fadd v7.4s, v6.4s, v23.4s
	fadd v6.4s, v6.4s, v5.4s
	and v2.16b, v2.16b, v11.16b
	fminnm v16.4s, v2.4s, v5.4s
	fdiv v2.4s, v7.4s, v6.4s
	add v7.4s, v17.4s, v20.4s
	cmeq v15.4s, v16.4s, #0
	add v6.4s, v16.4s, v18.4s
	fadd v16.4s, v7.4s, v23.4s
	fadd v7.4s, v7.4s, v5.4s
	bit v6.16b, v19.16b, v15.16b
	ldp q19, q28, [sp, #160]
	mov v26.16b, v19.16b
	and v17.16b, v6.16b, v22.16b
	mvni v22.4s, #126
	fdiv v7.4s, v16.4s, v7.4s
	add v16.4s, v17.4s, v20.4s
	mov v20.16b, v19.16b
	ssra v22.4s, v3.4s, #23
	mov v3.16b, v27.16b
	fmul v18.4s, v2.4s, v2.4s
	fadd v17.4s, v16.4s, v23.4s
	fadd v16.4s, v16.4s, v5.4s
	mov v23.16b, v19.16b
	scvtf v22.4s, v22.4s
	fmla v20.4s, v28.4s, v18.4s
	fdiv v17.4s, v17.4s, v16.4s
	fmul v16.4s, v1.4s, v1.4s
	fmla v0.4s, v18.4s, v20.4s
	mov v20.16b, v19.16b
	ldr q19, [sp, #128]
	fmul v12.4s, v7.4s, v7.4s
	mov v4.16b, v19.16b
	fmla v26.4s, v28.4s, v16.4s
	fmla v23.4s, v28.4s, v12.4s
	fmla v31.4s, v16.4s, v26.4s
	mvni v26.4s, #126
	fmla v3.4s, v12.4s, v23.4s
	mov v23.16b, v19.16b
	ssra v26.4s, v6.4s, #23
	fmla v4.4s, v16.4s, v31.4s
	scvtf v6.4s, v25.4s
	fmul v24.4s, v17.4s, v17.4s
	fmla v23.4s, v18.4s, v0.4s
	mov v0.16b, v19.16b
	fmla v13.4s, v1.4s, v4.4s
	mov v1.16b, v19.16b
	ldr q4, [sp, #16]
	fmla v20.4s, v28.4s, v24.4s
	fmla v0.4s, v12.4s, v3.4s
	fmla v22.4s, v2.4s, v23.4s
	scvtf v2.4s, v26.4s
	movi v26.4s, #67, lsl #24
	ldp q31, q28, [sp, #64]
	fmla v27.4s, v24.4s, v20.4s
	fmla v6.4s, v7.4s, v0.4s
	fmul v0.4s, v13.4s, v4.s[0]
	fmul v16.4s, v22.4s, v4.s[0]
	mov v25.16b, v31.16b
	fmov v13.4s, #1.00000000
	fmla v1.4s, v24.4s, v27.4s
	ldp q27, q23, [sp, #96]
	fmaxnm v3.4s, v16.4s, v21.4s
	mov v24.16b, v28.16b
	fmla v2.4s, v17.4s, v1.4s
	fmul v17.4s, v6.4s, v4.s[0]
	fmaxnm v1.4s, v0.4s, v21.4s
	fminnm v19.4s, v3.4s, v26.4s
	fmul v7.4s, v2.4s, v4.s[0]
	fmaxnm v2.4s, v17.4s, v21.4s
	fminnm v1.4s, v1.4s, v26.4s
	frintn v18.4s, v19.4s
	fmaxnm v3.4s, v7.4s, v21.4s
	fminnm v2.4s, v2.4s, v26.4s
	frintn v4.4s, v1.4s
	fminnm v20.4s, v3.4s, v26.4s
	frintn v3.4s, v2.4s
	fminnm v6.4s, v4.4s, v23.4s
	fminnm v4.4s, v18.4s, v23.4s
	frintn v22.4s, v20.4s
	fminnm v3.4s, v3.4s, v23.4s
	fsub v18.4s, v1.4s, v6.4s
	fsub v19.4s, v19.4s, v4.4s
	fcvtns v6.4s, v6.4s
	fcvtns v4.4s, v4.4s
	fminnm v12.4s, v22.4s, v23.4s
	mov v22.16b, v28.16b
	mov v23.16b, v28.16b
	fsub v2.4s, v2.4s, v3.4s
	fcvtns v3.4s, v3.4s
	shl v6.4s, v6.4s, #23
	shl v4.4s, v4.4s, #23
	fmla v22.4s, v27.4s, v18.4s
	fmla v23.4s, v27.4s, v19.4s
	fsub v1.4s, v20.4s, v12.4s
	mov v20.16b, v31.16b
	add v6.4s, v6.4s, v5.4s
	fmla v24.4s, v27.4s, v2.4s
	shl v3.4s, v3.4s, #23
	add v4.4s, v4.4s, v5.4s
	fmla v20.4s, v18.4s, v22.4s
	mov v22.16b, v28.16b
	fmla v25.4s, v19.4s, v23.4s
	mov v23.16b, v31.16b
	ldr q28, [sp, #48]
	add v3.4s, v3.4s, v5.4s
	fmla v22.4s, v27.4s, v1.4s
	mov v27.16b, v28.16b
	fmla v23.4s, v2.4s, v24.4s
	mov v24.16b, v28.16b
	fmla v27.4s, v18.4s, v20.4s
	mov v20.16b, v31.16b
	fmla v24.4s, v19.4s, v25.4s
	mov v25.16b, v28.16b
	fmla v20.4s, v1.4s, v22.4s
	mov v22.16b, v29.16b
	fmla v25.4s, v2.4s, v23.4s
	mov v23.16b, v29.16b
	fmla v22.4s, v18.4s, v27.4s
	mov v27.16b, v28.16b
	fmla v23.4s, v19.4s, v24.4s
	mov v24.16b, v29.16b
	fmla v27.4s, v1.4s, v20.4s
	mov v20.16b, v30.16b
	fmla v24.4s, v2.4s, v25.4s
	mov v25.16b, v30.16b
	fmla v20.4s, v18.4s, v22.4s
	mov v22.16b, v29.16b
	fmla v25.4s, v19.4s, v23.4s
	fmov v23.4s, #1.00000000
	fmla v22.4s, v1.4s, v27.4s
	mov v27.16b, v30.16b
	fmla v13.4s, v18.4s, v20.4s
	mov v18.16b, v30.16b
	fcmge v20.4s, v0.4s, v26.4s
	fmla v23.4s, v19.4s, v25.4s
	fcvtns v25.4s, v12.4s
	fmov v19.4s, #1.00000000
	fmla v27.4s, v2.4s, v24.4s
	fmov v24.4s, #1.00000000
	fcmgt v0.4s, v21.4s, v0.4s
	fmla v18.4s, v1.4s, v22.4s
	fcmge v22.4s, v16.4s, v26.4s
	fcmgt v16.4s, v21.4s, v16.4s
	fmul v6.4s, v13.4s, v6.4s
	shl v25.4s, v25.4s, #23
	fmla v24.4s, v2.4s, v27.4s
	fcmge v2.4s, v17.4s, v26.4s
	fcmge v27.4s, v7.4s, v26.4s
	fmla v19.4s, v1.4s, v18.4s
	mvn v1.16b, v20.16b
	mvn v18.16b, v22.16b
	add v25.4s, v25.4s, v5.4s
	fcmgt v17.4s, v21.4s, v17.4s
	fcmgt v7.4s, v21.4s, v7.4s
	mvn v12.16b, v2.16b
	bic v0.16b, v1.16b, v0.16b
	fmul v1.4s, v23.4s, v4.4s
	mvn v4.16b, v27.16b
	bic v16.16b, v18.16b, v16.16b
	fmul v18.4s, v19.4s, v25.4s
	ldr q19, [sp]
	fmul v3.4s, v24.4s, v3.4s
	bic v17.16b, v12.16b, v17.16b
	and v0.16b, v0.16b, v6.16b
	bic v4.16b, v4.16b, v7.16b
	and v6.16b, v20.16b, v19.16b
	and v1.16b, v16.16b, v1.16b
	and v7.16b, v22.16b, v19.16b
	and v2.16b, v2.16b, v19.16b
	ldr q16, [sp, #32]
	and v3.16b, v17.16b, v3.16b
	and v4.16b, v4.16b, v18.16b
	orr v0.16b, v0.16b, v6.16b
	and v6.16b, v27.16b, v19.16b
	orr v1.16b, v1.16b, v7.16b
	fcmeq v7.4s, v10.4s, v10.4s
	orr v2.16b, v3.16b, v2.16b
	orr v3.16b, v4.16b, v6.16b
	ldp q6, q4, [sp, #240]
	bit v2.16b, v16.16b, v14.16b
	bit v0.16b, v16.16b, v4.16b
	fcmeq v4.4s, v9.4s, v9.4s
	bit v1.16b, v16.16b, v6.16b
	fcmeq v6.4s, v8.4s, v8.4s
	bit v3.16b, v16.16b, v15.16b
	fcmeq v16.4s, v11.4s, v11.4s
	bif v2.16b, v10.16b, v7.16b
	bif v0.16b, v9.16b, v4.16b
	bif v1.16b, v8.16b, v6.16b
	bif v3.16b, v11.16b, v16.16b
	fmul v2.4s, v2.4s, v10.s[3]
	fmul v0.4s, v0.4s, v9.s[3]
	fmul v1.4s, v1.4s, v8.s[3]
	fmul v3.4s, v3.4s, v11.s[3]
	mov v2.s[3], v10.s[3]
	mov v0.s[3], v9.s[3]
	mov v1.s[3], v8.s[3]
	mov v3.s[3], v11.s[3]
	stp q0, q1, [x0]
	stp q2, q3, [x0, #32]
	add x0, x0, #64
	cmp x0, x8
	b.ne .LBB8_17
.LBB8_18:
	ands x8, x1, #0xc
	b.eq .LBB8_22
	lsl x20, x8, #2
	mov w8, #64
	mov w1, wzr
	ldr s8, [x29, #28]
	sub x2, x8, x20
	sub x8, x29, #128
	add x0, x8, x20
	bl memset
	sub x0, x29, #128
	mov x1, x19
	mov x2, x20
	bl memcpy
	movi v7.4s, #128, lsl #16
	ldp q1, q6, [x29, #-128]
	fmov v4.4s, #1.00000000
	mov w8, #64269
	movi v22.4s, #127, msl #16
	movk w8, #74, lsl #16
	fmov v26.4s, #-1.00000000
	mov w9, #27455
	dup v19.4s, w8
	mov w8, #64269
	movk w9, #16094, lsl #16
	fcmge v2.4s, v1.4s, v7.4s
	fcmge v5.4s, v6.4s, v7.4s
	movk w8, #16330, lsl #16
	dup v20.4s, w8
	mov w8, #1267
	fmov s31, #4.00000000
	movk w8, #16181, lsl #16
	mvni v12.4s, #126
	mvni v11.4s, #126
	dup v25.4s, w8
	mov w8, #46233
	movi d9, #0000000000000000
	and v2.16b, v2.16b, v1.16b
	and v5.16b, v5.16b, v6.16b
	movk w8, #16147, lsl #16
	fcmp s8, s31
	mov w10, #-1023672320
	fminnm v3.4s, v2.4s, v4.4s
	fminnm v16.4s, v5.4s, v4.4s
	cmeq v0.4s, v3.4s, #0
	add v3.4s, v3.4s, v19.4s
	cmeq v2.4s, v16.4s, #0
	add v16.4s, v16.4s, v19.4s
	mov v18.16b, v0.16b
	mov v24.16b, v2.16b
	str q0, [sp, #256]
	bsl v18.16b, v20.16b, v3.16b
	ldp q3, q5, [x29, #-96]
	bsl v24.16b, v20.16b, v16.16b
	fcmge v21.4s, v3.4s, v7.4s
	and v17.16b, v18.16b, v22.16b
	ssra v11.4s, v18.4s, #23
	fcsel s18, s31, s8, gt
	and v27.16b, v24.16b, v22.16b
	ssra v12.4s, v24.4s, #23
	mvni v31.4s, #126
	add v17.4s, v17.4s, v25.4s
	and v21.16b, v21.16b, v3.16b
	add v27.4s, v27.4s, v25.4s
	fadd v16.4s, v17.4s, v26.4s
	fadd v17.4s, v17.4s, v4.4s
	fminnm v23.4s, v21.4s, v4.4s
	fdiv v21.4s, v16.4s, v17.4s
	fcmge v16.4s, v5.4s, v7.4s
	cmeq v7.4s, v23.4s, #0
	add v17.4s, v23.4s, v19.4s
	mov v23.16b, v7.16b
	and v16.16b, v16.16b, v5.16b
	bsl v23.16b, v20.16b, v17.16b
	fadd v17.4s, v27.4s, v26.4s
	fadd v27.4s, v27.4s, v4.4s
	fminnm v28.4s, v16.4s, v4.4s
	and v29.16b, v23.16b, v22.16b
	fdiv v17.4s, v17.4s, v27.4s
	ssra v31.4s, v23.4s, #23
	cmeq v16.4s, v28.4s, #0
	add v19.4s, v28.4s, v19.4s
	dup v28.4s, w9
	mov w9, #43579
	add v27.4s, v29.4s, v25.4s
	movk w9, #16440, lsl #16
	bit v19.16b, v20.16b, v16.16b
	dup v14.4s, w9
	fmov w9, s18
	fadd v20.4s, v27.4s, v26.4s
	fadd v27.4s, v27.4s, v4.4s
	mov v24.16b, v14.16b
	sub w9, w9, #1
	mov v23.16b, v14.16b
	and v22.16b, v19.16b, v22.16b
	fdiv v20.4s, v20.4s, v27.4s
	fmul v27.4s, v21.4s, v21.4s
	add v22.4s, v22.4s, v25.4s
	fmul v29.4s, v17.4s, v17.4s
	fadd v25.4s, v22.4s, v26.4s
	fadd v22.4s, v22.4s, v4.4s
	dup v26.4s, w8
	mov w8, #14483
	movk w8, #16246, lsl #16
	mov v30.16b, v26.16b
	mov v10.16b, v26.16b
	mov v8.16b, v26.16b
	fdiv v22.4s, v25.4s, v22.4s
	dup v25.4s, w8
	fmla v30.4s, v28.4s, v27.4s
	fmla v10.4s, v28.4s, v29.4s
	mov w8, #8388607
	cmp w9, w8
	mov v13.16b, v25.16b
	mov v15.16b, v25.16b
	fcsel s9, s9, s18, lo
	fmla v13.4s, v27.4s, v30.4s
	fmul v30.4s, v20.4s, v20.4s
	fmla v15.4s, v29.4s, v10.4s
	scvtf v10.4s, v11.4s
	mov v11.16b, v25.16b
	fmla v24.4s, v27.4s, v13.4s
	mov v27.16b, v14.16b
	fmla v8.4s, v28.4s, v30.4s
	fmul v0.4s, v22.4s, v22.4s
	fmla v27.4s, v29.4s, v15.4s
	mvni v29.4s, #126
	fmla v10.4s, v21.4s, v24.4s
	fmla v11.4s, v30.4s, v8.4s
	scvtf v24.4s, v31.4s
	movi v21.4s, #67, lsl #24
	ssra v29.4s, v19.4s, #23
	fmla v26.4s, v28.4s, v0.4s
	scvtf v28.4s, v12.4s
	fmul v18.4s, v10.4s, v9.s[0]
	fmla v23.4s, v30.4s, v11.4s
	fmla v25.4s, v0.4s, v26.4s
	fmla v28.4s, v17.4s, v27.4s
	dup v17.4s, w10
	scvtf v26.4s, v29.4s
	fmla v24.4s, v20.4s, v23.4s
	mov w10, #1123942400
	fmla v14.4s, v0.4s, v25.4s
	fmul v19.4s, v28.4s, v9.s[0]
	fmaxnm v0.4s, v18.4s, v17.4s
	fmla v26.4s, v22.4s, v14.4s
	fmaxnm v23.4s, v19.4s, v17.4s
	fminnm v0.4s, v0.4s, v21.4s
	fmul v22.4s, v24.4s, v9.s[0]
	fmul v20.4s, v26.4s, v9.s[0]
	fminnm v25.4s, v23.4s, v21.4s
	frintn v23.4s, v0.4s
	dup v26.4s, w10
	fmaxnm v27.4s, v22.4s, v17.4s
	mov w10, #50297
	movk w10, #15022, lsl #16
	dup v9.4s, w10
	mov w10, #63802
	fmaxnm v28.4s, v20.4s, v17.4s
	frintn v29.4s, v25.4s
	fminnm v24.4s, v23.4s, v26.4s
	fminnm v31.4s, v27.4s, v21.4s
	movk w10, #14625, lsl #16
	mov v12.16b, v9.16b
	mov v15.16b, v9.16b
	fminnm v8.4s, v28.4s, v21.4s
	fminnm v23.4s, v29.4s, v26.4s
	fsub v30.4s, v0.4s, v24.4s
	dup v0.4s, w10
	mov w10, #38156
	frintn v27.4s, v31.4s
	movk w10, #15389, lsl #16
	fcvtns v24.4s, v24.4s
	dup v10.4s, w10
	mov w10, #22716
	frintn v11.4s, v8.4s
	fsub v29.4s, v25.4s, v23.4s
	mov v25.16b, v9.16b
	fminnm v28.4s, v27.4s, v26.4s
	movk w10, #15715, lsl #16
	fcvtns v23.4s, v23.4s
	dup v13.4s, w10
	mov v14.16b, v10.16b
	mov w10, #65005
	fmla v25.4s, v0.4s, v30.4s
	movk w10, #15989, lsl #16
	shl v24.4s, v24.4s, #23
	fminnm v27.4s, v11.4s, v26.4s
	fmla v12.4s, v0.4s, v29.4s
	mov v11.16b, v10.16b
	fsub v26.4s, v31.4s, v28.4s
	fcvtns v28.4s, v28.4s
	shl v23.4s, v23.4s, #23
	add v24.4s, v24.4s, v4.4s
	fmla v11.4s, v30.4s, v25.4s
	fsub v25.4s, v8.4s, v27.4s
	fmla v14.4s, v29.4s, v12.4s
	mov v12.16b, v13.16b
	dup v8.4s, w10
	fmla v15.4s, v0.4s, v26.4s
	mov w10, #29208
	movk w10, #16177, lsl #16
	fcvtns v27.4s, v27.4s
	shl v28.4s, v28.4s, #23
	fmla v12.4s, v30.4s, v11.4s
	dup v31.4s, w10
	mov v11.16b, v13.16b
	fmla v9.4s, v0.4s, v25.4s
	mov v0.16b, v8.16b
	add v23.4s, v23.4s, v4.4s
	fmla v11.4s, v29.4s, v14.4s
	mov v14.16b, v8.16b
	fmla v0.4s, v30.4s, v12.4s
	mov v12.16b, v10.16b
	fmla v10.4s, v25.4s, v9.4s
	fmov v9.4s, #1.00000000
	fmla v12.4s, v26.4s, v15.4s
	mov v15.16b, v31.16b
	fmla v14.4s, v29.4s, v11.4s
	fmov v11.4s, #1.00000000
	fmla v15.4s, v30.4s, v0.4s
	mov v0.16b, v13.16b
	fmla v13.4s, v25.4s, v10.4s
	fmov v10.4s, #1.00000000
	fmla v0.4s, v26.4s, v12.4s
	mov v12.16b, v31.16b
	fmla v9.4s, v30.4s, v15.4s
	fmov v30.4s, #1.00000000
	fmla v12.4s, v29.4s, v14.4s
	mov v14.16b, v8.16b
	fmla v8.4s, v25.4s, v13.4s
	fcmge v13.4s, v19.4s, v21.4s
	fcmgt v19.4s, v17.4s, v19.4s
	fmul v24.4s, v9.4s, v24.4s
	fmla v14.4s, v26.4s, v0.4s
	mov v0.16b, v31.16b
	fmla v11.4s, v29.4s, v12.4s
	fmov s29, w9
	fmov s12, w8
	fmla v31.4s, v25.4s, v8.4s
	fcmge v8.4s, v22.4s, v21.4s
	fcmgt v22.4s, v17.4s, v22.4s
	fmla v0.4s, v26.4s, v14.4s
	cmhi v29.4s, v12.4s, v29.4s
	fcmge v12.4s, v18.4s, v21.4s
	fcmge v21.4s, v20.4s, v21.4s
	fcmgt v18.4s, v17.4s, v18.4s
	fcmgt v17.4s, v17.4s, v20.4s
	shl v20.4s, v27.4s, #23
	fmla v30.4s, v25.4s, v31.4s
	add v27.4s, v28.4s, v4.4s
	mvn v25.16b, v8.16b
	fmla v10.4s, v26.4s, v0.4s
	mvn v0.16b, v12.16b
	mvn v26.16b, v13.16b
	add v20.4s, v20.4s, v4.4s
	mvn v28.16b, v21.16b
	and v4.16b, v29.16b, v4.16b
	bic v22.16b, v25.16b, v22.16b
	bic v0.16b, v0.16b, v18.16b
	fmul v18.4s, v11.4s, v23.4s
	bic v19.16b, v26.16b, v19.16b
	fmul v26.4s, v10.4s, v27.4s
	fmul v20.4s, v30.4s, v20.4s
	bic v17.16b, v28.16b, v17.16b
	ldr q23, [x21, :lo12:.LCPI8_0]
	dup v4.4s, v4.s[0]
	and v0.16b, v0.16b, v24.16b
	and v25.16b, v12.16b, v23.16b
	and v24.16b, v13.16b, v23.16b
	and v18.16b, v19.16b, v18.16b
	and v19.16b, v8.16b, v23.16b
	and v22.16b, v22.16b, v26.16b
	and v21.16b, v21.16b, v23.16b
	and v17.16b, v17.16b, v20.16b
	fcmeq v20.4s, v1.4s, v1.4s
	orr v0.16b, v0.16b, v25.16b
	orr v18.16b, v18.16b, v24.16b
	orr v19.16b, v22.16b, v19.16b
	orr v17.16b, v17.16b, v21.16b
	ldr q21, [sp, #256]
	bsl v2.16b, v4.16b, v18.16b
	fcmeq v18.4s, v3.4s, v3.4s
	bit v0.16b, v4.16b, v21.16b
	fcmeq v21.4s, v6.4s, v6.4s
	bsl v7.16b, v4.16b, v19.16b
	fcmeq v19.4s, v5.4s, v5.4s
	bif v4.16b, v17.16b, v16.16b
	bif v0.16b, v1.16b, v20.16b
	bif v2.16b, v6.16b, v21.16b
	bif v7.16b, v3.16b, v18.16b
	bif v4.16b, v5.16b, v19.16b
	fmul v0.4s, v0.4s, v1.s[3]
	fmul v2.4s, v2.4s, v6.s[3]
	fmul v7.4s, v7.4s, v3.s[3]
	fmul v4.4s, v4.4s, v5.s[3]
	mov v0.s[3], v1.s[3]
	mov v2.s[3], v6.s[3]
	mov v7.s[3], v3.s[3]
	mov v4.s[3], v5.s[3]
	stp q0, q2, [x29, #-128]
	stp q7, q4, [x29, #-96]
.LBB8_20:
	sub x1, x29, #128
	mov x0, x19
	mov x2, x20
.LBB8_21:
	bl memcpy
.LBB8_22:
	add sp, sp, #464
	.cfi_def_cfa wsp, 128
	ldp x20, x19, [sp, #112]
	ldr x28, [sp, #80]
	ldp x22, x21, [sp, #96]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #128
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
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
.LBB8_23:
	.cfi_restore_state
	mov x20, x1
	mov x19, x0
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB8_2
.LBB8_24:
	ldr s0, [x29, #28]
	add sp, sp, #464
	.cfi_def_cfa wsp, 128
	ldp x20, x19, [sp, #112]
	ldr x28, [sp, #80]
	ldp x22, x21, [sp, #96]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #128
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
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
