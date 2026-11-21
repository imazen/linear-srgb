.section .text.stub_unpremultiply_linear_to_gamma_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_gamma_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_gamma_rgba_slice,@function
stub_unpremultiply_linear_to_gamma_rgba_slice:
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
	sub sp, sp, #544
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB18_23
	cmp w8, #2
	b.ne .LBB18_22
.LBB18_2:
	fmov s1, #1.00000000
	fdiv s8, s1, s0
	fmov s0, #4.00000000
	fcmp s8, #0.0
	fccmp s8, s0, #2, ge
	b.ls .LBB18_8
	lsl x8, x1, #2
	mov x20, x0
	mov x19, x1
	ands x8, x8, #0x7fffffffffffffc0
	b.eq .LBB18_6
	mov w9, #981467136
	mov x22, x20
	add x21, x20, x8
	dup v0.4s, w9
	str q0, [sp, #272]
.LBB18_5:
	ldp q17, q18, [x22]
	fmov v7.4s, #1.00000000
	ldp q20, q19, [x22, #32]
	sub x8, x29, #192
	ldr q16, [sp, #272]
	sub x0, x29, #128
	dup v0.4s, v17.s[3]
	dup v2.4s, v18.s[3]
	stp q17, q18, [sp, #288]
	dup v4.4s, v20.s[3]
	dup v6.4s, v19.s[3]
	stp q20, q19, [sp, #320]
	fdiv v1.4s, v7.4s, v0.4s
	fcmgt v0.4s, v0.4s, v16.4s
	fdiv v3.4s, v7.4s, v2.4s
	fcmgt v2.4s, v2.4s, v16.4s
	and v0.16b, v0.16b, v1.16b
	fmul v0.4s, v17.4s, v0.4s
	fdiv v5.4s, v7.4s, v4.4s
	fcmgt v4.4s, v4.4s, v16.4s
	and v1.16b, v2.16b, v3.16b
	fmul v1.4s, v18.4s, v1.4s
	fdiv v7.4s, v7.4s, v6.4s
	fcmgt v6.4s, v6.4s, v16.4s
	stp q0, q1, [x29, #-128]
	fmov s0, s8
	and v2.16b, v4.16b, v5.16b
	fmul v2.4s, v20.4s, v2.4s
	and v3.16b, v6.16b, v7.16b
	fmul v3.4s, v19.4s, v3.4s
	stp q2, q3, [x29, #-96]
	bl linear_srgb::tf::math::gamma_outside_fit::<16>
	ldp q0, q1, [x29, #-192]
	ldp q2, q4, [sp, #288]
	mov v0.s[3], v2.s[3]
	ldp q2, q3, [x29, #-160]
	mov v1.s[3], v4.s[3]
	ldp q5, q4, [sp, #320]
	mov v2.s[3], v5.s[3]
	mov v3.s[3], v4.s[3]
	stp q0, q1, [x22]
	stp q2, q3, [x22, #32]
	add x22, x22, #64
	cmp x22, x21
	b.ne .LBB18_5
.LBB18_6:
	ands x8, x19, #0xc
	b.eq .LBB18_21
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
	ldp q18, q17, [x29, #-256]
	fmov v0.4s, #1.00000000
	ldp q20, q19, [x29, #-224]
	mov w8, #981467136
	dup v16.4s, w8
	sub x8, x29, #192
	sub x0, x29, #128
	dup v1.4s, v18.s[3]
	dup v3.4s, v17.s[3]
	stp q18, q17, [sp, #288]
	dup v5.4s, v20.s[3]
	dup v7.4s, v19.s[3]
	stp q20, q19, [sp, #320]
	fdiv v2.4s, v0.4s, v1.4s
	fcmgt v1.4s, v1.4s, v16.4s
	fdiv v4.4s, v0.4s, v3.4s
	fcmgt v3.4s, v3.4s, v16.4s
	and v1.16b, v1.16b, v2.16b
	fmul v1.4s, v18.4s, v1.4s
	fdiv v6.4s, v0.4s, v5.4s
	fcmgt v5.4s, v5.4s, v16.4s
	and v2.16b, v3.16b, v4.16b
	fmul v2.4s, v17.4s, v2.4s
	fdiv v0.4s, v0.4s, v7.4s
	fcmgt v7.4s, v7.4s, v16.4s
	stp q1, q2, [x29, #-128]
	and v3.16b, v5.16b, v6.16b
	fmul v3.4s, v20.4s, v3.4s
	and v0.16b, v7.16b, v0.16b
	fmul v0.4s, v19.4s, v0.4s
	stp q3, q0, [x29, #-96]
	fmov s0, s8
	bl linear_srgb::tf::math::gamma_outside_fit::<16>
	ldp q0, q1, [x29, #-192]
	sub x1, x29, #256
	ldp q2, q4, [sp, #288]
	mov x0, x20
	mov x2, x19
	mov v0.s[3], v2.s[3]
	ldp q2, q3, [x29, #-160]
	mov v1.s[3], v4.s[3]
	ldp q5, q4, [sp, #320]
	mov v2.s[3], v5.s[3]
	mov v3.s[3], v4.s[3]
	stp q0, q1, [x29, #-256]
	stp q2, q3, [x29, #-224]
	b .LBB18_20
.LBB18_8:
	lsl x8, x1, #2
	fcmp s8, #0.0
	and x9, x1, #0x1ffffffffffffff0
	add x19, x0, x9, lsl #2
	and x9, x8, #0x7fffffffffffffc0
	add x8, x0, x9
	b.ne .LBB18_14
	cbz x9, .LBB18_12
	fmov v0.4s, #1.00000000
	mov w9, #981467136
	dup v1.4s, w9
	adrp x9, .LCPI18_1
	ldr q2, [x9, :lo12:.LCPI18_1]
.LBB18_11:
	ldp q4, q3, [x0]
	ldp q17, q18, [x0, #32]
	dup v5.4s, v4.s[3]
	dup v7.4s, v3.s[3]
	dup v19.4s, v17.s[3]
	dup v21.4s, v18.s[3]
	fdiv v6.4s, v0.4s, v5.4s
	fcmgt v5.4s, v5.4s, v1.4s
	fdiv v16.4s, v0.4s, v7.4s
	fcmgt v7.4s, v7.4s, v1.4s
	and v5.16b, v5.16b, v6.16b
	fmul v5.4s, v4.4s, v5.4s
	fdiv v20.4s, v0.4s, v19.4s
	fcmgt v19.4s, v19.4s, v1.4s
	and v6.16b, v7.16b, v16.16b
	fmul v6.4s, v3.4s, v6.4s
	fdiv v22.4s, v0.4s, v21.4s
	fcmgt v21.4s, v21.4s, v1.4s
	and v7.16b, v19.16b, v20.16b
	fcmgt v19.4s, v5.4s, #0.0
	fcmeq v20.4s, v5.4s, v5.4s
	fmul v7.4s, v17.4s, v7.4s
	bic v5.16b, v5.16b, v20.16b
	and v19.16b, v19.16b, v2.16b
	fcmgt v23.4s, v7.4s, #0.0
	fcmeq v24.4s, v7.4s, v7.4s
	orr v5.16b, v19.16b, v5.16b
	and v16.16b, v21.16b, v22.16b
	fcmgt v21.4s, v6.4s, #0.0
	fcmeq v22.4s, v6.4s, v6.4s
	bic v7.16b, v7.16b, v24.16b
	mov v5.s[3], v4.s[3]
	fmul v16.4s, v18.4s, v16.4s
	bic v6.16b, v6.16b, v22.16b
	and v20.16b, v21.16b, v2.16b
	and v21.16b, v23.16b, v2.16b
	fcmgt v25.4s, v16.4s, #0.0
	fcmeq v26.4s, v16.4s, v16.4s
	orr v6.16b, v20.16b, v6.16b
	orr v7.16b, v21.16b, v7.16b
	mov v6.s[3], v3.s[3]
	bic v16.16b, v16.16b, v26.16b
	and v19.16b, v25.16b, v2.16b
	mov v7.s[3], v17.s[3]
	orr v16.16b, v19.16b, v16.16b
	stp q5, q6, [x0]
	mov v16.s[3], v18.s[3]
	stp q7, q16, [x0, #32]
	add x0, x0, #64
	cmp x0, x8
	b.ne .LBB18_11
.LBB18_12:
	ands x8, x1, #0xc
	b.eq .LBB18_21
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
	fmov v2.4s, #1.00000000
	ldp q7, q16, [x29, #-96]
	mov w8, #981467136
	dup v20.4s, w8
	adrp x8, .LCPI18_1
	dup v3.4s, v0.s[3]
	dup v5.4s, v1.s[3]
	ldr q21, [x8, :lo12:.LCPI18_1]
	dup v17.4s, v7.s[3]
	dup v19.4s, v16.s[3]
	fdiv v4.4s, v2.4s, v3.4s
	fcmgt v3.4s, v3.4s, v20.4s
	fdiv v6.4s, v2.4s, v5.4s
	fcmgt v5.4s, v5.4s, v20.4s
	and v3.16b, v3.16b, v4.16b
	fmul v3.4s, v0.4s, v3.4s
	fdiv v18.4s, v2.4s, v17.4s
	fcmgt v17.4s, v17.4s, v20.4s
	and v4.16b, v5.16b, v6.16b
	fcmgt v6.4s, v3.4s, #0.0
	fmul v4.4s, v1.4s, v4.4s
	and v6.16b, v6.16b, v21.16b
	fdiv v2.4s, v2.4s, v19.4s
	fcmgt v19.4s, v19.4s, v20.4s
	and v5.16b, v17.16b, v18.16b
	fcmeq v17.4s, v3.4s, v3.4s
	fcmgt v18.4s, v4.4s, #0.0
	fmul v5.4s, v7.4s, v5.4s
	bic v3.16b, v3.16b, v17.16b
	and v17.16b, v18.16b, v21.16b
	fcmgt v20.4s, v5.4s, #0.0
	fcmeq v22.4s, v5.4s, v5.4s
	orr v3.16b, v6.16b, v3.16b
	and v2.16b, v19.16b, v2.16b
	fcmeq v19.4s, v4.4s, v4.4s
	mov v3.s[3], v0.s[3]
	bic v5.16b, v5.16b, v22.16b
	and v18.16b, v20.16b, v21.16b
	fmul v2.4s, v16.4s, v2.4s
	bic v4.16b, v4.16b, v19.16b
	orr v5.16b, v18.16b, v5.16b
	fcmgt v23.4s, v2.4s, #0.0
	fcmeq v24.4s, v2.4s, v2.4s
	orr v4.16b, v17.16b, v4.16b
	mov v5.s[3], v7.s[3]
	mov v4.s[3], v1.s[3]
	bic v2.16b, v2.16b, v24.16b
	and v6.16b, v23.16b, v21.16b
	orr v2.16b, v6.16b, v2.16b
	stp q3, q4, [x29, #-128]
	mov v2.s[3], v16.s[3]
	stp q5, q2, [x29, #-96]
	b .LBB18_19
.LBB18_14:
	adrp x21, .LCPI18_0
	str s8, [x29, #28]
	cbz x9, .LBB18_17
	ldr s1, [x29, #28]
	mov w10, #64269
	mov w9, #981467136
	movk w10, #74, lsl #16
	dup v3.4s, w9
	mov w9, #64269
	fcmp s1, s0
	dup v2.4s, w10
	mov w10, #1267
	movk w9, #16330, lsl #16
	movk w10, #16181, lsl #16
	fmov v11.4s, #1.00000000
	stp q2, q3, [sp, #256]
	fcsel s0, s0, s1, gt
	dup v2.4s, w9
	dup v1.4s, w10
	mov w9, #27455
	mov w10, #8388607
	movk w9, #16094, lsl #16
	fmov s3, w10
	stp q1, q2, [sp, #224]
	dup v2.4s, w9
	mov w9, #46233
	movk w9, #16147, lsl #16
	dup v1.4s, w9
	fmov w9, s0
	sub w9, w9, #1
	stp q1, q2, [sp, #192]
	movi d1, #0000000000000000
	fmov s2, w9
	cmp w9, w10
	mov w9, #14483
	mov w10, #43579
	movk w9, #16246, lsl #16
	movk w10, #16440, lsl #16
	dup v4.4s, w9
	mov w9, #-1023672320
	cmhi v2.4s, v3.4s, v2.4s
	dup v3.4s, w10
	dup v21.4s, w9
	mov w9, #63802
	mov w10, #1123942400
	fcsel s0, s1, s0, lo
	movk w9, #14625, lsl #16
	stp q3, q4, [sp, #160]
	dup v4.4s, w10
	dup v3.4s, w9
	mov w10, #50297
	mov w9, #38156
	and v2.16b, v2.16b, v11.16b
	movk w10, #15022, lsl #16
	movk w9, #15389, lsl #16
	str q0, [sp, #48]
	stp q3, q4, [sp, #128]
	dup v4.4s, w10
	dup v3.4s, w9
	mov w10, #22716
	mov w9, #65005
	dup v2.4s, v2.s[0]
	movk w10, #15715, lsl #16
	movk w9, #15989, lsl #16
	stp q3, q4, [sp, #96]
	dup v3.4s, w10
	mov w10, #29208
	movk w10, #16177, lsl #16
	dup v1.4s, w9
	dup v0.4s, w10
	stp q2, q3, [sp, #64]
	stp q0, q1, [sp, #16]
	ldr q0, [x21, :lo12:.LCPI18_0]
	str q0, [sp]
.LBB18_16:
	ldp q4, q6, [x0]
	movi v5.4s, #128, lsl #16
	ldr q7, [sp, #272]
	ldp q12, q17, [x0, #32]
	ldp q23, q22, [sp, #240]
	movi v25.4s, #127, msl #16
	dup v0.4s, v4.s[3]
	dup v2.4s, v6.s[3]
	stp q6, q4, [sp, #320]
	ldr q24, [sp, #224]
	fmov v26.4s, #-1.00000000
	mvni v29.4s, #126
	ldr q30, [sp, #176]
	mvni v27.4s, #126
	mvni v28.4s, #126
	fdiv v1.4s, v11.4s, v0.4s
	fcmgt v0.4s, v0.4s, v7.4s
	mov v10.16b, v30.16b
	fdiv v3.4s, v11.4s, v2.4s
	and v0.16b, v0.16b, v1.16b
	dup v1.4s, v12.s[3]
	fmul v13.4s, v4.4s, v0.4s
	fcmgt v0.4s, v2.4s, v7.4s
	fdiv v2.4s, v11.4s, v1.4s
	fcmge v4.4s, v13.4s, v5.4s
	fcmgt v1.4s, v1.4s, v7.4s
	and v0.16b, v0.16b, v3.16b
	and v3.16b, v4.16b, v13.16b
	dup v4.4s, v17.s[3]
	fmul v14.4s, v6.4s, v0.4s
	fminnm v0.4s, v3.4s, v11.4s
	fcmge v6.4s, v14.4s, v5.4s
	fdiv v3.4s, v11.4s, v4.4s
	cmeq v16.4s, v0.4s, #0
	add v0.4s, v0.4s, v22.4s
	and v2.16b, v1.16b, v2.16b
	and v6.16b, v6.16b, v14.16b
	mov v1.16b, v16.16b
	stp q17, q16, [sp, #288]
	fmul v15.4s, v12.4s, v2.4s
	fcmgt v2.4s, v4.4s, v7.4s
	bsl v1.16b, v23.16b, v0.16b
	fminnm v0.4s, v6.4s, v11.4s
	fcmge v6.4s, v15.4s, v5.4s
	and v4.16b, v1.16b, v25.16b
	cmeq v7.4s, v0.4s, #0
	add v0.4s, v0.4s, v22.4s
	ssra v27.4s, v1.4s, #23
	mov v1.16b, v30.16b
	and v2.16b, v2.16b, v3.16b
	add v4.4s, v4.4s, v24.4s
	and v3.16b, v6.16b, v15.16b
	mov v6.16b, v7.16b
	scvtf v27.4s, v27.4s
	fmul v16.4s, v17.4s, v2.4s
	bsl v6.16b, v23.16b, v0.16b
	fadd v0.4s, v4.4s, v26.4s
	fadd v4.4s, v4.4s, v11.4s
	fminnm v3.4s, v3.4s, v11.4s
	fdiv v2.4s, v0.4s, v4.4s
	fcmge v0.4s, v16.4s, v5.4s
	and v18.16b, v6.16b, v25.16b
	cmeq v17.4s, v3.4s, #0
	add v3.4s, v3.4s, v22.4s
	ssra v28.4s, v6.4s, #23
	add v4.4s, v18.4s, v24.4s
	and v0.16b, v0.16b, v16.16b
	bit v3.16b, v23.16b, v17.16b
	scvtf v28.4s, v28.4s
	fadd v18.4s, v4.4s, v26.4s
	fadd v4.4s, v4.4s, v11.4s
	fminnm v19.4s, v0.4s, v11.4s
	ldp q8, q0, [sp, #192]
	and v20.16b, v3.16b, v25.16b
	ssra v29.4s, v3.4s, #23
	mov v3.16b, v30.16b
	fdiv v4.4s, v18.4s, v4.4s
	mov v5.16b, v8.16b
	mov v31.16b, v8.16b
	cmeq v9.4s, v19.4s, #0
	add v18.4s, v19.4s, v22.4s
	add v19.4s, v20.4s, v24.4s
	scvtf v6.4s, v29.4s
	bit v18.16b, v23.16b, v9.16b
	fadd v20.4s, v19.4s, v26.4s
	fadd v19.4s, v19.4s, v11.4s
	and v22.16b, v18.16b, v25.16b
	mov v25.16b, v8.16b
	fdiv v19.4s, v20.4s, v19.4s
	add v20.4s, v22.4s, v24.4s
	fmul v23.4s, v4.4s, v4.4s
	fadd v22.4s, v20.4s, v26.4s
	fadd v20.4s, v20.4s, v11.4s
	fmla v25.4s, v0.4s, v23.4s
	fdiv v20.4s, v22.4s, v20.4s
	fmul v22.4s, v2.4s, v2.4s
	fmla v1.4s, v23.4s, v25.4s
	mvni v25.4s, #126
	fmul v24.4s, v19.4s, v19.4s
	fmla v5.4s, v0.4s, v22.4s
	ssra v25.4s, v18.4s, #23
	fmla v31.4s, v0.4s, v24.4s
	fmla v10.4s, v22.4s, v5.4s
	fmla v3.4s, v24.4s, v31.4s
	fmul v26.4s, v20.4s, v20.4s
	fmla v8.4s, v0.4s, v26.4s
	ldp q5, q0, [sp, #144]
	mov v31.16b, v0.16b
	mov v18.16b, v0.16b
	fmla v30.4s, v26.4s, v8.4s
	fmla v31.4s, v22.4s, v10.4s
	mov v22.16b, v0.16b
	fmla v22.4s, v23.4s, v1.4s
	mov v1.16b, v0.16b
	fmla v18.4s, v26.4s, v30.4s
	fmla v27.4s, v2.4s, v31.4s
	ldp q10, q0, [sp, #32]
	ldr q31, [sp, #96]
	fmla v1.4s, v24.4s, v3.4s
	mov v29.16b, v31.16b
	fmla v28.4s, v4.4s, v22.4s
	scvtf v22.4s, v25.4s
	fmul v2.4s, v27.4s, v0.s[0]
	fmla v6.4s, v19.4s, v1.4s
	fmul v3.4s, v28.4s, v0.s[0]
	fmla v22.4s, v20.4s, v18.4s
	fmul v4.4s, v6.4s, v0.s[0]
	fmaxnm v6.4s, v2.4s, v21.4s
	fmaxnm v18.4s, v3.4s, v21.4s
	fmul v1.4s, v22.4s, v0.s[0]
	movi v0.4s, #67, lsl #24
	fmaxnm v19.4s, v4.4s, v21.4s
	fminnm v18.4s, v18.4s, v0.4s
	fmaxnm v20.4s, v1.4s, v21.4s
	fminnm v6.4s, v6.4s, v0.4s
	fminnm v19.4s, v19.4s, v0.4s
	frintn v23.4s, v18.4s
	fminnm v25.4s, v20.4s, v0.4s
	frintn v22.4s, v6.4s
	frintn v20.4s, v19.4s
	fminnm v8.4s, v23.4s, v5.4s
	frintn v26.4s, v25.4s
	fminnm v22.4s, v22.4s, v5.4s
	fminnm v20.4s, v20.4s, v5.4s
	fsub v23.4s, v18.4s, v8.4s
	fminnm v18.4s, v26.4s, v5.4s
	fsub v24.4s, v6.4s, v22.4s
	ldp q30, q5, [sp, #112]
	fcvtns v22.4s, v22.4s
	fsub v19.4s, v19.4s, v20.4s
	fcvtns v20.4s, v20.4s
	mov v26.16b, v30.16b
	fsub v6.4s, v25.4s, v18.4s
	mov v25.16b, v31.16b
	mov v27.16b, v30.16b
	mov v28.16b, v30.16b
	fcvtns v18.4s, v18.4s
	shl v22.4s, v22.4s, #23
	fmla v26.4s, v5.4s, v24.4s
	shl v20.4s, v20.4s, #23
	fmla v27.4s, v5.4s, v23.4s
	fmla v28.4s, v5.4s, v19.4s
	shl v18.4s, v18.4s, #23
	add v22.4s, v22.4s, v11.4s
	add v20.4s, v20.4s, v11.4s
	fmla v25.4s, v24.4s, v26.4s
	mov v26.16b, v30.16b
	fmla v29.4s, v23.4s, v27.4s
	mov v27.16b, v31.16b
	add v18.4s, v18.4s, v11.4s
	fmla v26.4s, v5.4s, v6.4s
	ldr q5, [sp, #80]
	fmla v27.4s, v19.4s, v28.4s
	mov v30.16b, v5.16b
	mov v28.16b, v5.16b
	fmla v30.4s, v24.4s, v25.4s
	mov v25.16b, v31.16b
	fmla v28.4s, v23.4s, v29.4s
	mov v29.16b, v5.16b
	fmov v31.4s, #1.00000000
	fmla v25.4s, v6.4s, v26.4s
	mov v26.16b, v10.16b
	fmla v29.4s, v19.4s, v27.4s
	mov v27.16b, v10.16b
	fmla v26.4s, v24.4s, v30.4s
	mov v30.16b, v5.16b
	ldr q5, [sp, #16]
	fmla v27.4s, v23.4s, v28.4s
	mov v28.16b, v10.16b
	fmla v30.4s, v6.4s, v25.4s
	mov v25.16b, v5.16b
	fmla v28.4s, v19.4s, v29.4s
	mov v29.16b, v5.16b
	fmla v25.4s, v24.4s, v26.4s
	mov v26.16b, v10.16b
	fmla v29.4s, v23.4s, v27.4s
	fmov v27.4s, #1.00000000
	fmla v26.4s, v6.4s, v30.4s
	mov v30.16b, v5.16b
	fmla v31.4s, v24.4s, v25.4s
	mov v24.16b, v5.16b
	fcvtns v25.4s, v8.4s
	fmla v27.4s, v23.4s, v29.4s
	fmov v23.4s, #1.00000000
	fcmge v29.4s, v2.4s, v0.4s
	fmla v30.4s, v19.4s, v28.4s
	fmov v28.4s, #1.00000000
	fcmgt v2.4s, v21.4s, v2.4s
	fmla v24.4s, v6.4s, v26.4s
	fcmge v26.4s, v3.4s, v0.4s
	fcmgt v3.4s, v21.4s, v3.4s
	shl v25.4s, v25.4s, #23
	fmul v22.4s, v31.4s, v22.4s
	ldr q5, [sp, #304]
	fmla v28.4s, v19.4s, v30.4s
	fcmge v19.4s, v4.4s, v0.4s
	fcmge v30.4s, v1.4s, v0.4s
	fmla v23.4s, v6.4s, v24.4s
	mvn v6.16b, v29.16b
	add v24.4s, v25.4s, v11.4s
	fcmgt v4.4s, v21.4s, v4.4s
	fcmgt v1.4s, v21.4s, v1.4s
	mvn v25.16b, v26.16b
	ldr q0, [sp]
	mvn v31.16b, v19.16b
	bic v2.16b, v6.16b, v2.16b
	fmul v6.4s, v27.4s, v24.4s
	mvn v24.16b, v30.16b
	bic v3.16b, v25.16b, v3.16b
	fmul v20.4s, v28.4s, v20.4s
	fmul v18.4s, v23.4s, v18.4s
	and v19.16b, v19.16b, v0.16b
	bic v4.16b, v31.16b, v4.16b
	and v2.16b, v2.16b, v22.16b
	and v22.16b, v29.16b, v0.16b
	bic v1.16b, v24.16b, v1.16b
	and v3.16b, v3.16b, v6.16b
	and v6.16b, v26.16b, v0.16b
	and v4.16b, v4.16b, v20.16b
	orr v2.16b, v2.16b, v22.16b
	and v1.16b, v1.16b, v18.16b
	and v18.16b, v30.16b, v0.16b
	orr v3.16b, v3.16b, v6.16b
	ldr q0, [sp, #64]
	fcmeq v6.4s, v13.4s, v13.4s
	orr v4.16b, v4.16b, v19.16b
	orr v1.16b, v1.16b, v18.16b
	bit v3.16b, v0.16b, v7.16b
	fcmeq v7.4s, v14.4s, v14.4s
	bit v2.16b, v0.16b, v5.16b
	bit v4.16b, v0.16b, v17.16b
	fcmeq v17.4s, v15.4s, v15.4s
	bif v0.16b, v1.16b, v9.16b
	fcmeq v1.4s, v16.4s, v16.4s
	bif v3.16b, v14.16b, v7.16b
	bif v2.16b, v13.16b, v6.16b
	bif v4.16b, v15.16b, v17.16b
	bif v0.16b, v16.16b, v1.16b
	ldp q1, q5, [sp, #320]
	mov v3.s[3], v1.s[3]
	ldr q1, [sp, #288]
	mov v2.s[3], v5.s[3]
	mov v4.s[3], v12.s[3]
	mov v0.s[3], v1.s[3]
	stp q2, q3, [x0]
	stp q4, q0, [x0, #32]
	add x0, x0, #64
	cmp x0, x8
	b.ne .LBB18_16
.LBB18_17:
	ldr s10, [x29, #28]
	ands x8, x1, #0xc
	b.eq .LBB18_21
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
	fmov v4.4s, #1.00000000
	mov w8, #981467136
	movi v19.4s, #128, lsl #16
	movi v29.4s, #127, msl #16
	dup v17.4s, w8
	mov w8, #64269
	fmov v31.4s, #-1.00000000
	dup v2.4s, v0.s[3]
	dup v6.4s, v1.s[3]
	movk w8, #74, lsl #16
	dup v25.4s, w8
	mov w8, #64269
	mov w9, #27455
	movk w8, #16330, lsl #16
	movk w9, #16094, lsl #16
	stp q1, q0, [sp, #320]
	fdiv v3.4s, v4.4s, v2.4s
	fcmgt v2.4s, v2.4s, v17.4s
	dup v26.4s, w8
	mov w8, #1267
	fmov s11, #4.00000000
	mvni v15.4s, #126
	movk w8, #16181, lsl #16
	mov w10, #-1023672320
	dup v30.4s, w8
	mov w8, #46233
	movk w8, #16147, lsl #16
	fcmp s10, s11
	fdiv v7.4s, v4.4s, v6.4s
	fcmgt v6.4s, v6.4s, v17.4s
	and v5.16b, v2.16b, v3.16b
	ldp q3, q2, [x29, #-96]
	dup v16.4s, v3.s[3]
	fmul v5.4s, v0.4s, v5.4s
	stp q2, q3, [sp, #288]
	fdiv v18.4s, v4.4s, v16.4s
	fcmge v20.4s, v5.4s, v19.4s
	fcmgt v16.4s, v16.4s, v17.4s
	and v6.16b, v6.16b, v7.16b
	and v7.16b, v20.16b, v5.16b
	dup v20.4s, v2.s[3]
	fmul v6.4s, v1.4s, v6.4s
	fminnm v21.4s, v7.4s, v4.4s
	fcmgt v17.4s, v20.4s, v17.4s
	fcmge v23.4s, v6.4s, v19.4s
	fdiv v22.4s, v4.4s, v20.4s
	cmeq v7.4s, v21.4s, #0
	add v21.4s, v21.4s, v25.4s
	and v16.16b, v16.16b, v18.16b
	and v18.16b, v23.16b, v6.16b
	bit v21.16b, v26.16b, v7.16b
	fmul v16.4s, v3.4s, v16.4s
	fmov s3, s10
	fminnm v23.4s, v18.4s, v4.4s
	and v20.16b, v21.16b, v29.16b
	ssra v15.4s, v21.4s, #23
	mvni v21.4s, #126
	fcmge v24.4s, v16.4s, v19.4s
	cmeq v18.4s, v23.4s, #0
	add v23.4s, v23.4s, v25.4s
	add v20.4s, v20.4s, v30.4s
	and v17.16b, v17.16b, v22.16b
	and v22.16b, v24.16b, v16.16b
	mov v24.16b, v18.16b
	fmul v17.4s, v2.4s, v17.4s
	mvni v2.4s, #126
	bsl v24.16b, v26.16b, v23.16b
	fadd v23.4s, v20.4s, v31.4s
	fadd v20.4s, v20.4s, v4.4s
	fminnm v27.4s, v22.4s, v4.4s
	fdiv v22.4s, v23.4s, v20.4s
	fcmge v20.4s, v17.4s, v19.4s
	and v28.16b, v24.16b, v29.16b
	cmeq v19.4s, v27.4s, #0
	add v23.4s, v27.4s, v25.4s
	ssra v2.4s, v24.4s, #23
	add v28.4s, v28.4s, v30.4s
	and v20.16b, v20.16b, v17.16b
	mov v27.16b, v19.16b
	scvtf v2.4s, v2.4s
	fminnm v8.4s, v20.4s, v4.4s
	bsl v27.16b, v26.16b, v23.16b
	fadd v23.4s, v28.4s, v31.4s
	fadd v28.4s, v28.4s, v4.4s
	cmeq v20.4s, v8.4s, #0
	and v9.16b, v27.16b, v29.16b
	add v25.4s, v8.4s, v25.4s
	fdiv v23.4s, v23.4s, v28.4s
	ssra v21.4s, v27.4s, #23
	scvtf v27.4s, v15.4s
	mov v28.16b, v20.16b
	add v8.4s, v9.4s, v30.4s
	dup v9.4s, w8
	mov w8, #14483
	movk w8, #16246, lsl #16
	bsl v28.16b, v26.16b, v25.16b
	fadd v25.4s, v8.4s, v31.4s
	fadd v26.4s, v8.4s, v4.4s
	mov v0.16b, v9.16b
	dup v1.4s, w8
	mov v13.16b, v9.16b
	mov w8, #43579
	movk w8, #16440, lsl #16
	and v29.16b, v28.16b, v29.16b
	mov v14.16b, v1.16b
	mov v10.16b, v1.16b
	fdiv v26.4s, v25.4s, v26.4s
	mov v24.16b, v1.16b
	add v25.4s, v29.4s, v30.4s
	mov v30.16b, v9.16b
	fmul v8.4s, v23.4s, v23.4s
	fadd v29.4s, v25.4s, v31.4s
	fadd v25.4s, v25.4s, v4.4s
	dup v31.4s, w9
	fmla v13.4s, v31.4s, v8.4s
	fdiv v25.4s, v29.4s, v25.4s
	fmul v29.4s, v22.4s, v22.4s
	fmla v10.4s, v8.4s, v13.4s
	fmul v12.4s, v26.4s, v26.4s
	fmla v0.4s, v31.4s, v29.4s
	fmla v30.4s, v31.4s, v12.4s
	fmla v14.4s, v29.4s, v0.4s
	dup v0.4s, w8
	mov w8, #8388607
	mov v13.16b, v0.16b
	fmla v24.4s, v12.4s, v30.4s
	mvni v30.4s, #126
	fmul v4.4s, v25.4s, v25.4s
	fmla v13.4s, v29.4s, v14.4s
	fcsel s29, s11, s3, gt
	ssra v30.4s, v28.4s, #23
	scvtf v28.4s, v21.4s
	dup v21.4s, w10
	mov w10, #1123942400
	fmla v9.4s, v31.4s, v4.4s
	mov v31.16b, v0.16b
	fmov w9, s29
	fmla v27.4s, v22.4s, v13.4s
	scvtf v30.4s, v30.4s
	fmla v31.4s, v8.4s, v10.4s
	movi d8, #0000000000000000
	sub w9, w9, #1
	fmla v1.4s, v4.4s, v9.4s
	mov v9.16b, v0.16b
	cmp w9, w8
	fmla v9.4s, v12.4s, v24.4s
	fcsel s29, s8, s29, lo
	fmla v2.4s, v23.4s, v31.4s
	fmla v0.4s, v4.4s, v1.4s
	fmov v4.4s, #1.00000000
	fmul v23.4s, v27.4s, v29.s[0]
	fmla v28.4s, v26.4s, v9.4s
	fmul v24.4s, v2.4s, v29.s[0]
	movi v26.4s, #67, lsl #24
	fmla v30.4s, v25.4s, v0.4s
	fmaxnm v0.4s, v23.4s, v21.4s
	fmul v22.4s, v28.4s, v29.s[0]
	fmaxnm v1.4s, v24.4s, v21.4s
	fmul v25.4s, v30.4s, v29.s[0]
	dup v30.4s, w10
	mov w10, #50297
	movk w10, #15022, lsl #16
	fminnm v0.4s, v0.4s, v26.4s
	dup v11.4s, w10
	mov w10, #63802
	fmaxnm v2.4s, v22.4s, v21.4s
	fminnm v1.4s, v1.4s, v26.4s
	movk w10, #14625, lsl #16
	fmaxnm v3.4s, v25.4s, v21.4s
	mov v12.16b, v11.16b
	frintn v27.4s, v0.4s
	fminnm v2.4s, v2.4s, v26.4s
	frintn v29.4s, v1.4s
	fminnm v3.4s, v3.4s, v26.4s
	fminnm v28.4s, v27.4s, v30.4s
	frintn v31.4s, v2.4s
	fminnm v27.4s, v29.4s, v30.4s
	frintn v9.4s, v3.4s
	fsub v8.4s, v0.4s, v28.4s
	dup v0.4s, w10
	mov w10, #38156
	fminnm v29.4s, v31.4s, v30.4s
	fsub v10.4s, v1.4s, v27.4s
	movk w10, #15389, lsl #16
	fminnm v31.4s, v9.4s, v30.4s
	mov v1.16b, v11.16b
	fcvtns v27.4s, v27.4s
	fsub v9.4s, v2.4s, v29.4s
	dup v2.4s, w10
	fmla v1.4s, v0.4s, v8.4s
	fsub v30.4s, v3.4s, v31.4s
	mov v3.16b, v11.16b
	mov w10, #22716
	movk w10, #15715, lsl #16
	fmla v12.4s, v0.4s, v10.4s
	fcvtns v29.4s, v29.4s
	mov v13.16b, v2.16b
	mov v14.16b, v2.16b
	fcvtns v31.4s, v31.4s
	fmla v3.4s, v0.4s, v9.4s
	shl v27.4s, v27.4s, #23
	fmla v11.4s, v0.4s, v30.4s
	mov v0.16b, v2.16b
	fmla v13.4s, v8.4s, v1.4s
	dup v1.4s, w10
	mov w10, #65005
	movk w10, #15989, lsl #16
	fmla v14.4s, v10.4s, v12.4s
	shl v29.4s, v29.4s, #23
	fmla v0.4s, v9.4s, v3.4s
	dup v12.4s, w10
	mov w10, #29208
	mov v3.16b, v1.16b
	fmla v2.4s, v30.4s, v11.4s
	mov v11.16b, v1.16b
	movk w10, #16177, lsl #16
	add v27.4s, v27.4s, v4.4s
	add v29.4s, v29.4s, v4.4s
	fmla v3.4s, v8.4s, v13.4s
	mov v13.16b, v1.16b
	fmla v11.4s, v9.4s, v0.4s
	mov v0.16b, v12.16b
	fmla v1.4s, v30.4s, v2.4s
	mov v2.16b, v12.16b
	fmla v13.4s, v10.4s, v14.4s
	mov v14.16b, v12.16b
	fmla v0.4s, v8.4s, v3.4s
	dup v3.4s, w10
	fmla v2.4s, v9.4s, v11.4s
	fmla v12.4s, v30.4s, v1.4s
	fmov v11.4s, #1.00000000
	fmla v14.4s, v10.4s, v13.4s
	mov v13.16b, v3.16b
	mov v1.16b, v3.16b
	fmla v13.4s, v8.4s, v0.4s
	mov v0.16b, v3.16b
	fmla v1.4s, v9.4s, v2.4s
	fmov v2.4s, #1.00000000
	fmla v3.4s, v30.4s, v12.4s
	fmla v0.4s, v10.4s, v14.4s
	fmov v14.4s, #1.00000000
	fmla v11.4s, v8.4s, v13.4s
	fmov s8, w9
	fmla v2.4s, v9.4s, v1.4s
	fcmge v1.4s, v23.4s, v26.4s
	fcmge v9.4s, v24.4s, v26.4s
	fcmgt v23.4s, v21.4s, v23.4s
	fcmgt v24.4s, v21.4s, v24.4s
	fmla v14.4s, v10.4s, v0.4s
	fcvtns v0.4s, v28.4s
	fmov v28.4s, #1.00000000
	fmov s10, w8
	fmul v2.4s, v2.4s, v29.4s
	shl v0.4s, v0.4s, #23
	fmla v28.4s, v30.4s, v3.4s
	fcmge v3.4s, v22.4s, v26.4s
	fcmge v26.4s, v25.4s, v26.4s
	shl v30.4s, v31.4s, #23
	mvn v31.16b, v1.16b
	fcmgt v22.4s, v21.4s, v22.4s
	fcmgt v21.4s, v21.4s, v25.4s
	mvn v25.16b, v9.16b
	add v0.4s, v0.4s, v4.4s
	cmhi v8.4s, v10.4s, v8.4s
	fmul v27.4s, v14.4s, v27.4s
	add v30.4s, v30.4s, v4.4s
	mvn v10.16b, v3.16b
	bic v23.16b, v31.16b, v23.16b
	mvn v31.16b, v26.16b
	bic v24.16b, v25.16b, v24.16b
	ldr q25, [x21, :lo12:.LCPI18_0]
	fmul v0.4s, v11.4s, v0.4s
	and v4.16b, v8.16b, v4.16b
	fmul v28.4s, v28.4s, v30.4s
	bic v22.16b, v10.16b, v22.16b
	and v1.16b, v1.16b, v25.16b
	bic v21.16b, v31.16b, v21.16b
	and v3.16b, v3.16b, v25.16b
	and v0.16b, v23.16b, v0.16b
	and v23.16b, v24.16b, v27.16b
	and v24.16b, v9.16b, v25.16b
	and v25.16b, v26.16b, v25.16b
	and v2.16b, v22.16b, v2.16b
	and v21.16b, v21.16b, v28.16b
	orr v0.16b, v0.16b, v1.16b
	dup v1.4s, v4.s[0]
	orr v4.16b, v23.16b, v24.16b
	orr v2.16b, v2.16b, v3.16b
	orr v3.16b, v21.16b, v25.16b
	bit v0.16b, v1.16b, v7.16b
	fcmeq v7.4s, v5.4s, v5.4s
	bit v4.16b, v1.16b, v18.16b
	fcmeq v18.4s, v6.4s, v6.4s
	bit v2.16b, v1.16b, v19.16b
	bif v1.16b, v3.16b, v20.16b
	fcmeq v3.4s, v17.4s, v17.4s
	fcmeq v19.4s, v16.4s, v16.4s
	bif v0.16b, v5.16b, v7.16b
	bif v4.16b, v6.16b, v18.16b
	bif v1.16b, v17.16b, v3.16b
	ldp q3, q5, [sp, #320]
	bif v2.16b, v16.16b, v19.16b
	mov v0.s[3], v5.s[3]
	mov v4.s[3], v3.s[3]
	ldp q3, q5, [sp, #288]
	mov v2.s[3], v5.s[3]
	mov v1.s[3], v3.s[3]
	stp q0, q4, [x29, #-128]
	stp q2, q1, [x29, #-96]
.LBB18_19:
	sub x1, x29, #128
	mov x0, x19
	mov x2, x20
.LBB18_20:
	bl memcpy
.LBB18_21:
	add sp, sp, #544
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
.LBB18_22:
	.cfi_restore_state
	mov x20, x1
	mov x19, x0
	fmov s8, s0
	bl archmage::tokens::generated::arm::neon_detect
	fmov s0, s8
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB18_2
.LBB18_23:
	add sp, sp, #544
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
	b linear_srgb::simd::unpremultiply_linear_to_gamma_rgba_slice_tier_scalar
