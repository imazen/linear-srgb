.section .text.stub_gamma_to_linear_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_slice
	.p2align	2
.type	stub_gamma_to_linear_slice,@function
stub_gamma_to_linear_slice:
	.cfi_startproc
	sub sp, sp, #464
	.cfi_def_cfa_offset 464
	stp d15, d14, [sp, #336]
	stp d13, d12, [sp, #352]
	stp d11, d10, [sp, #368]
	stp d9, d8, [sp, #384]
	stp x29, x30, [sp, #400]
	stp x28, x23, [sp, #416]
	stp x22, x21, [sp, #432]
	stp x20, x19, [sp, #448]
	add x29, sp, #400
	.cfi_def_cfa w29, 64
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
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
	str q0, [sp, #176]
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB6_15
	cmp w8, #2
	b.ne .LBB6_14
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
	sub x8, x29, #128
	add x0, x8, x20
	bl memset
	sub x0, x29, #128
	mov x1, x19
	mov x2, x20
	bl memcpy
	ldp q0, q1, [x29, #-128]
	mov x0, x23
	mov w9, #1
	stp q0, q1, [x29, #-192]
	ldp q0, q2, [x29, #-96]
	stp q0, q2, [x29, #-160]
	b .LBB6_5
.LBB6_4:
	mov w9, wzr
.LBB6_5:
	mov w8, #64269
	mov w10, #1267
	fmov v1.4s, #1.00000000
	movk w8, #74, lsl #16
	movk w10, #16181, lsl #16
	cmp x21, #0
	dup v2.4s, w8
	mov w8, #27455
	dup v0.4s, w10
	movk w8, #16094, lsl #16
	mov w10, #14483
	stur w9, [x29, #-196]
	dup v4.4s, w8
	mov w8, #46233
	movk w10, #16246, lsl #16
	movk w8, #16147, lsl #16
	stp q0, q2, [sp, #144]
	mvni v2.4s, #127, msl #16
	dup v3.4s, w8
	mvni v0.4s, #63, msl #16
	mov w8, #43579
	movk w8, #16440, lsl #16
	fneg v19.4s, v2.4s
	stp q3, q4, [sp, #112]
	dup v3.4s, w8
	mov w8, #-1023672320
	fneg v2.4s, v0.4s
	dup v21.4s, w8
	mov w8, #1123942400
	dup v0.4s, w8
	mov w8, #63802
	dup v4.4s, w10
	movk w8, #14625, lsl #16
	mov w10, #38156
	movk w10, #15389, lsl #16
	stp q0, q2, [sp, #48]
	dup v2.4s, w8
	mov w8, #50297
	movk w8, #15022, lsl #16
	stp q3, q4, [sp, #80]
	dup v0.4s, w8
	mov w8, #22716
	movk w8, #15715, lsl #16
	dup v27.4s, w8
	mov w8, #65005
	stp q0, q2, [sp, #16]
	dup v0.4s, w10
	mov w10, #29208
	movk w8, #15989, lsl #16
	movk w10, #16177, lsl #16
	dup v28.4s, w8
	dup v29.4s, w10
	sub x10, x29, #196
	add x8, x0, x22, lsl #6
	add x1, x10, #4
	str q0, [sp]
	csel x9, xzr, x1, eq
	b .LBB6_8
.LBB6_6:
	add x10, x0, #64
.LBB6_7:
	movi v3.2d, #0000000000000000
	ldp q0, q2, [x0]
	ldp q16, q6, [sp, #144]
	movi v4.4s, #127, msl #16
	ldp q8, q9, [x0, #32]
	fmov v17.4s, #-1.00000000
	ldp q26, q23, [sp, #112]
	mvni v20.4s, #126
	fmax v0.4s, v0.4s, v3.4s
	fmax v2.4s, v2.4s, v3.4s
	fmax v8.4s, v8.4s, v3.4s
	fmax v9.4s, v9.4s, v3.4s
	mov v22.16b, v26.16b
	mov v3.16b, v26.16b
	mov v24.16b, v26.16b
	fmin v30.4s, v0.4s, v1.4s
	fmin v31.4s, v2.4s, v1.4s
	fmin v8.4s, v8.4s, v1.4s
	fmin v9.4s, v9.4s, v1.4s
	add v0.4s, v30.4s, v6.4s
	add v13.4s, v31.4s, v6.4s
	add v14.4s, v8.4s, v6.4s
	add v15.4s, v9.4s, v6.4s
	mvni v6.4s, #126
	and v5.16b, v0.16b, v4.16b
	and v10.16b, v13.16b, v4.16b
	ssra v20.4s, v13.4s, #23
	and v11.16b, v14.16b, v4.16b
	and v7.16b, v15.16b, v4.16b
	ssra v6.4s, v0.4s, #23
	mvni v0.4s, #126
	add v2.4s, v5.4s, v16.4s
	scvtf v20.4s, v20.4s
	scvtf v6.4s, v6.4s
	ssra v0.4s, v14.4s, #23
	fadd v5.4s, v2.4s, v17.4s
	fadd v2.4s, v2.4s, v1.4s
	scvtf v0.4s, v0.4s
	fdiv v2.4s, v5.4s, v2.4s
	add v5.4s, v10.4s, v16.4s
	fadd v10.4s, v5.4s, v17.4s
	fadd v5.4s, v5.4s, v1.4s
	fdiv v12.4s, v10.4s, v5.4s
	add v5.4s, v11.4s, v16.4s
	fadd v10.4s, v5.4s, v17.4s
	fadd v5.4s, v5.4s, v1.4s
	fdiv v11.4s, v10.4s, v5.4s
	add v5.4s, v7.4s, v16.4s
	fadd v7.4s, v5.4s, v17.4s
	fadd v5.4s, v5.4s, v1.4s
	ldr q17, [sp, #96]
	mov v25.16b, v17.16b
	mov v16.16b, v17.16b
	fdiv v10.4s, v7.4s, v5.4s
	fmul v5.4s, v2.4s, v2.4s
	fmul v7.4s, v12.4s, v12.4s
	fmul v4.4s, v11.4s, v11.4s
	fmla v22.4s, v23.4s, v5.4s
	fmla v3.4s, v23.4s, v7.4s
	fmla v24.4s, v23.4s, v4.4s
	fmla v25.4s, v5.4s, v22.4s
	mov v22.16b, v17.16b
	fmla v22.4s, v7.4s, v3.4s
	ldr q3, [sp, #80]
	fmul v18.4s, v10.4s, v10.4s
	mov v13.16b, v3.16b
	fmla v13.4s, v5.4s, v25.4s
	mov v5.16b, v17.16b
	mov v17.16b, v3.16b
	fmla v26.4s, v23.4s, v18.4s
	mov v23.16b, v3.16b
	fmla v5.4s, v4.4s, v24.4s
	fmla v23.4s, v7.4s, v22.4s
	mvni v7.4s, #126
	mov v22.16b, v3.16b
	fmla v16.4s, v18.4s, v26.4s
	fmla v6.4s, v2.4s, v13.4s
	fcmeq v2.4s, v30.4s, #0.0
	fcmeq v3.4s, v9.4s, #0.0
	ssra v7.4s, v15.4s, #23
	fmla v17.4s, v4.4s, v5.4s
	fcmeq v4.4s, v31.4s, #0.0
	fmla v20.4s, v12.4s, v23.4s
	fcmlt v5.4s, v30.4s, #0.0
	movi v23.4s, #67, lsl #24
	fmla v22.4s, v18.4s, v16.4s
	mvni v18.4s, #127, msl #16
	fcmlt v16.4s, v31.4s, #0.0
	scvtf v7.4s, v7.4s
	fmla v0.4s, v11.4s, v17.4s
	fcmeq v17.4s, v30.4s, v19.4s
	bsl v2.16b, v18.16b, v6.16b
	fcmeq v6.4s, v8.4s, #0.0
	bsl v4.16b, v18.16b, v20.16b
	ldr q20, [sp, #64]
	fmla v7.4s, v10.4s, v22.4s
	bit v2.16b, v20.16b, v5.16b
	fcmlt v5.4s, v8.4s, #0.0
	bit v0.16b, v18.16b, v6.16b
	fcmeq v6.4s, v31.4s, v19.4s
	bit v4.16b, v20.16b, v16.16b
	fcmlt v16.4s, v9.4s, #0.0
	bsl v3.16b, v18.16b, v7.16b
	fcmeq v7.4s, v8.4s, v19.4s
	bit v2.16b, v19.16b, v17.16b
	bit v0.16b, v20.16b, v5.16b
	fcmeq v5.4s, v9.4s, v19.4s
	bit v4.16b, v19.16b, v6.16b
	ldr q6, [sp, #176]
	bit v3.16b, v20.16b, v16.16b
	ldp q18, q16, [sp]
	fmul v9.4s, v2.4s, v6.s[0]
	bit v0.16b, v19.16b, v7.16b
	mov v2.16b, v5.16b
	fmul v8.4s, v4.4s, v6.s[0]
	ldp q17, q7, [sp, #32]
	bsl v2.16b, v19.16b, v3.16b
	fmax v3.4s, v9.4s, v21.4s
	fmul v31.4s, v0.4s, v6.s[0]
	fcmgt v20.4s, v21.4s, v9.4s
	fmax v0.4s, v8.4s, v21.4s
	fcmge v9.4s, v9.4s, v23.4s
	fmul v30.4s, v2.4s, v6.s[0]
	fmin v2.4s, v3.4s, v23.4s
	fmax v3.4s, v31.4s, v21.4s
	fmin v0.4s, v0.4s, v23.4s
	fmax v4.4s, v30.4s, v21.4s
	frintn v5.4s, v2.4s
	fmin v3.4s, v3.4s, v23.4s
	frintn v6.4s, v0.4s
	fmin v4.4s, v4.4s, v23.4s
	fmin v14.4s, v5.4s, v7.4s
	frintn v5.4s, v3.4s
	fmin v13.4s, v6.4s, v7.4s
	frintn v6.4s, v4.4s
	fsub v2.4s, v2.4s, v14.4s
	fmin v15.4s, v5.4s, v7.4s
	mov v5.16b, v16.16b
	fsub v0.4s, v0.4s, v13.4s
	fmin v12.4s, v6.4s, v7.4s
	mov v6.16b, v16.16b
	mov v7.16b, v16.16b
	fmla v5.4s, v17.4s, v2.4s
	fsub v11.4s, v3.4s, v15.4s
	mov v3.16b, v18.16b
	fmla v6.4s, v17.4s, v0.4s
	fsub v10.4s, v4.4s, v12.4s
	mov v4.16b, v16.16b
	mov v16.16b, v27.16b
	fmla v3.4s, v2.4s, v5.4s
	mov v5.16b, v18.16b
	fcvtns v22.4s, v12.4s
	fmla v4.4s, v17.4s, v11.4s
	fmla v7.4s, v17.4s, v10.4s
	fmla v5.4s, v0.4s, v6.4s
	mov v6.16b, v27.16b
	fmla v16.4s, v2.4s, v3.4s
	mov v3.16b, v18.16b
	mov v17.16b, v28.16b
	shl v22.4s, v22.4s, #23
	fmla v3.4s, v11.4s, v4.4s
	mov v4.16b, v18.16b
	fmla v6.4s, v0.4s, v5.4s
	fmla v17.4s, v2.4s, v16.4s
	mov v16.16b, v28.16b
	mov v18.16b, v29.16b
	fmov v5.4s, #1.00000000
	add v22.4s, v22.4s, v1.4s
	fmla v4.4s, v10.4s, v7.4s
	mov v7.16b, v27.16b
	fmla v16.4s, v0.4s, v6.4s
	mov v6.16b, v29.16b
	fmla v18.4s, v2.4s, v17.4s
	fcvtns v17.4s, v14.4s
	fmla v7.4s, v11.4s, v3.4s
	mov v3.16b, v27.16b
	fmla v6.4s, v0.4s, v16.4s
	fmov v16.4s, #1.00000000
	fmla v3.4s, v10.4s, v4.4s
	mov v4.16b, v28.16b
	fmla v5.4s, v2.4s, v18.4s
	mov v2.16b, v29.16b
	fcvtns v18.4s, v13.4s
	shl v17.4s, v17.4s, #23
	fmla v4.4s, v11.4s, v7.4s
	mov v7.16b, v28.16b
	fmla v16.4s, v0.4s, v6.4s
	fcvtns v6.4s, v15.4s
	fmov v0.4s, #1.00000000
	add v17.4s, v17.4s, v1.4s
	shl v18.4s, v18.4s, #23
	fmla v7.4s, v10.4s, v3.4s
	mov v3.16b, v29.16b
	fmla v2.4s, v11.4s, v4.4s
	fmov v4.4s, #1.00000000
	fmul v5.4s, v5.4s, v17.4s
	shl v6.4s, v6.4s, #23
	add v18.4s, v18.4s, v1.4s
	fmla v3.4s, v10.4s, v7.4s
	fcmgt v7.4s, v21.4s, v8.4s
	fcmge v8.4s, v8.4s, v23.4s
	fmla v4.4s, v11.4s, v2.4s
	fcmgt v2.4s, v21.4s, v31.4s
	fcmge v31.4s, v31.4s, v23.4s
	fcmge v11.4s, v30.4s, v23.4s
	fcmgt v30.4s, v21.4s, v30.4s
	add v6.4s, v6.4s, v1.4s
	fmul v16.4s, v16.4s, v18.4s
	fmla v0.4s, v10.4s, v3.4s
	mvn v3.16b, v9.16b
	mvn v10.16b, v8.16b
	mvn v17.16b, v31.16b
	fmul v4.4s, v4.4s, v6.4s
	mvn v18.16b, v11.16b
	bic v3.16b, v3.16b, v20.16b
	bic v7.16b, v10.16b, v7.16b
	fmul v0.4s, v0.4s, v22.4s
	bic v2.16b, v17.16b, v2.16b
	bic v6.16b, v18.16b, v30.16b
	and v3.16b, v3.16b, v5.16b
	and v5.16b, v9.16b, v19.16b
	and v7.16b, v7.16b, v16.16b
	and v16.16b, v8.16b, v19.16b
	and v2.16b, v2.16b, v4.16b
	and v4.16b, v31.16b, v19.16b
	and v0.16b, v6.16b, v0.16b
	orr v3.16b, v3.16b, v5.16b
	and v5.16b, v11.16b, v19.16b
	orr v6.16b, v7.16b, v16.16b
	orr v2.16b, v2.16b, v4.16b
	orr v0.16b, v0.16b, v5.16b
	stp q3, q6, [x0]
	stp q2, q0, [x0, #32]
	mov x0, x10
.LBB6_8:
	cmp x0, x8
	ccmp x0, #0, #4, ne
	b.ne .LBB6_6
	cbz x9, .LBB6_11
	mov x10, xzr
	mov x0, x9
	mov x9, xzr
	b .LBB6_7
.LBB6_11:
	ldur w8, [x29, #-196]
	cbz w8, .LBB6_13
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB6_13:
	.cfi_def_cfa wsp, 464
	ldp x20, x19, [sp, #448]
	ldp x22, x21, [sp, #432]
	ldp x28, x23, [sp, #416]
	ldp x29, x30, [sp, #400]
	ldp d9, d8, [sp, #384]
	ldp d11, d10, [sp, #368]
	ldp d13, d12, [sp, #352]
	ldp d15, d14, [sp, #336]
	add sp, sp, #464
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
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
.LBB6_14:
	.cfi_restore_state
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB6_2
.LBB6_15:
	ldr q0, [sp, #176]
	.cfi_def_cfa wsp, 464
	ldp x20, x19, [sp, #448]
	ldp x22, x21, [sp, #432]
	ldp x28, x23, [sp, #416]
	ldp x29, x30, [sp, #400]
	ldp d9, d8, [sp, #384]
	ldp d11, d10, [sp, #368]
	ldp d13, d12, [sp, #352]
	ldp d15, d14, [sp, #336]
	add sp, sp, #464
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
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
