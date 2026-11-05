.section .text.stub_linear_to_gamma_slice,"ax",@progbits
	.globl	stub_linear_to_gamma_slice
	.p2align	2
.type	stub_linear_to_gamma_slice,@function
stub_linear_to_gamma_slice:
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
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB7_15
	cmp w8, #2
	b.ne .LBB7_14
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
	sub x8, x29, #128
	fmov s8, s0
	add x0, x8, x20
	bl memset
	sub x0, x29, #128
	mov x1, x19
	mov x2, x20
	bl memcpy
	ldp q1, q2, [x29, #-128]
	fmov s0, s8
	mov x0, x23
	mov w9, #1
	stp q1, q2, [x29, #-192]
	ldp q1, q3, [x29, #-96]
	stp q1, q3, [x29, #-160]
	b .LBB7_5
.LBB7_4:
	mov w9, wzr
.LBB7_5:
	fmov s1, #1.00000000
	mov w8, #64269
	mov w10, #14483
	movk w8, #74, lsl #16
	movk w10, #16246, lsl #16
	fmov v2.4s, #1.00000000
	dup v4.4s, w8
	mov w8, #1267
	cmp x21, #0
	fdiv s0, s1, s0
	movk w8, #16181, lsl #16
	mvni v1.4s, #63, msl #16
	dup v3.4s, w8
	mov w8, #27455
	stur w9, [x29, #-196]
	movk w8, #16094, lsl #16
	stp q3, q4, [sp, #160]
	dup v4.4s, w8
	mov w8, #46233
	movk w8, #16147, lsl #16
	dup v3.4s, w8
	mov w8, #43579
	movk w8, #16440, lsl #16
	stp q3, q4, [sp, #112]
	dup v4.4s, w10
	dup v3.4s, w8
	mov w8, #-1023672320
	mov w10, #38156
	dup v22.4s, w8
	mov w8, #1123942400
	movk w10, #15389, lsl #16
	stp q3, q4, [sp, #80]
	fneg v3.4s, v1.4s
	str q0, [sp, #144]
	mvni v0.4s, #127, msl #16
	fneg v20.4s, v0.4s
	dup v0.4s, w8
	mov w8, #63802
	movk w8, #14625, lsl #16
	dup v1.4s, w8
	mov w8, #50297
	movk w8, #15022, lsl #16
	stp q0, q3, [sp, #48]
	dup v0.4s, w8
	mov w8, #22716
	movk w8, #15715, lsl #16
	dup v28.4s, w8
	mov w8, #65005
	stp q0, q1, [sp, #16]
	dup v0.4s, w10
	mov w10, #29208
	movk w8, #15989, lsl #16
	movk w10, #16177, lsl #16
	dup v29.4s, w8
	dup v30.4s, w10
	sub x10, x29, #196
	add x8, x0, x22, lsl #6
	add x1, x10, #4
	str q0, [sp]
	csel x9, xzr, x1, eq
	b .LBB7_8
.LBB7_6:
	add x10, x0, #64
.LBB7_7:
	movi v0.2d, #0000000000000000
	ldp q1, q6, [x0]
	ldp q7, q3, [sp, #160]
	movi v5.4s, #127, msl #16
	ldp q9, q10, [x0, #32]
	fmov v17.4s, #-1.00000000
	mvni v21.4s, #126
	fmax v1.4s, v1.4s, v0.4s
	fmax v9.4s, v9.4s, v0.4s
	fmax v10.4s, v10.4s, v0.4s
	fmin v31.4s, v1.4s, v2.4s
	fmax v1.4s, v6.4s, v0.4s
	fmin v9.4s, v9.4s, v2.4s
	fmin v10.4s, v10.4s, v2.4s
	add v4.4s, v31.4s, v3.4s
	fmin v8.4s, v1.4s, v2.4s
	add v13.4s, v9.4s, v3.4s
	and v6.16b, v4.16b, v5.16b
	add v16.4s, v8.4s, v3.4s
	and v14.16b, v13.16b, v5.16b
	add v1.4s, v6.4s, v7.4s
	and v11.16b, v16.16b, v5.16b
	ssra v21.4s, v16.4s, #23
	fadd v6.4s, v1.4s, v17.4s
	fadd v1.4s, v1.4s, v2.4s
	scvtf v21.4s, v21.4s
	fdiv v6.4s, v6.4s, v1.4s
	add v1.4s, v11.4s, v7.4s
	fadd v11.4s, v1.4s, v17.4s
	fadd v1.4s, v1.4s, v2.4s
	fdiv v12.4s, v11.4s, v1.4s
	add v1.4s, v14.4s, v7.4s
	add v14.4s, v10.4s, v3.4s
	fadd v11.4s, v1.4s, v17.4s
	fadd v1.4s, v1.4s, v2.4s
	and v15.16b, v14.16b, v5.16b
	fmul v27.4s, v6.4s, v6.4s
	fdiv v11.4s, v11.4s, v1.4s
	add v1.4s, v15.4s, v7.4s
	mvni v7.4s, #126
	fadd v15.4s, v1.4s, v17.4s
	fadd v1.4s, v1.4s, v2.4s
	fmul v3.4s, v12.4s, v12.4s
	ldp q17, q24, [sp, #112]
	ssra v7.4s, v4.4s, #23
	mvni v4.4s, #126
	mov v23.16b, v17.16b
	mov v5.16b, v17.16b
	mov v25.16b, v17.16b
	scvtf v7.4s, v7.4s
	ssra v4.4s, v13.4s, #23
	fdiv v1.4s, v15.4s, v1.4s
	ldp q15, q18, [sp, #80]
	fmla v23.4s, v24.4s, v27.4s
	fmla v5.4s, v24.4s, v3.4s
	scvtf v4.4s, v4.4s
	mov v26.16b, v18.16b
	mov v16.16b, v15.16b
	fmul v0.4s, v11.4s, v11.4s
	fmla v26.4s, v27.4s, v23.4s
	mov v23.16b, v18.16b
	fmla v25.4s, v24.4s, v0.4s
	fmla v23.4s, v3.4s, v5.4s
	mov v5.16b, v17.16b
	mov v17.16b, v18.16b
	fmla v16.4s, v27.4s, v26.4s
	mov v26.16b, v15.16b
	fmul v19.4s, v1.4s, v1.4s
	fmla v26.4s, v3.4s, v23.4s
	mvni v3.4s, #126
	fmla v7.4s, v6.4s, v16.4s
	fcmeq v6.4s, v31.4s, #0.0
	mvni v23.4s, #127, msl #16
	fcmlt v16.4s, v10.4s, #0.0
	ssra v3.4s, v14.4s, #23
	fmla v5.4s, v24.4s, v19.4s
	mov v24.16b, v18.16b
	fcmlt v18.4s, v31.4s, #0.0
	fmla v21.4s, v12.4s, v26.4s
	bsl v6.16b, v23.16b, v7.16b
	fcmeq v7.4s, v9.4s, #0.0
	scvtf v3.4s, v3.4s
	fmla v24.4s, v0.4s, v25.4s
	fmla v17.4s, v19.4s, v5.4s
	mov v5.16b, v15.16b
	fmla v5.4s, v0.4s, v24.4s
	fcmeq v0.4s, v8.4s, #0.0
	movi v24.4s, #67, lsl #24
	fmla v15.4s, v19.4s, v17.4s
	fcmlt v17.4s, v8.4s, #0.0
	fcmeq v19.4s, v31.4s, v20.4s
	fmla v4.4s, v11.4s, v5.4s
	bsl v0.16b, v23.16b, v21.16b
	ldr q21, [sp, #64]
	fmla v3.4s, v1.4s, v15.4s
	mov v1.16b, v18.16b
	fcmeq v5.4s, v10.4s, #0.0
	bsl v1.16b, v21.16b, v6.16b
	bit v4.16b, v23.16b, v7.16b
	fcmeq v7.4s, v8.4s, v20.4s
	bit v0.16b, v21.16b, v17.16b
	fcmlt v6.4s, v9.4s, #0.0
	bit v3.16b, v23.16b, v5.16b
	fcmeq v5.4s, v9.4s, v20.4s
	bit v1.16b, v20.16b, v19.16b
	ldp q19, q17, [sp]
	bit v0.16b, v20.16b, v7.16b
	ldr q7, [sp, #144]
	bit v4.16b, v21.16b, v6.16b
	fcmeq v6.4s, v10.4s, v20.4s
	bit v3.16b, v21.16b, v16.16b
	ldp q18, q16, [sp, #32]
	fmul v10.4s, v1.4s, v7.s[0]
	mov v1.16b, v5.16b
	fmul v9.4s, v0.4s, v7.s[0]
	mov v0.16b, v6.16b
	bsl v1.16b, v20.16b, v4.16b
	fcmge v23.4s, v10.4s, v24.4s
	bsl v0.16b, v20.16b, v3.16b
	fmax v3.4s, v10.4s, v22.4s
	fmul v8.4s, v1.4s, v7.s[0]
	fmax v1.4s, v9.4s, v22.4s
	fmul v31.4s, v0.4s, v7.s[0]
	fmin v0.4s, v3.4s, v24.4s
	fmax v3.4s, v8.4s, v22.4s
	fmin v4.4s, v1.4s, v24.4s
	frintn v5.4s, v0.4s
	fmax v1.4s, v31.4s, v22.4s
	fmin v3.4s, v3.4s, v24.4s
	frintn v6.4s, v4.4s
	fmin v15.4s, v5.4s, v16.4s
	fmin v7.4s, v1.4s, v24.4s
	frintn v1.4s, v3.4s
	fmin v14.4s, v6.4s, v16.4s
	fsub v6.4s, v0.4s, v15.4s
	frintn v5.4s, v7.4s
	mov v0.16b, v17.16b
	fmin v1.4s, v1.4s, v16.4s
	fsub v4.4s, v4.4s, v14.4s
	fmla v0.4s, v18.4s, v6.4s
	fmin v13.4s, v5.4s, v16.4s
	mov v5.16b, v17.16b
	mov v16.16b, v17.16b
	fsub v12.4s, v3.4s, v1.4s
	mov v3.16b, v19.16b
	fcvtns v1.4s, v1.4s
	fmla v5.4s, v18.4s, v4.4s
	fsub v11.4s, v7.4s, v13.4s
	mov v7.16b, v19.16b
	fcvtns v21.4s, v13.4s
	fmla v3.4s, v6.4s, v0.4s
	mov v0.16b, v17.16b
	mov v17.16b, v28.16b
	shl v1.4s, v1.4s, #23
	fmla v7.4s, v4.4s, v5.4s
	mov v5.16b, v28.16b
	fmla v0.4s, v18.4s, v12.4s
	fmla v16.4s, v18.4s, v11.4s
	mov v18.16b, v29.16b
	fmla v17.4s, v6.4s, v3.4s
	mov v3.16b, v19.16b
	shl v21.4s, v21.4s, #23
	add v1.4s, v1.4s, v2.4s
	fmla v5.4s, v4.4s, v7.4s
	mov v7.16b, v28.16b
	fmla v3.4s, v12.4s, v0.4s
	mov v0.16b, v19.16b
	mov v19.16b, v30.16b
	fmla v18.4s, v6.4s, v17.4s
	mov v17.16b, v29.16b
	add v21.4s, v21.4s, v2.4s
	fmla v0.4s, v11.4s, v16.4s
	fmov v16.4s, #1.00000000
	fmla v7.4s, v12.4s, v3.4s
	mov v3.16b, v28.16b
	fmla v17.4s, v4.4s, v5.4s
	fmla v19.4s, v6.4s, v18.4s
	mov v5.16b, v30.16b
	fcvtns v18.4s, v15.4s
	fmla v3.4s, v11.4s, v0.4s
	mov v0.16b, v29.16b
	fmla v5.4s, v4.4s, v17.4s
	fmov v17.4s, #1.00000000
	fmla v16.4s, v6.4s, v19.4s
	mov v6.16b, v30.16b
	fcvtns v19.4s, v14.4s
	fmla v0.4s, v12.4s, v7.4s
	mov v7.16b, v29.16b
	shl v18.4s, v18.4s, #23
	fmla v17.4s, v4.4s, v5.4s
	fmov v4.4s, #1.00000000
	fcmgt v5.4s, v22.4s, v10.4s
	fmla v7.4s, v11.4s, v3.4s
	mov v3.16b, v30.16b
	shl v19.4s, v19.4s, #23
	fcmge v10.4s, v31.4s, v24.4s
	add v18.4s, v18.4s, v2.4s
	fcmgt v31.4s, v22.4s, v31.4s
	fmla v3.4s, v12.4s, v0.4s
	fmov v0.4s, #1.00000000
	add v19.4s, v19.4s, v2.4s
	fmla v6.4s, v11.4s, v7.4s
	fcmgt v7.4s, v22.4s, v9.4s
	fcmge v9.4s, v9.4s, v24.4s
	fmul v16.4s, v16.4s, v18.4s
	fmla v0.4s, v12.4s, v3.4s
	fcmgt v3.4s, v22.4s, v8.4s
	fcmge v8.4s, v8.4s, v24.4s
	fmla v4.4s, v11.4s, v6.4s
	mvn v6.16b, v23.16b
	mvn v11.16b, v9.16b
	mvn v18.16b, v8.16b
	bic v5.16b, v6.16b, v5.16b
	fmul v6.4s, v17.4s, v19.4s
	mvn v17.16b, v10.16b
	fmul v0.4s, v0.4s, v1.4s
	bic v7.16b, v11.16b, v7.16b
	bic v1.16b, v18.16b, v3.16b
	fmul v3.4s, v4.4s, v21.4s
	and v5.16b, v5.16b, v16.16b
	bic v4.16b, v17.16b, v31.16b
	and v16.16b, v23.16b, v20.16b
	and v6.16b, v7.16b, v6.16b
	and v7.16b, v9.16b, v20.16b
	and v0.16b, v1.16b, v0.16b
	and v1.16b, v8.16b, v20.16b
	and v3.16b, v4.16b, v3.16b
	orr v4.16b, v5.16b, v16.16b
	and v5.16b, v10.16b, v20.16b
	orr v6.16b, v6.16b, v7.16b
	orr v0.16b, v0.16b, v1.16b
	orr v1.16b, v3.16b, v5.16b
	stp q4, q6, [x0]
	stp q0, q1, [x0, #32]
	mov x0, x10
.LBB7_8:
	cmp x0, x8
	ccmp x0, #0, #4, ne
	b.ne .LBB7_6
	cbz x9, .LBB7_11
	mov x10, xzr
	mov x0, x9
	mov x9, xzr
	b .LBB7_7
.LBB7_11:
	ldur w8, [x29, #-196]
	cbz w8, .LBB7_13
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB7_13:
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
.LBB7_14:
	.cfi_restore_state
	mov x19, x0
	fmov s8, s0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	fmov s0, s8
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB7_2
.LBB7_15:
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
	b linear_srgb::simd::linear_to_gamma_slice_tier_scalar_t
