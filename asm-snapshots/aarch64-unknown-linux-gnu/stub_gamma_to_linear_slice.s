.section .text.stub_gamma_to_linear_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_slice
	.p2align	2
.type	stub_gamma_to_linear_slice,@function
stub_gamma_to_linear_slice:
	.cfi_startproc
	sub sp, sp, #448
	.cfi_def_cfa_offset 448
	stp d15, d14, [sp, #320]
	stp d13, d12, [sp, #336]
	stp d11, d10, [sp, #352]
	stp d9, d8, [sp, #368]
	stp x29, x30, [sp, #384]
	stp x28, x23, [sp, #400]
	stp x22, x21, [sp, #416]
	stp x20, x19, [sp, #432]
	add x29, sp, #384
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
	str q0, [sp, #160]
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
	stp q0, q1, [sp, #192]
	ldp q0, q2, [x29, #-96]
	stp q0, q2, [sp, #224]
	b .LBB6_5
.LBB6_4:
	mov w9, wzr
.LBB6_5:
	mov w8, #64269
	mov w10, #1267
	mvni v0.4s, #127, msl #16
	movk w8, #74, lsl #16
	movk w10, #16181, lsl #16
	mvni v20.4s, #63, msl #16
	dup v3.4s, w8
	dup v2.4s, w10
	mov w8, #27455
	mov w10, #46233
	movk w8, #16094, lsl #16
	fneg v6.4s, v0.4s
	movk w10, #16147, lsl #16
	fmov v1.4s, #1.00000000
	cmp x21, #0
	stp q2, q3, [sp, #128]
	dup v2.4s, w8
	dup v0.4s, w10
	mov w8, #14483
	mov w10, #43579
	str w9, [sp, #188]
	movk w8, #16246, lsl #16
	movk w10, #16440, lsl #16
	stp q0, q2, [sp, #96]
	dup v2.4s, w8
	dup v0.4s, w10
	mov w10, #1123942400
	mov w8, #-1023672320
	dup v22.4s, w8
	mov w8, #63802
	stp q0, q2, [sp, #64]
	fneg v2.4s, v20.4s
	dup v0.4s, w10
	movk w8, #14625, lsl #16
	mov w10, #38156
	movk w10, #15389, lsl #16
	dup v26.4s, w10
	mov w10, #29208
	stp q0, q2, [sp, #32]
	dup v2.4s, w8
	mov w8, #50297
	movk w8, #15022, lsl #16
	movk w10, #16177, lsl #16
	dup v0.4s, w8
	mov w8, #22716
	dup v29.4s, w10
	movk w8, #15715, lsl #16
	add x10, sp, #188
	dup v27.4s, w8
	mov w8, #65005
	add x1, x10, #4
	movk w8, #15989, lsl #16
	csel x9, xzr, x1, eq
	stp q0, q2, [sp]
	dup v28.4s, w8
	add x8, x0, x22, lsl #6
	b .LBB6_8
.LBB6_6:
	add x10, x0, #64
.LBB6_7:
	movi v0.2d, #0000000000000000
	ldr q30, [x0, #48]
	movi v3.4s, #127, msl #16
	ldp q5, q2, [sp, #128]
	fmov v7.4s, #-1.00000000
	ldp q12, q8, [x0]
	mvni v16.4s, #126
	ldp q19, q23, [sp, #96]
	mvni v21.4s, #126
	fmax v30.4s, v30.4s, v0.4s
	ldp q25, q24, [sp, #64]
	fmax v8.4s, v8.4s, v0.4s
	fmax v12.4s, v12.4s, v0.4s
	mov v17.16b, v19.16b
	mov v18.16b, v24.16b
	fmin v30.4s, v30.4s, v1.4s
	fmin v8.4s, v8.4s, v1.4s
	add v31.4s, v30.4s, v2.4s
	and v9.16b, v31.16b, v3.16b
	ssra v16.4s, v31.4s, #23
	add v9.4s, v9.4s, v5.4s
	scvtf v16.4s, v16.4s
	fadd v10.4s, v9.4s, v7.4s
	fadd v11.4s, v9.4s, v1.4s
	add v9.4s, v8.4s, v2.4s
	and v13.16b, v9.16b, v3.16b
	ssra v21.4s, v9.4s, #23
	mvni v9.4s, #126
	fdiv v14.4s, v10.4s, v11.4s
	fmin v10.4s, v12.4s, v1.4s
	add v11.4s, v13.4s, v5.4s
	ldr q13, [x0, #32]
	fmax v13.4s, v13.4s, v0.4s
	add v15.4s, v10.4s, v2.4s
	fadd v12.4s, v11.4s, v7.4s
	fadd v11.4s, v11.4s, v1.4s
	and v0.16b, v15.16b, v3.16b
	ssra v9.4s, v15.4s, #23
	fdiv v12.4s, v12.4s, v11.4s
	fmin v11.4s, v13.4s, v1.4s
	add v0.4s, v0.4s, v5.4s
	fadd v13.4s, v0.4s, v7.4s
	fadd v0.4s, v0.4s, v1.4s
	add v2.4s, v11.4s, v2.4s
	and v4.16b, v2.16b, v3.16b
	mov v3.16b, v19.16b
	add v4.4s, v4.4s, v5.4s
	fdiv v0.4s, v13.4s, v0.4s
	mov v5.16b, v24.16b
	fadd v13.4s, v4.4s, v7.4s
	fadd v4.4s, v4.4s, v1.4s
	fmul v7.4s, v12.4s, v12.4s
	fmla v3.4s, v23.4s, v7.4s
	fdiv v4.4s, v13.4s, v4.4s
	fmul v13.4s, v14.4s, v14.4s
	fmla v5.4s, v7.4s, v3.4s
	mov v3.16b, v25.16b
	fmul v20.4s, v0.4s, v0.4s
	fmla v17.4s, v23.4s, v13.4s
	fmla v18.4s, v13.4s, v17.4s
	mov v17.16b, v19.16b
	fmla v17.4s, v23.4s, v20.4s
	fmla v3.4s, v13.4s, v18.4s
	mov v18.16b, v25.16b
	mov v13.16b, v24.16b
	fmul v31.4s, v4.4s, v4.4s
	fmla v18.4s, v7.4s, v5.4s
	fmla v13.4s, v20.4s, v17.4s
	scvtf v7.4s, v21.4s
	mvni v5.4s, #126
	fmla v16.4s, v14.4s, v3.4s
	fcmeq v17.4s, v30.4s, #0.0
	fmla v19.4s, v23.4s, v31.4s
	fcmlt v21.4s, v30.4s, #0.0
	fcmeq v3.4s, v10.4s, #0.0
	movi v23.4s, #67, lsl #24
	ssra v5.4s, v2.4s, #23
	scvtf v2.4s, v9.4s
	fmla v7.4s, v12.4s, v18.4s
	mvni v9.4s, #127, msl #16
	ldr q18, [sp, #160]
	fmla v24.4s, v31.4s, v19.4s
	mov v19.16b, v25.16b
	scvtf v5.4s, v5.4s
	bit v16.16b, v9.16b, v17.16b
	fcmeq v17.4s, v30.4s, v6.4s
	fmla v19.4s, v20.4s, v13.4s
	fcmeq v20.4s, v8.4s, #0.0
	fmla v25.4s, v31.4s, v24.4s
	fmla v2.4s, v0.4s, v19.4s
	fcmlt v0.4s, v8.4s, #0.0
	bit v7.16b, v9.16b, v20.16b
	ldr q20, [sp, #48]
	fcmeq v19.4s, v11.4s, #0.0
	fmla v5.4s, v4.4s, v25.4s
	fcmlt v4.4s, v10.4s, #0.0
	ldr q25, [sp]
	bit v16.16b, v20.16b, v21.16b
	mov v21.16b, v26.16b
	bsl v0.16b, v20.16b, v7.16b
	mov v7.16b, v17.16b
	bit v2.16b, v9.16b, v3.16b
	fcmeq v3.4s, v8.4s, v6.4s
	bit v5.16b, v9.16b, v19.16b
	fcmeq v17.4s, v10.4s, v6.4s
	ldp q24, q19, [sp, #16]
	mov v10.16b, v27.16b
	bsl v7.16b, v6.16b, v16.16b
	fcmlt v16.4s, v11.4s, #0.0
	bit v2.16b, v20.16b, v4.16b
	bit v0.16b, v6.16b, v3.16b
	fcmeq v3.4s, v11.4s, v6.4s
	mov v11.16b, v27.16b
	fmul v9.4s, v7.4s, v18.s[0]
	mov v4.16b, v16.16b
	bit v2.16b, v6.16b, v17.16b
	fmul v31.4s, v0.4s, v18.s[0]
	bsl v4.16b, v20.16b, v5.16b
	mov v20.16b, v26.16b
	fmax v0.4s, v9.4s, v22.4s
	fmul v8.4s, v2.4s, v18.s[0]
	fmax v2.4s, v31.4s, v22.4s
	bsl v3.16b, v6.16b, v4.16b
	fmin v0.4s, v0.4s, v23.4s
	fmin v2.4s, v2.4s, v23.4s
	fmul v30.4s, v3.4s, v18.s[0]
	fmax v3.4s, v8.4s, v22.4s
	mov v18.16b, v25.16b
	frintn v4.4s, v0.4s
	frintn v7.4s, v2.4s
	fmin v3.4s, v3.4s, v23.4s
	fmax v5.4s, v30.4s, v22.4s
	fmin v4.4s, v4.4s, v19.4s
	fmin v7.4s, v7.4s, v19.4s
	frintn v16.4s, v3.4s
	fmin v5.4s, v5.4s, v23.4s
	fsub v0.4s, v0.4s, v4.4s
	fcvtns v4.4s, v4.4s
	fsub v2.4s, v2.4s, v7.4s
	fcvtns v7.4s, v7.4s
	fmin v16.4s, v16.4s, v19.4s
	frintn v17.4s, v5.4s
	fmla v18.4s, v24.4s, v0.4s
	shl v4.4s, v4.4s, #23
	shl v7.4s, v7.4s, #23
	fsub v3.4s, v3.4s, v16.4s
	fmin v17.4s, v17.4s, v19.4s
	mov v19.16b, v25.16b
	fmla v20.4s, v0.4s, v18.4s
	mov v18.16b, v25.16b
	fcvtns v16.4s, v16.4s
	add v7.4s, v7.4s, v1.4s
	add v4.4s, v4.4s, v1.4s
	fmla v19.4s, v24.4s, v2.4s
	fmla v18.4s, v24.4s, v3.4s
	fsub v5.4s, v5.4s, v17.4s
	fcvtns v17.4s, v17.4s
	fmla v10.4s, v0.4s, v20.4s
	mov v20.16b, v26.16b
	shl v16.4s, v16.4s, #23
	fmla v21.4s, v2.4s, v19.4s
	mov v19.16b, v25.16b
	fmla v20.4s, v3.4s, v18.4s
	mov v18.16b, v28.16b
	shl v17.4s, v17.4s, #23
	add v16.4s, v16.4s, v1.4s
	fmla v19.4s, v24.4s, v5.4s
	fmla v18.4s, v0.4s, v10.4s
	mov v10.16b, v27.16b
	fmla v11.4s, v2.4s, v21.4s
	mov v21.16b, v26.16b
	add v17.4s, v17.4s, v1.4s
	fmla v10.4s, v3.4s, v20.4s
	mov v20.16b, v29.16b
	fmla v21.4s, v5.4s, v19.4s
	mov v19.16b, v28.16b
	fmla v20.4s, v0.4s, v18.4s
	mov v18.16b, v27.16b
	fmla v19.4s, v2.4s, v11.4s
	fmov v11.4s, #1.00000000
	fmla v18.4s, v5.4s, v21.4s
	mov v21.16b, v28.16b
	fmla v11.4s, v0.4s, v20.4s
	mov v0.16b, v28.16b
	mov v20.16b, v29.16b
	fmla v21.4s, v3.4s, v10.4s
	mov v10.16b, v29.16b
	fmla v0.4s, v5.4s, v18.4s
	mov v18.16b, v29.16b
	fmla v10.4s, v2.4s, v19.4s
	fmov v19.4s, #1.00000000
	fmul v4.4s, v11.4s, v4.4s
	fmla v18.4s, v3.4s, v21.4s
	fmov v21.4s, #1.00000000
	fmla v20.4s, v5.4s, v0.4s
	fcmge v0.4s, v8.4s, v23.4s
	fcmgt v8.4s, v22.4s, v8.4s
	fmla v19.4s, v2.4s, v10.4s
	fmov v2.4s, #1.00000000
	fcmge v10.4s, v9.4s, v23.4s
	fcmgt v9.4s, v22.4s, v9.4s
	fmla v21.4s, v3.4s, v18.4s
	fcmge v3.4s, v31.4s, v23.4s
	fcmge v18.4s, v30.4s, v23.4s
	fcmgt v31.4s, v22.4s, v31.4s
	fmla v2.4s, v5.4s, v20.4s
	mvn v5.16b, v0.16b
	fcmgt v20.4s, v22.4s, v30.4s
	fmul v7.4s, v19.4s, v7.4s
	mvn v19.16b, v10.16b
	and v0.16b, v0.16b, v6.16b
	fmul v16.4s, v21.4s, v16.4s
	mvn v30.16b, v3.16b
	mvn v21.16b, v18.16b
	bic v5.16b, v5.16b, v8.16b
	and v3.16b, v3.16b, v6.16b
	fmul v2.4s, v2.4s, v17.4s
	bic v19.16b, v19.16b, v9.16b
	bic v30.16b, v30.16b, v31.16b
	bic v17.16b, v21.16b, v20.16b
	and v5.16b, v5.16b, v16.16b
	and v16.16b, v18.16b, v6.16b
	and v4.16b, v19.16b, v4.16b
	and v7.16b, v30.16b, v7.16b
	and v2.16b, v17.16b, v2.16b
	orr v0.16b, v5.16b, v0.16b
	and v5.16b, v10.16b, v6.16b
	orr v3.16b, v7.16b, v3.16b
	orr v2.16b, v2.16b, v16.16b
	orr v4.16b, v4.16b, v5.16b
	stp q0, q3, [x0]
	stp q2, q4, [x0, #32]
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
	ldr w8, [sp, #188]
	cbz w8, .LBB6_13
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB6_13:
	.cfi_def_cfa wsp, 448
	ldp x20, x19, [sp, #432]
	ldp x22, x21, [sp, #416]
	ldp x28, x23, [sp, #400]
	ldp x29, x30, [sp, #384]
	ldp d9, d8, [sp, #368]
	ldp d11, d10, [sp, #352]
	ldp d13, d12, [sp, #336]
	ldp d15, d14, [sp, #320]
	add sp, sp, #448
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
	ldr q0, [sp, #160]
	.cfi_def_cfa wsp, 448
	ldp x20, x19, [sp, #432]
	ldp x22, x21, [sp, #416]
	ldp x28, x23, [sp, #400]
	ldp x29, x30, [sp, #384]
	ldp d9, d8, [sp, #368]
	ldp d11, d10, [sp, #352]
	ldp d13, d12, [sp, #336]
	ldp d15, d14, [sp, #320]
	add sp, sp, #448
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
