.section .text.stub_linear_to_gamma_slice,"ax",@progbits
	.globl	stub_linear_to_gamma_slice
	.p2align	2
.type	stub_linear_to_gamma_slice,@function
stub_linear_to_gamma_slice:
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
	stp q1, q2, [sp, #192]
	ldp q1, q3, [x29, #-96]
	stp q1, q3, [sp, #224]
	b .LBB7_5
.LBB7_4:
	mov w9, wzr
.LBB7_5:
	fmov s7, #1.00000000
	mov w8, #64269
	mvni v1.4s, #127, msl #16
	movk w8, #74, lsl #16
	mov w10, #43579
	fmov v3.4s, #1.00000000
	dup v2.4s, w8
	mov w8, #1267
	movk w10, #16440, lsl #16
	fdiv s0, s7, s0
	movk w8, #16181, lsl #16
	fneg v5.4s, v1.4s
	cmp x21, #0
	str w9, [sp, #188]
	stp q0, q2, [sp, #144]
	dup v2.4s, w8
	mov w8, #27455
	movk w8, #16094, lsl #16
	mvni v0.4s, #63, msl #16
	dup v1.4s, w8
	mov w8, #46233
	movk w8, #16147, lsl #16
	fneg v0.4s, v0.4s
	stp q1, q2, [sp, #112]
	dup v2.4s, w8
	mov w8, #14483
	movk w8, #16246, lsl #16
	dup v1.4s, w8
	mov w8, #-1023672320
	dup v23.4s, w8
	mov w8, #63802
	movk w8, #14625, lsl #16
	stp q1, q2, [sp, #80]
	dup v1.4s, w10
	mov w10, #1123942400
	stp q0, q1, [sp, #48]
	dup v1.4s, w10
	dup v0.4s, w8
	mov w8, #50297
	mov w10, #38156
	movk w8, #15022, lsl #16
	movk w10, #15389, lsl #16
	stp q0, q1, [sp, #16]
	dup v0.4s, w8
	mov w8, #22716
	movk w8, #15715, lsl #16
	dup v27.4s, w10
	mov w10, #29208
	dup v28.4s, w8
	mov w8, #65005
	movk w10, #16177, lsl #16
	movk w8, #15989, lsl #16
	dup v30.4s, w10
	add x10, sp, #188
	dup v29.4s, w8
	add x8, x0, x22, lsl #6
	add x1, x10, #4
	csel x9, xzr, x1, eq
	str q0, [sp]
	b .LBB7_8
.LBB7_6:
	add x10, x0, #64
.LBB7_7:
	movi v0.2d, #0000000000000000
	ldr q31, [x0, #48]
	ldr q4, [sp, #160]
	ldp q13, q9, [x0]
	movi v7.4s, #127, msl #16
	ldr q17, [sp, #128]
	fmov v18.4s, #-1.00000000
	mvni v22.4s, #126
	ldp q20, q24, [sp, #96]
	fmax v31.4s, v31.4s, v0.4s
	fmax v9.4s, v9.4s, v0.4s
	fmax v13.4s, v13.4s, v0.4s
	ldp q26, q21, [sp, #64]
	fmin v31.4s, v31.4s, v3.4s
	fmin v9.4s, v9.4s, v3.4s
	mov v19.16b, v21.16b
	mov v25.16b, v21.16b
	add v8.4s, v31.4s, v4.4s
	and v10.16b, v8.16b, v7.16b
	add v10.4s, v10.4s, v17.4s
	fadd v11.4s, v10.4s, v18.4s
	fadd v12.4s, v10.4s, v3.4s
	add v10.4s, v9.4s, v4.4s
	and v14.16b, v10.16b, v7.16b
	ssra v22.4s, v10.4s, #23
	mvni v10.4s, #126
	fdiv v15.4s, v11.4s, v12.4s
	fmin v11.4s, v13.4s, v3.4s
	add v12.4s, v14.4s, v17.4s
	ldr q14, [x0, #32]
	fmax v14.4s, v14.4s, v0.4s
	add v2.4s, v11.4s, v4.4s
	mov v0.16b, v20.16b
	fadd v13.4s, v12.4s, v18.4s
	fadd v12.4s, v12.4s, v3.4s
	and v1.16b, v2.16b, v7.16b
	ssra v10.4s, v2.4s, #23
	mvni v2.4s, #126
	fdiv v13.4s, v13.4s, v12.4s
	fmin v12.4s, v14.4s, v3.4s
	add v1.4s, v1.4s, v17.4s
	fadd v14.4s, v1.4s, v18.4s
	fadd v1.4s, v1.4s, v3.4s
	add v6.4s, v12.4s, v4.4s
	and v16.16b, v6.16b, v7.16b
	ssra v2.4s, v6.4s, #23
	scvtf v6.4s, v10.4s
	add v16.4s, v16.4s, v17.4s
	fdiv v1.4s, v14.4s, v1.4s
	mvni v17.4s, #126
	scvtf v2.4s, v2.4s
	fadd v14.4s, v16.4s, v18.4s
	fadd v16.4s, v16.4s, v3.4s
	fmul v4.4s, v13.4s, v13.4s
	mov v18.16b, v20.16b
	ssra v17.4s, v8.4s, #23
	fmla v0.4s, v24.4s, v4.4s
	scvtf v17.4s, v17.4s
	fdiv v16.4s, v14.4s, v16.4s
	fmul v14.4s, v15.4s, v15.4s
	fmla v25.4s, v4.4s, v0.4s
	mov v0.16b, v26.16b
	fmul v7.4s, v1.4s, v1.4s
	fmla v18.4s, v24.4s, v14.4s
	fmla v19.4s, v14.4s, v18.4s
	mov v18.16b, v20.16b
	fmla v18.4s, v24.4s, v7.4s
	fmla v0.4s, v14.4s, v19.4s
	mov v19.16b, v26.16b
	mov v14.16b, v21.16b
	fmul v8.4s, v16.4s, v16.4s
	fmla v19.4s, v4.4s, v25.4s
	fmla v14.4s, v7.4s, v18.4s
	scvtf v4.4s, v22.4s
	fmla v17.4s, v15.4s, v0.4s
	mov v0.16b, v21.16b
	fcmeq v18.4s, v31.4s, #0.0
	fmla v20.4s, v24.4s, v8.4s
	mvni v22.4s, #127, msl #16
	fcmlt v21.4s, v31.4s, #0.0
	movi v24.4s, #67, lsl #24
	fmla v4.4s, v13.4s, v19.4s
	ldp q25, q19, [sp, #16]
	bit v17.16b, v22.16b, v18.16b
	fcmeq v18.4s, v12.4s, #0.0
	fmla v0.4s, v8.4s, v20.4s
	mov v20.16b, v26.16b
	fmla v20.4s, v7.4s, v14.4s
	fcmeq v7.4s, v9.4s, #0.0
	fmla v26.4s, v8.4s, v0.4s
	fcmeq v0.4s, v11.4s, #0.0
	fmla v6.4s, v1.4s, v20.4s
	fcmlt v1.4s, v9.4s, #0.0
	bit v4.16b, v22.16b, v7.16b
	fcmeq v7.4s, v31.4s, v5.4s
	ldr q20, [sp, #48]
	fmla v2.4s, v16.4s, v26.4s
	fcmlt v16.4s, v11.4s, #0.0
	ldr q26, [sp]
	bit v17.16b, v20.16b, v21.16b
	mov v21.16b, v27.16b
	bsl v1.16b, v20.16b, v4.16b
	bsl v0.16b, v22.16b, v6.16b
	fcmeq v6.4s, v9.4s, v5.4s
	mov v4.16b, v7.16b
	fcmlt v7.4s, v12.4s, #0.0
	bit v2.16b, v22.16b, v18.16b
	mov v18.16b, v26.16b
	mov v22.16b, v28.16b
	bsl v4.16b, v5.16b, v17.16b
	bit v1.16b, v5.16b, v6.16b
	ldr q6, [sp, #144]
	fcmeq v17.4s, v11.4s, v5.4s
	bit v0.16b, v20.16b, v16.16b
	bit v2.16b, v20.16b, v7.16b
	mov v20.16b, v27.16b
	mov v11.16b, v28.16b
	fmul v10.4s, v4.4s, v6.s[0]
	fcmeq v4.4s, v12.4s, v5.4s
	fmul v8.4s, v1.4s, v6.s[0]
	bit v0.16b, v5.16b, v17.16b
	fmax v1.4s, v10.4s, v23.4s
	bit v2.16b, v5.16b, v4.16b
	fmul v9.4s, v0.4s, v6.s[0]
	fmax v0.4s, v8.4s, v23.4s
	fmin v1.4s, v1.4s, v24.4s
	fmul v31.4s, v2.4s, v6.s[0]
	fmax v2.4s, v9.4s, v23.4s
	fmin v0.4s, v0.4s, v24.4s
	frintn v4.4s, v1.4s
	fmax v6.4s, v31.4s, v23.4s
	fmin v2.4s, v2.4s, v24.4s
	frintn v7.4s, v0.4s
	fmin v4.4s, v4.4s, v19.4s
	fmin v6.4s, v6.4s, v24.4s
	frintn v16.4s, v2.4s
	fmin v7.4s, v7.4s, v19.4s
	fsub v1.4s, v1.4s, v4.4s
	frintn v17.4s, v6.4s
	fcvtns v4.4s, v4.4s
	fmin v16.4s, v16.4s, v19.4s
	fsub v0.4s, v0.4s, v7.4s
	fcvtns v7.4s, v7.4s
	fmla v18.4s, v25.4s, v1.4s
	fmin v17.4s, v17.4s, v19.4s
	mov v19.16b, v26.16b
	fsub v2.4s, v2.4s, v16.4s
	fcvtns v16.4s, v16.4s
	shl v7.4s, v7.4s, #23
	shl v4.4s, v4.4s, #23
	fmla v19.4s, v25.4s, v0.4s
	fmla v20.4s, v1.4s, v18.4s
	mov v18.16b, v26.16b
	fsub v6.4s, v6.4s, v17.4s
	fcvtns v17.4s, v17.4s
	shl v16.4s, v16.4s, #23
	add v7.4s, v7.4s, v3.4s
	add v4.4s, v4.4s, v3.4s
	fmla v18.4s, v25.4s, v2.4s
	fmla v21.4s, v0.4s, v19.4s
	mov v19.16b, v26.16b
	fmla v22.4s, v1.4s, v20.4s
	mov v20.16b, v27.16b
	add v16.4s, v16.4s, v3.4s
	shl v17.4s, v17.4s, #23
	fmla v19.4s, v25.4s, v6.4s
	fmla v20.4s, v2.4s, v18.4s
	mov v18.16b, v29.16b
	fmla v11.4s, v0.4s, v21.4s
	mov v21.16b, v27.16b
	add v17.4s, v17.4s, v3.4s
	fmla v18.4s, v1.4s, v22.4s
	mov v22.16b, v28.16b
	fmla v21.4s, v6.4s, v19.4s
	mov v19.16b, v29.16b
	fmla v22.4s, v2.4s, v20.4s
	mov v20.16b, v30.16b
	fmla v19.4s, v0.4s, v11.4s
	fmov v11.4s, #1.00000000
	fmla v20.4s, v1.4s, v18.4s
	mov v18.16b, v28.16b
	fmla v18.4s, v6.4s, v21.4s
	mov v21.16b, v29.16b
	fmla v11.4s, v1.4s, v20.4s
	mov v1.16b, v29.16b
	mov v20.16b, v30.16b
	fmla v21.4s, v2.4s, v22.4s
	mov v22.16b, v30.16b
	fmla v1.4s, v6.4s, v18.4s
	mov v18.16b, v30.16b
	fmul v4.4s, v11.4s, v4.4s
	fmla v22.4s, v0.4s, v19.4s
	fmov v19.4s, #1.00000000
	fmla v18.4s, v2.4s, v21.4s
	fmov v21.4s, #1.00000000
	fmla v20.4s, v6.4s, v1.4s
	fcmge v1.4s, v9.4s, v24.4s
	fcmgt v9.4s, v23.4s, v9.4s
	fmla v19.4s, v0.4s, v22.4s
	fmov v0.4s, #1.00000000
	fcmge v22.4s, v10.4s, v24.4s
	fmla v21.4s, v2.4s, v18.4s
	fcmge v2.4s, v8.4s, v24.4s
	fcmge v18.4s, v31.4s, v24.4s
	fcmgt v10.4s, v23.4s, v10.4s
	fcmgt v8.4s, v23.4s, v8.4s
	fmla v0.4s, v6.4s, v20.4s
	mvn v6.16b, v1.16b
	fcmgt v20.4s, v23.4s, v31.4s
	fmul v16.4s, v21.4s, v16.4s
	mvn v31.16b, v2.16b
	mvn v21.16b, v18.16b
	fmul v7.4s, v19.4s, v7.4s
	mvn v19.16b, v22.16b
	and v1.16b, v1.16b, v5.16b
	bic v6.16b, v6.16b, v9.16b
	and v2.16b, v2.16b, v5.16b
	fmul v0.4s, v0.4s, v17.4s
	bic v31.16b, v31.16b, v8.16b
	bic v17.16b, v21.16b, v20.16b
	bic v19.16b, v19.16b, v10.16b
	and v6.16b, v6.16b, v16.16b
	and v16.16b, v18.16b, v5.16b
	and v7.16b, v31.16b, v7.16b
	and v0.16b, v17.16b, v0.16b
	and v4.16b, v19.16b, v4.16b
	orr v1.16b, v6.16b, v1.16b
	and v6.16b, v22.16b, v5.16b
	orr v2.16b, v7.16b, v2.16b
	orr v0.16b, v0.16b, v16.16b
	orr v4.16b, v4.16b, v6.16b
	stp q1, q2, [x0]
	stp q0, q4, [x0, #32]
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
	ldr w8, [sp, #188]
	cbz w8, .LBB7_13
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB7_13:
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
	b linear_srgb::simd::linear_to_gamma_slice_tier_scalar_t
