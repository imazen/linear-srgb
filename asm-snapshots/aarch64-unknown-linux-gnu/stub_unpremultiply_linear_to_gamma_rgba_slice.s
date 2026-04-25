.section .text.stub_unpremultiply_linear_to_gamma_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_gamma_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_gamma_rgba_slice,@function
stub_unpremultiply_linear_to_gamma_rgba_slice:
	.cfi_startproc
	lsl x8, x1, #2
	ands x8, x8, #0x7ffffffffffffff0
	b.eq .LBB15_6
	stp d15, d14, [sp, #-64]!
	.cfi_def_cfa_offset 64
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_offset b10, -24
	.cfi_offset b11, -32
	.cfi_offset b12, -40
	.cfi_offset b13, -48
	.cfi_offset b14, -56
	.cfi_offset b15, -64
	mov w9, #64269
	fmov s1, #1.00000000
	mov w10, #27455
	movk w9, #74, lsl #16
	movk w10, #16094, lsl #16
	mvni v21.4s, #63, msl #16
	dup v4.4s, w9
	mov w9, #1267
	dup v16.4s, w10
	movk w9, #16181, lsl #16
	fdiv s0, s1, s0
	mov w10, #14483
	dup v6.4s, w9
	mov w9, #46233
	movk w10, #16246, lsl #16
	movk w9, #16147, lsl #16
	dup v2.4s, w10
	mvni v20.4s, #127, msl #16
	dup v17.4s, w9
	mov w9, #43579
	mov w10, #38156
	movk w9, #16440, lsl #16
	movk w10, #15389, lsl #16
	mov w11, #29208
	dup v19.4s, w9
	mov w9, #-1023672320
	dup v27.4s, w10
	dup v22.4s, w9
	mov w9, #1123942400
	mov w10, #65005
	dup v24.4s, w9
	mov w9, #63802
	fmov v3.4s, #1.00000000
	movk w9, #14625, lsl #16
	movi v5.4s, #127, msl #16
	fmov v7.4s, #-1.00000000
	dup v25.4s, w9
	mov w9, #50297
	fneg v21.4s, v21.4s
	movk w9, #15022, lsl #16
	movi v23.4s, #67, lsl #24
	movk w10, #15989, lsl #16
	dup v26.4s, w9
	mov w9, #22716
	movk w11, #16177, lsl #16
	movk w9, #15715, lsl #16
	fneg v28.4s, v20.4s
	dup v30.4s, w10
	dup v29.4s, w9
	mov w9, #981467136
	dup v31.4s, w11
	fmov s8, w9
	b .LBB15_3
.LBB15_2:
	fdiv s10, s1, s9
	ldr s11, [x0, #8]
	ldr d12, [x0]
	movi v18.2d, #0000000000000000
	mov v14.16b, v17.16b
	mvni v15.4s, #126
	fmul s11, s10, s11
	fmul v10.4s, v12.4s, v10.s[0]
	mov v10.s[2], v11.s[0]
	mov v10.s[3], v1.s[0]
	fmax v10.4s, v10.4s, v18.4s
	mov v18.16b, v2.16b
	fmin v10.4s, v10.4s, v3.4s
	add v11.4s, v10.4s, v4.4s
	and v12.16b, v11.16b, v5.16b
	ssra v15.4s, v11.4s, #23
	mov v11.16b, v19.16b
	add v12.4s, v12.4s, v6.4s
	fadd v13.4s, v12.4s, v7.4s
	fadd v12.4s, v12.4s, v3.4s
	fdiv v12.4s, v13.4s, v12.4s
	fmul v13.4s, v12.4s, v12.4s
	fmla v14.4s, v16.4s, v13.4s
	fmla v18.4s, v13.4s, v14.4s
	scvtf v14.4s, v15.4s
	fmla v11.4s, v13.4s, v18.4s
	fcmeq v18.4s, v10.4s, #0.0
	fcmlt v10.4s, v10.4s, #0.0
	mov v13.16b, v27.16b
	fmla v14.4s, v12.4s, v11.4s
	mov v12.16b, v26.16b
	bsl v18.16b, v20.16b, v14.16b
	fmov v14.4s, #1.00000000
	bit v18.16b, v21.16b, v10.16b
	fmul v18.4s, v18.4s, v0.s[0]
	fmax v10.4s, v18.4s, v22.4s
	fmin v10.4s, v10.4s, v23.4s
	frintn v11.4s, v10.4s
	fmin v11.4s, v11.4s, v24.4s
	fsub v10.4s, v10.4s, v11.4s
	fcvtns v11.4s, v11.4s
	fmla v12.4s, v25.4s, v10.4s
	shl v11.4s, v11.4s, #23
	fmla v13.4s, v10.4s, v12.4s
	mov v12.16b, v29.16b
	fmla v12.4s, v10.4s, v13.4s
	mov v13.16b, v30.16b
	fmla v13.4s, v10.4s, v12.4s
	mov v12.16b, v31.16b
	fmla v12.4s, v10.4s, v13.4s
	fcmge v13.4s, v18.4s, v23.4s
	fcmgt v18.4s, v22.4s, v18.4s
	fmla v14.4s, v10.4s, v12.4s
	add v10.4s, v11.4s, v3.4s
	mvn v11.16b, v13.16b
	bic v18.16b, v11.16b, v18.16b
	and v11.16b, v13.16b, v28.16b
	fmul v10.4s, v14.4s, v10.4s
	and v18.16b, v18.16b, v10.16b
	orr v18.16b, v18.16b, v11.16b
	str q18, [x0]
	str s9, [x0, #12]
	subs x8, x8, #16
	add x0, x0, #16
	b.eq .LBB15_5
.LBB15_3:
	ldr s9, [x0, #12]
	fcmp s9, s8
	b.gt .LBB15_2
	str xzr, [x0]
	str wzr, [x0, #8]
	subs x8, x8, #16
	add x0, x0, #16
	b.ne .LBB15_3
.LBB15_5:
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #64
	.cfi_def_cfa_offset 0
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	.cfi_restore b13
	.cfi_restore b14
	.cfi_restore b15
.LBB15_6:
	ret
