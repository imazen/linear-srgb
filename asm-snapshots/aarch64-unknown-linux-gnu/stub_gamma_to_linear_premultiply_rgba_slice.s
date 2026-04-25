.section .text.stub_gamma_to_linear_premultiply_rgba_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_premultiply_rgba_slice
	.p2align	2
.type	stub_gamma_to_linear_premultiply_rgba_slice,@function
stub_gamma_to_linear_premultiply_rgba_slice:
	.cfi_startproc
	lsl x8, x1, #2
	ands x8, x8, #0x7ffffffffffffff0
	b.eq .LBB5_4
	stp d13, d12, [sp, #-48]!
	.cfi_def_cfa_offset 48
	stp d11, d10, [sp, #16]
	stp d9, d8, [sp, #32]
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_offset b10, -24
	.cfi_offset b11, -32
	.cfi_offset b12, -40
	.cfi_offset b13, -48
	mov w9, #64269
	mov w10, #1267
	mvni v24.4s, #63, msl #16
	movk w9, #74, lsl #16
	movk w10, #16181, lsl #16
	movi v7.2d, #0000000000000000
	dup v1.4s, w9
	mov w9, #27455
	dup v2.4s, w10
	movk w9, #16094, lsl #16
	mov w10, #43579
	fmov v17.4s, #1.00000000
	dup v3.4s, w9
	mov w9, #46233
	movk w10, #16440, lsl #16
	movk w9, #16147, lsl #16
	movi v20.4s, #127, msl #16
	fmov v22.4s, #-1.00000000
	dup v4.4s, w9
	mov w9, #14483
	fneg v24.4s, v24.4s
	movk w9, #16246, lsl #16
	movi v26.4s, #67, lsl #24
	dup v6.4s, w10
	dup v5.4s, w9
	mov w9, #-1023672320
	mvni v28.4s, #127, msl #16
	dup v16.4s, w9
	mov w9, #1123942400
	dup v18.4s, w9
	mov w9, #63802
	movk w9, #14625, lsl #16
	dup v19.4s, w9
	mov w9, #50297
	movk w9, #15022, lsl #16
	dup v21.4s, w9
	mov w9, #38156
	movk w9, #15389, lsl #16
	dup v23.4s, w9
	mov w9, #22716
	movk w9, #15715, lsl #16
	dup v25.4s, w9
	mov w9, #65005
	movk w9, #15989, lsl #16
	dup v27.4s, w9
	mov w9, #29208
	movk w9, #16177, lsl #16
	dup v29.4s, w9
	adrp x9, .LCPI5_0
	ldr q30, [x9, :lo12:.LCPI5_0]
.LBB5_2:
	ldr q31, [x0]
	mov v11.16b, v4.16b
	mov v13.16b, v5.16b
	mvni v12.4s, #126
	subs x8, x8, #16
	fmax v31.4s, v31.4s, v7.4s
	fmin v31.4s, v31.4s, v17.4s
	add v8.4s, v31.4s, v1.4s
	and v9.16b, v8.16b, v20.16b
	ssra v12.4s, v8.4s, #23
	mov v8.16b, v6.16b
	add v9.4s, v9.4s, v2.4s
	fadd v10.4s, v9.4s, v22.4s
	fadd v9.4s, v9.4s, v17.4s
	fdiv v9.4s, v10.4s, v9.4s
	fmul v10.4s, v9.4s, v9.4s
	fmla v11.4s, v3.4s, v10.4s
	fmla v13.4s, v10.4s, v11.4s
	scvtf v11.4s, v12.4s
	fmov v12.4s, #1.00000000
	fmla v8.4s, v10.4s, v13.4s
	fcmeq v10.4s, v31.4s, #0.0
	fcmlt v31.4s, v31.4s, #0.0
	fmla v11.4s, v9.4s, v8.4s
	mov v8.16b, v10.16b
	mov v10.16b, v21.16b
	bsl v8.16b, v28.16b, v11.16b
	mov v11.16b, v23.16b
	bsl v31.16b, v24.16b, v8.16b
	fmul v31.4s, v31.4s, v0.s[0]
	fmax v8.4s, v31.4s, v16.4s
	fmin v8.4s, v8.4s, v26.4s
	frintn v9.4s, v8.4s
	fmin v9.4s, v9.4s, v18.4s
	fsub v8.4s, v8.4s, v9.4s
	fcvtns v9.4s, v9.4s
	fmla v10.4s, v19.4s, v8.4s
	shl v9.4s, v9.4s, #23
	fmla v11.4s, v8.4s, v10.4s
	mov v10.16b, v25.16b
	fmla v10.4s, v8.4s, v11.4s
	mov v11.16b, v27.16b
	fmla v11.4s, v8.4s, v10.4s
	mov v10.16b, v29.16b
	fmla v10.4s, v8.4s, v11.4s
	fcmge v11.4s, v31.4s, v26.4s
	fcmgt v31.4s, v16.4s, v31.4s
	fmla v12.4s, v8.4s, v10.4s
	add v8.4s, v9.4s, v17.4s
	mvn v9.16b, v11.16b
	bic v31.16b, v9.16b, v31.16b
	and v9.16b, v11.16b, v30.16b
	fmul v8.4s, v12.4s, v8.4s
	and v31.16b, v31.16b, v8.16b
	ldr s8, [x0, #12]
	orr v31.16b, v31.16b, v9.16b
	fmul v9.2s, v31.2s, v8.s[0]
	fmul s31, s8, v31.s[2]
	str d9, [x0]
	str s31, [x0, #8]
	add x0, x0, #16
	b.ne .LBB5_2
	ldp d9, d8, [sp, #32]
	ldp d11, d10, [sp, #16]
	ldp d13, d12, [sp], #48
	.cfi_def_cfa_offset 0
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	.cfi_restore b13
.LBB5_4:
	ret
