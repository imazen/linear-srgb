.section .text.stub_srgb_to_linear_rgba_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_rgba_slice
	.p2align	2
.type	stub_srgb_to_linear_rgba_slice,@function
stub_srgb_to_linear_rgba_slice:
	.cfi_startproc
	str d12, [sp, #-48]!
	.cfi_def_cfa_offset 48
	stp d11, d10, [sp, #16]
	stp d9, d8, [sp, #32]
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_offset b10, -24
	.cfi_offset b11, -32
	.cfi_offset b12, -48
	lsl x8, x1, #2
	ands x8, x8, #0x7fffffffffffffc0
	b.eq .LBB13_3
	mov w9, #33681
	mov w10, #32277
	mov w11, #6038
	movk w9, #15774, lsl #16
	movk w10, #17060, lsl #16
	movk w11, #16207, lsl #16
	dup v2.4s, w9
	dup v3.4s, w10
	mov w9, #5734
	mov w10, #9853
	movk w9, #17033, lsl #16
	dup v6.4s, w11
	movk w10, #16718, lsl #16
	dup v4.4s, w9
	mov w9, #2423
	dup v5.4s, w10
	mov w10, #1338
	mov w11, #6038
	movk w9, #15497, lsl #16
	movk w10, #49380, lsl #16
	movk w11, #16983, lsl #16
	dup v7.4s, w9
	dup v16.4s, w10
	dup v17.4s, w11
	mov w9, #41246
	mov w10, #19964
	mov w11, #61974
	movi v0.2d, #0000000000000000
	fmov v1.4s, #1.00000000
	movk w9, #17089, lsl #16
	movk w10, #16800, lsl #16
	movk w11, #15648, lsl #16
	dup v18.4s, w9
	dup v19.4s, w10
	dup v20.4s, w11
	add x9, x0, #32
.LBB13_2:
	ldur q21, [x9, #-32]
	mov v24.16b, v4.16b
	mov v27.16b, v5.16b
	mov v28.16b, v17.16b
	mov v29.16b, v18.16b
	subs x8, x8, #64
	fmax v22.4s, v21.4s, v0.4s
	mov v30.16b, v7.16b
	mov v31.16b, v19.16b
	mov v9.16b, v7.16b
	mov v10.16b, v19.16b
	mov v12.16b, v19.16b
	fcmge v21.4s, v21.4s, v1.4s
	fmin v23.4s, v22.4s, v1.4s
	ldur q22, [x9, #-16]
	fmax v26.4s, v22.4s, v0.4s
	fcmge v22.4s, v22.4s, v1.4s
	fmla v24.4s, v3.4s, v23.4s
	fadd v25.4s, v23.4s, v16.4s
	fmla v27.4s, v23.4s, v24.4s
	fmla v28.4s, v23.4s, v25.4s
	fmin v25.4s, v26.4s, v1.4s
	mov v26.16b, v6.16b
	ldr q24, [x9]
	fmax v8.4s, v24.4s, v0.4s
	fcmge v24.4s, v24.4s, v1.4s
	fmla v26.4s, v23.4s, v27.4s
	fmla v29.4s, v23.4s, v28.4s
	mov v27.16b, v4.16b
	fadd v28.4s, v25.4s, v16.4s
	fmla v27.4s, v3.4s, v25.4s
	fmla v30.4s, v23.4s, v26.4s
	fmla v31.4s, v23.4s, v29.4s
	mov v26.16b, v5.16b
	mov v29.16b, v17.16b
	fmla v26.4s, v25.4s, v27.4s
	fmin v27.4s, v8.4s, v1.4s
	fmla v29.4s, v25.4s, v28.4s
	fdiv v28.4s, v30.4s, v31.4s
	mov v30.16b, v6.16b
	mov v31.16b, v18.16b
	fmla v30.4s, v25.4s, v26.4s
	fadd v8.4s, v27.4s, v16.4s
	ldr q26, [x9, #16]
	fmla v31.4s, v25.4s, v29.4s
	mov v29.16b, v4.16b
	fmax v11.4s, v26.4s, v0.4s
	fcmge v26.4s, v26.4s, v1.4s
	fmla v29.4s, v3.4s, v27.4s
	fmla v9.4s, v25.4s, v30.4s
	mov v30.16b, v5.16b
	fmla v10.4s, v25.4s, v31.4s
	mov v31.16b, v17.16b
	fmla v30.4s, v27.4s, v29.4s
	fmla v31.4s, v27.4s, v8.4s
	fmin v29.4s, v11.4s, v1.4s
	fdiv v8.4s, v9.4s, v10.4s
	mov v9.16b, v6.16b
	mov v10.16b, v18.16b
	mov v11.16b, v7.16b
	fmin v28.4s, v28.4s, v1.4s
	fmla v9.4s, v27.4s, v30.4s
	fmla v10.4s, v27.4s, v31.4s
	mov v30.16b, v4.16b
	fadd v31.4s, v29.4s, v16.4s
	fmla v30.4s, v3.4s, v29.4s
	fmla v11.4s, v27.4s, v9.4s
	fmla v12.4s, v27.4s, v10.4s
	mov v9.16b, v5.16b
	mov v10.16b, v17.16b
	fmla v9.4s, v29.4s, v30.4s
	fmla v10.4s, v29.4s, v31.4s
	fdiv v30.4s, v11.4s, v12.4s
	mov v31.16b, v6.16b
	mov v11.16b, v18.16b
	fmul v12.4s, v29.4s, v2.4s
	fmin v8.4s, v8.4s, v1.4s
	fmla v31.4s, v29.4s, v9.4s
	mov v9.16b, v7.16b
	fmla v11.4s, v29.4s, v10.4s
	mov v10.16b, v19.16b
	fmla v9.4s, v29.4s, v31.4s
	fmla v10.4s, v29.4s, v11.4s
	fcmgt v29.4s, v20.4s, v29.4s
	fmul v11.4s, v27.4s, v2.4s
	fcmgt v27.4s, v20.4s, v27.4s
	fdiv v31.4s, v9.4s, v10.4s
	fmul v9.4s, v23.4s, v2.4s
	fcmgt v23.4s, v20.4s, v23.4s
	fmul v10.4s, v25.4s, v2.4s
	fcmgt v25.4s, v20.4s, v25.4s
	fmin v30.4s, v30.4s, v1.4s
	bsl v23.16b, v9.16b, v28.16b
	mov v28.16b, v29.16b
	bsl v25.16b, v10.16b, v8.16b
	bsl v27.16b, v11.16b, v30.16b
	bsl v21.16b, v1.16b, v23.16b
	mov v23.16b, v24.16b
	mov v24.16b, v26.16b
	bsl v22.16b, v1.16b, v25.16b
	ldur s25, [x9, #-20]
	ldur s26, [x9, #-4]
	bsl v23.16b, v1.16b, v27.16b
	fmin v31.4s, v31.4s, v1.4s
	stp q21, q22, [x9, #-32]
	ldr s21, [x9, #12]
	ldr s22, [x9, #28]
	stur s25, [x9, #-20]
	stur s26, [x9, #-4]
	bsl v28.16b, v12.16b, v31.16b
	bsl v24.16b, v1.16b, v28.16b
	stp q23, q24, [x9]
	str s21, [x9, #12]
	str s22, [x9, #28]
	add x9, x9, #64
	b.ne .LBB13_2
.LBB13_3:
	tst x1, #0xc
	b.eq .LBB13_21
	adrp x10, .LCPI13_0
	adrp x11, .LCPI13_1
	and x8, x1, #0x1ffffffffffffff0
	ldr d0, [x10, :lo12:.LCPI13_0]
	adrp x10, .LCPI13_2
	ldr d1, [x11, :lo12:.LCPI13_1]
	ldr d2, [x10, :lo12:.LCPI13_2]
	adrp x10, .LCPI13_3
	adrp x11, .LCPI13_4
	ldr q3, [x10, :lo12:.LCPI13_3]
	adrp x10, .LCPI13_5
	ldr q4, [x11, :lo12:.LCPI13_4]
	mov w11, #61974
	ldr q5, [x10, :lo12:.LCPI13_5]
	mov w10, #33681
	movk w11, #15648, lsl #16
	movk w10, #15774, lsl #16
	and x9, x1, #0xc
	fmov s6, w11
	fmov s7, w10
	add x8, x0, x8, lsl #2
	neg x9, x9
	b .LBB13_7
.LBB13_5:
	fcvt d16, s18
	fmadd d17, d16, d1, d0
	fadd d18, d16, d2
	mov v17.d[1], v18.d[0]
	mov v18.16b, v3.16b
	fmla v18.2d, v17.2d, v16.d[0]
	mov v17.16b, v4.16b
	fmla v17.2d, v18.2d, v16.d[0]
	mov v18.16b, v5.16b
	fmla v18.2d, v17.2d, v16.d[0]
	dup v16.2d, v18.d[1]
	fdiv v16.2d, v18.2d, v16.2d
	fcvt s17, d16
.LBB13_6:
	adds x9, x9, #4
	str s17, [x8, #8]
	add x8, x8, #16
	b.eq .LBB13_21
.LBB13_7:
	ldr s17, [x8]
	movi d16, #0000000000000000
	movi d18, #0000000000000000
	fcmp s17, #0.0
	b.mi .LBB13_11
	fmov s18, #1.00000000
	fcmp s17, s18
	b.ge .LBB13_11
	fcmp s17, s6
	b.ls .LBB13_15
	fcvt d17, s17
	fmadd d18, d17, d1, d0
	fadd d19, d17, d2
	mov v18.d[1], v19.d[0]
	mov v19.16b, v3.16b
	fmla v19.2d, v18.2d, v17.d[0]
	mov v18.16b, v4.16b
	fmla v18.2d, v19.2d, v17.d[0]
	mov v19.16b, v5.16b
	fmla v19.2d, v18.2d, v17.d[0]
	dup v17.2d, v19.d[1]
	fdiv v17.2d, v19.2d, v17.2d
	fcvt s18, d17
.LBB13_11:
	ldr s17, [x8, #4]
	str s18, [x8]
	fcmp s17, #0.0
	b.mi .LBB13_17
.LBB13_12:
	fmov s16, #1.00000000
	fcmp s17, s16
	b.ge .LBB13_17
	fcmp s17, s6
	b.ls .LBB13_16
	fcvt d16, s17
	fmadd d17, d16, d1, d0
	fadd d18, d16, d2
	mov v17.d[1], v18.d[0]
	mov v18.16b, v3.16b
	fmla v18.2d, v17.2d, v16.d[0]
	mov v17.16b, v4.16b
	fmla v17.2d, v18.2d, v16.d[0]
	mov v18.16b, v5.16b
	fmla v18.2d, v17.2d, v16.d[0]
	dup v16.2d, v18.d[1]
	fdiv v16.2d, v18.2d, v16.2d
	fcvt s16, d16
	b .LBB13_17
.LBB13_15:
	fmul s18, s17, s7
	ldr s17, [x8, #4]
	str s18, [x8]
	fcmp s17, #0.0
	b.pl .LBB13_12
	b .LBB13_17
.LBB13_16:
	fmul s16, s17, s7
.LBB13_17:
	ldr s18, [x8, #8]
	movi d17, #0000000000000000
	str s16, [x8, #4]
	fcmp s18, #0.0
	b.mi .LBB13_6
	fmov s17, #1.00000000
	fcmp s18, s17
	b.ge .LBB13_6
	fcmp s18, s6
	b.hi .LBB13_5
	fmul s17, s18, s7
	b .LBB13_6
.LBB13_21:
	ldp d9, d8, [sp, #32]
	ldp d11, d10, [sp, #16]
	ldr d12, [sp], #48
	.cfi_def_cfa_offset 0
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	ret
