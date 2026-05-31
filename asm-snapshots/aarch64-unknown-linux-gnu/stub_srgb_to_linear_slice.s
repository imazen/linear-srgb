.section .text.stub_srgb_to_linear_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_slice
	.p2align	2
.type	stub_srgb_to_linear_slice,@function
stub_srgb_to_linear_slice:
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
	ands x9, x8, #0x7fffffffffffffc0
	b.eq .LBB14_3
	mov w10, #33681
	mov w11, #32277
	mov w12, #6038
	movk w10, #15774, lsl #16
	movk w11, #17060, lsl #16
	movk w12, #16207, lsl #16
	dup v2.4s, w10
	dup v3.4s, w11
	mov w10, #5734
	mov w11, #9853
	movk w10, #17033, lsl #16
	dup v6.4s, w12
	movk w11, #16718, lsl #16
	dup v4.4s, w10
	mov w10, #2423
	dup v5.4s, w11
	mov w11, #1338
	mov w12, #6038
	movk w10, #15497, lsl #16
	movk w11, #49380, lsl #16
	movk w12, #16983, lsl #16
	dup v7.4s, w10
	dup v16.4s, w11
	dup v17.4s, w12
	mov w10, #41246
	mov w11, #19964
	mov w12, #61974
	movi v0.2d, #0000000000000000
	fmov v1.4s, #1.00000000
	movk w10, #17089, lsl #16
	movk w11, #16800, lsl #16
	movk w12, #15648, lsl #16
	dup v18.4s, w10
	dup v19.4s, w11
	dup v20.4s, w12
	add x9, x0, x9
	mov x10, x0
.LBB14_2:
	ldp q22, q21, [x10]
	mov v24.16b, v4.16b
	mov v27.16b, v5.16b
	mov v28.16b, v17.16b
	mov v29.16b, v6.16b
	mov v30.16b, v18.16b
	mov v31.16b, v7.16b
	mov v8.16b, v19.16b
	fmax v23.4s, v22.4s, v0.4s
	fmax v26.4s, v21.4s, v0.4s
	mov v10.16b, v19.16b
	mov v12.16b, v19.16b
	fcmge v22.4s, v22.4s, v1.4s
	fcmge v21.4s, v21.4s, v1.4s
	fmin v23.4s, v23.4s, v1.4s
	fmla v24.4s, v3.4s, v23.4s
	fadd v25.4s, v23.4s, v16.4s
	fmla v27.4s, v23.4s, v24.4s
	fmla v28.4s, v23.4s, v25.4s
	fmin v25.4s, v26.4s, v1.4s
	ldp q26, q24, [x10, #32]
	fmla v29.4s, v23.4s, v27.4s
	fmla v30.4s, v23.4s, v28.4s
	mov v27.16b, v4.16b
	fadd v28.4s, v25.4s, v16.4s
	fmax v9.4s, v26.4s, v0.4s
	fmax v11.4s, v24.4s, v0.4s
	fcmge v26.4s, v26.4s, v1.4s
	fcmge v24.4s, v24.4s, v1.4s
	fmla v27.4s, v3.4s, v25.4s
	fmla v31.4s, v23.4s, v29.4s
	fmla v8.4s, v23.4s, v30.4s
	mov v29.16b, v5.16b
	mov v30.16b, v17.16b
	fmla v29.4s, v25.4s, v27.4s
	fmin v27.4s, v9.4s, v1.4s
	mov v9.16b, v7.16b
	fmla v30.4s, v25.4s, v28.4s
	fdiv v28.4s, v31.4s, v8.4s
	mov v31.16b, v6.16b
	mov v8.16b, v18.16b
	fmla v31.4s, v25.4s, v29.4s
	mov v29.16b, v4.16b
	fmla v8.4s, v25.4s, v30.4s
	fadd v30.4s, v27.4s, v16.4s
	fmla v29.4s, v3.4s, v27.4s
	fmla v9.4s, v25.4s, v31.4s
	mov v31.16b, v5.16b
	fmla v10.4s, v25.4s, v8.4s
	mov v8.16b, v17.16b
	fmla v31.4s, v27.4s, v29.4s
	fmin v29.4s, v11.4s, v1.4s
	mov v11.16b, v7.16b
	fmla v8.4s, v27.4s, v30.4s
	fdiv v30.4s, v9.4s, v10.4s
	mov v9.16b, v6.16b
	mov v10.16b, v18.16b
	fmin v28.4s, v28.4s, v1.4s
	fmla v9.4s, v27.4s, v31.4s
	fmla v10.4s, v27.4s, v8.4s
	mov v31.16b, v4.16b
	fadd v8.4s, v29.4s, v16.4s
	fmla v31.4s, v3.4s, v29.4s
	fmla v11.4s, v27.4s, v9.4s
	fmla v12.4s, v27.4s, v10.4s
	mov v9.16b, v5.16b
	mov v10.16b, v17.16b
	fmla v9.4s, v29.4s, v31.4s
	fmla v10.4s, v29.4s, v8.4s
	fdiv v31.4s, v11.4s, v12.4s
	mov v8.16b, v6.16b
	mov v11.16b, v18.16b
	fmul v12.4s, v29.4s, v2.4s
	fmin v30.4s, v30.4s, v1.4s
	fmla v8.4s, v29.4s, v9.4s
	mov v9.16b, v7.16b
	fmla v11.4s, v29.4s, v10.4s
	mov v10.16b, v19.16b
	fmla v9.4s, v29.4s, v8.4s
	fmla v10.4s, v29.4s, v11.4s
	fcmgt v29.4s, v20.4s, v29.4s
	fmul v11.4s, v27.4s, v2.4s
	fcmgt v27.4s, v20.4s, v27.4s
	fdiv v8.4s, v9.4s, v10.4s
	fmul v9.4s, v23.4s, v2.4s
	fcmgt v23.4s, v20.4s, v23.4s
	fmul v10.4s, v25.4s, v2.4s
	fcmgt v25.4s, v20.4s, v25.4s
	fmin v31.4s, v31.4s, v1.4s
	bsl v23.16b, v9.16b, v28.16b
	mov v28.16b, v29.16b
	bsl v25.16b, v10.16b, v30.16b
	bsl v27.16b, v11.16b, v31.16b
	bsl v22.16b, v1.16b, v23.16b
	mov v23.16b, v26.16b
	bsl v21.16b, v1.16b, v25.16b
	bsl v23.16b, v1.16b, v27.16b
	fmin v8.4s, v8.4s, v1.4s
	stp q22, q21, [x10]
	bsl v28.16b, v12.16b, v8.16b
	bsl v24.16b, v1.16b, v28.16b
	stp q23, q24, [x10, #32]
	add x10, x10, #64
	cmp x10, x9
	b.ne .LBB14_2
.LBB14_3:
	ands x8, x8, #0x3c
	b.eq .LBB14_11
	adrp x10, .LCPI14_0
	adrp x11, .LCPI14_1
	and x9, x1, #0x1ffffffffffffff0
	ldr d0, [x10, :lo12:.LCPI14_0]
	adrp x10, .LCPI14_2
	ldr d1, [x11, :lo12:.LCPI14_1]
	ldr d2, [x10, :lo12:.LCPI14_2]
	adrp x10, .LCPI14_3
	adrp x11, .LCPI14_4
	ldr q3, [x10, :lo12:.LCPI14_3]
	adrp x10, .LCPI14_5
	ldr q4, [x11, :lo12:.LCPI14_4]
	mov w11, #61974
	ldr q5, [x10, :lo12:.LCPI14_5]
	mov w10, #33681
	movk w11, #15648, lsl #16
	movk w10, #15774, lsl #16
	add x9, x0, x9, lsl #2
	fmov s6, w11
	fmov s7, w10
	mov x10, x9
	b .LBB14_7
.LBB14_5:
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
.LBB14_6:
	subs x8, x8, #4
	str s16, [x9]
	mov x9, x10
	b.eq .LBB14_11
.LBB14_7:
	ldr s17, [x10], #4
	movi d16, #0000000000000000
	fcmp s17, #0.0
	b.mi .LBB14_6
	fmov s16, #1.00000000
	fcmp s17, s16
	b.ge .LBB14_6
	fcmp s17, s6
	b.hi .LBB14_5
	fmul s16, s17, s7
	b .LBB14_6
.LBB14_11:
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
