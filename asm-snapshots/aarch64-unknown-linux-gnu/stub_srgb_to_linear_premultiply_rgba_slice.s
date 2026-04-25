.section .text.stub_srgb_to_linear_premultiply_rgba_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_premultiply_rgba_slice
	.p2align	2
.type	stub_srgb_to_linear_premultiply_rgba_slice,@function
stub_srgb_to_linear_premultiply_rgba_slice:
	.cfi_startproc
	lsl x8, x1, #2
	ands x8, x8, #0x7ffffffffffffff0
	b.eq .LBB12_3
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
.LBB12_2:
	ldr q21, [x0]
	mov v23.16b, v4.16b
	mov v25.16b, v5.16b
	mov v26.16b, v17.16b
	subs x8, x8, #16
	fmax v22.4s, v21.4s, v0.4s
	fcmge v21.4s, v21.4s, v1.4s
	fmin v22.4s, v22.4s, v1.4s
	fmla v23.4s, v3.4s, v22.4s
	fadd v24.4s, v22.4s, v16.4s
	fmla v25.4s, v22.4s, v23.4s
	fmla v26.4s, v22.4s, v24.4s
	mov v23.16b, v6.16b
	mov v24.16b, v18.16b
	fmla v23.4s, v22.4s, v25.4s
	fmla v24.4s, v22.4s, v26.4s
	mov v25.16b, v7.16b
	mov v26.16b, v19.16b
	fmla v25.4s, v22.4s, v23.4s
	fmla v26.4s, v22.4s, v24.4s
	fmul v24.4s, v22.4s, v2.4s
	fcmgt v22.4s, v20.4s, v22.4s
	fdiv v23.4s, v25.4s, v26.4s
	fmin v23.4s, v23.4s, v1.4s
	bsl v22.16b, v24.16b, v23.16b
	bsl v21.16b, v1.16b, v22.16b
	ldr s22, [x0, #12]
	fmul v23.2s, v21.2s, v22.s[0]
	fmul s21, s22, v21.s[2]
	str d23, [x0]
	str s21, [x0, #8]
	add x0, x0, #16
	b.ne .LBB12_2
.LBB12_3:
	ret
