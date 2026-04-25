.section .text.stub_linear_to_srgb_rgba_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_rgba_slice
	.p2align	2
.type	stub_linear_to_srgb_rgba_slice,@function
stub_linear_to_srgb_rgba_slice:
	.cfi_startproc
	lsl x8, x1, #2
	ands x8, x8, #0x7ffffffffffffff0
	b.eq .LBB9_3
	mov w9, #47186
	mov w10, #25800
	mov w11, #14394
	movk w9, #16718, lsl #16
	movk w10, #16863, lsl #16
	movk w11, #15807, lsl #16
	dup v2.4s, w9
	dup v3.4s, w10
	mov w9, #5570
	mov w10, #10701
	movk w9, #16968, lsl #16
	dup v6.4s, w11
	movk w10, #16697, lsl #16
	dup v4.4s, w9
	mov w9, #15682
	dup v5.4s, w10
	mov w10, #15285
	mov w11, #7182
	movk w9, #48222, lsl #16
	movk w10, #16906, lsl #16
	movk w11, #16947, lsl #16
	dup v7.4s, w9
	dup v16.4s, w10
	dup v17.4s, w11
	mov w9, #64401
	mov w10, #55785
	mov w11, #20545
	movi v0.2d, #0000000000000000
	fmov v1.4s, #1.00000000
	movk w9, #16655, lsl #16
	movk w10, #16006, lsl #16
	movk w11, #15175, lsl #16
	dup v18.4s, w9
	dup v19.4s, w10
	dup v20.4s, w11
.LBB9_2:
	ldr q21, [x0]
	mov v24.16b, v4.16b
	mov v26.16b, v5.16b
	mov v27.16b, v17.16b
	subs x8, x8, #16
	fmax v22.4s, v21.4s, v0.4s
	fcmge v21.4s, v21.4s, v1.4s
	fmin v22.4s, v22.4s, v1.4s
	fsqrt v23.4s, v22.4s
	fmla v24.4s, v3.4s, v23.4s
	fadd v25.4s, v23.4s, v16.4s
	fmla v26.4s, v23.4s, v24.4s
	fmla v27.4s, v23.4s, v25.4s
	mov v24.16b, v6.16b
	mov v25.16b, v18.16b
	fmla v24.4s, v23.4s, v26.4s
	fmla v25.4s, v23.4s, v27.4s
	mov v26.16b, v7.16b
	mov v27.16b, v19.16b
	fmla v26.4s, v23.4s, v24.4s
	fmla v27.4s, v23.4s, v25.4s
	fmul v24.4s, v22.4s, v2.4s
	fcmgt v22.4s, v20.4s, v22.4s
	fdiv v23.4s, v26.4s, v27.4s
	fmin v23.4s, v23.4s, v1.4s
	bsl v22.16b, v24.16b, v23.16b
	bsl v21.16b, v1.16b, v22.16b
	ldr s22, [x0, #12]
	str q21, [x0]
	str s22, [x0, #12]
	add x0, x0, #16
	b.ne .LBB9_2
.LBB9_3:
	ret
