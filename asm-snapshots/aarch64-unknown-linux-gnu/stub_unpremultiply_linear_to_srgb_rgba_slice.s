.section .text.stub_unpremultiply_linear_to_srgb_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_srgb_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_srgb_rgba_slice,@function
stub_unpremultiply_linear_to_srgb_rgba_slice:
	.cfi_startproc
	lsl x8, x1, #2
	ands x8, x8, #0x7ffffffffffffff0
	b.eq .LBB16_5
	mov w9, #47186
	mov w10, #25800
	movi v0.2d, #0000000000000000
	movk w9, #16718, lsl #16
	movk w10, #16863, lsl #16
	fmov v1.4s, #1.00000000
	dup v2.4s, w9
	mov w9, #5570
	dup v3.4s, w10
	movk w9, #16968, lsl #16
	mov w10, #14394
	fmov s20, #1.00000000
	dup v4.4s, w9
	mov w9, #10701
	movk w10, #15807, lsl #16
	movk w9, #16697, lsl #16
	dup v6.4s, w10
	mov w10, #7182
	dup v5.4s, w9
	mov w9, #15682
	movk w10, #16947, lsl #16
	movk w9, #48222, lsl #16
	dup v17.4s, w10
	mov w10, #20545
	dup v7.4s, w9
	mov w9, #15285
	movk w10, #15175, lsl #16
	movk w9, #16906, lsl #16
	dup v21.4s, w10
	dup v16.4s, w9
	mov w9, #64401
	movk w9, #16655, lsl #16
	dup v18.4s, w9
	mov w9, #55785
	movk w9, #16006, lsl #16
	dup v19.4s, w9
	mov w9, #981467136
	fmov s22, w9
	b .LBB16_3
.LBB16_2:
	fdiv s24, s20, s23
	ldr s25, [x0, #8]
	ldr d26, [x0]
	mov v27.16b, v4.16b
	mov v29.16b, v5.16b
	mov v30.16b, v17.16b
	fmul s25, s24, s25
	fmul v24.4s, v26.4s, v24.s[0]
	mov v24.s[2], v25.s[0]
	mov v24.s[3], v20.s[0]
	fmax v25.4s, v24.4s, v0.4s
	fcmge v24.4s, v24.4s, v1.4s
	fmin v25.4s, v25.4s, v1.4s
	fsqrt v26.4s, v25.4s
	fmla v27.4s, v3.4s, v26.4s
	fadd v28.4s, v26.4s, v16.4s
	fmla v29.4s, v26.4s, v27.4s
	fmla v30.4s, v26.4s, v28.4s
	mov v27.16b, v6.16b
	mov v28.16b, v18.16b
	fmla v27.4s, v26.4s, v29.4s
	fmla v28.4s, v26.4s, v30.4s
	mov v29.16b, v7.16b
	mov v30.16b, v19.16b
	fmla v29.4s, v26.4s, v27.4s
	fmla v30.4s, v26.4s, v28.4s
	fmul v27.4s, v25.4s, v2.4s
	fcmgt v25.4s, v21.4s, v25.4s
	fdiv v26.4s, v29.4s, v30.4s
	fmin v26.4s, v26.4s, v1.4s
	bsl v25.16b, v27.16b, v26.16b
	bsl v24.16b, v1.16b, v25.16b
	str q24, [x0]
	str s23, [x0, #12]
	subs x8, x8, #16
	add x0, x0, #16
	b.eq .LBB16_5
.LBB16_3:
	ldr s23, [x0, #12]
	fcmp s23, s22
	b.gt .LBB16_2
	str xzr, [x0]
	str wzr, [x0, #8]
	subs x8, x8, #16
	add x0, x0, #16
	b.ne .LBB16_3
.LBB16_5:
	ret
