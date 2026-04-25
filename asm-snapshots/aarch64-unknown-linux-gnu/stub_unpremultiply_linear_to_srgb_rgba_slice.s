.section .text.stub_unpremultiply_linear_to_srgb_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_srgb_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_srgb_rgba_slice,@function
stub_unpremultiply_linear_to_srgb_rgba_slice:
	.cfi_startproc
	sub sp, sp, #288
	.cfi_def_cfa_offset 288
	stp d15, d14, [sp, #176]
	stp d13, d12, [sp, #192]
	stp d11, d10, [sp, #208]
	stp d9, d8, [sp, #224]
	stp x29, x30, [sp, #240]
	str x28, [sp, #256]
	stp x20, x19, [sp, #272]
	add x29, sp, #240
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w28, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	.cfi_offset b8, -56
	.cfi_offset b9, -64
	.cfi_offset b10, -72
	.cfi_offset b11, -80
	.cfi_offset b12, -88
	.cfi_offset b13, -96
	.cfi_offset b14, -104
	.cfi_offset b15, -112
	lsl x8, x1, #2
	ands x8, x8, #0x7fffffffffffffc0
	b.eq .LBB16_3
	mov w9, #47186
	mov w10, #25800
	fmov v1.4s, #1.00000000
	movk w9, #16718, lsl #16
	movk w10, #16863, lsl #16
	dup v2.4s, w9
	dup v0.4s, w10
	mov w9, #5570
	movk w9, #16968, lsl #16
	mov w10, #14394
	movk w10, #15807, lsl #16
	stp q0, q2, [x29, #-96]
	dup v0.4s, w9
	mov w9, #10701
	movk w9, #16697, lsl #16
	ldur q6, [x29, #-96]
	dup v2.4s, w9
	mov w9, #15682
	stur q0, [x29, #-112]
	dup v0.4s, w10
	movk w9, #48222, lsl #16
	mov w10, #7182
	movk w10, #16947, lsl #16
	stp q0, q2, [sp, #96]
	dup v2.4s, w9
	mov w9, #15285
	movk w9, #16906, lsl #16
	dup v0.4s, w9
	mov w9, #64401
	movk w9, #16655, lsl #16
	stp q0, q2, [sp, #64]
	dup v2.4s, w10
	dup v0.4s, w9
	mov w9, #55785
	mov w10, #20545
	movk w9, #16006, lsl #16
	movk w10, #15175, lsl #16
	stp q0, q2, [sp, #32]
	dup v2.4s, w9
	dup v0.4s, w10
	mov w10, #981467136
	add x9, x0, #32
	str w10, [x29, #28]
	stp q0, q2, [sp]
.LBB16_2:
	fmov s0, #1.00000000
	ldur s24, [x9, #-20]
	ldr s5, [x29, #28]
	movi d3, #0000000000000000
	fmov v27.4s, #1.00000000
	ldur q28, [x9, #-32]
	fcmp s24, s5
	movi v16.2d, #0000000000000000
	fmov v8.4s, #1.00000000
	fdiv s25, s0, s24
	ldp q22, q2, [sp, #48]
	ldur q7, [x29, #-112]
	ldp q17, q4, [sp, #96]
	ldr q20, [sp, #80]
	mov v11.16b, v7.16b
	mov v15.16b, v22.16b
	mov v18.16b, v22.16b
	mov v14.16b, v4.16b
	fcsel s26, s25, s3, gt
	ldur s25, [x9, #-4]
	fdiv s29, s0, s25
	fcmp s25, s5
	mov v27.s[0], v26.s[0]
	mov v27.s[1], v26.s[0]
	mov v27.s[2], v26.s[0]
	ldr s26, [x9, #12]
	fdiv s31, s0, s26
	fmul v28.4s, v28.4s, v27.4s
	fcsel s29, s29, s3, gt
	fcmp s26, s5
	mov v8.s[0], v29.s[0]
	fmax v27.4s, v28.4s, v16.4s
	mov v8.s[1], v29.s[0]
	fmin v30.4s, v27.4s, v1.4s
	ldr s27, [x9, #28]
	fdiv s10, s0, s27
	fmov v0.4s, #1.00000000
	mov v8.s[2], v29.s[0]
	ldur q29, [x9, #-16]
	fcsel s13, s31, s3, gt
	fcmp s27, s5
	ldp q23, q5, [sp, #16]
	fmul v29.4s, v29.4s, v8.4s
	fmov v8.4s, #1.00000000
	mov v19.16b, v5.16b
	mov v21.16b, v23.16b
	mov v8.s[0], v13.s[0]
	fmax v31.4s, v29.4s, v16.4s
	fsqrt v9.4s, v30.4s
	mov v8.s[1], v13.s[0]
	fcsel s10, s10, s3, gt
	fmin v31.4s, v31.4s, v1.4s
	subs x8, x8, #64
	mov v0.s[0], v10.s[0]
	mov v8.s[2], v13.s[0]
	mov v13.16b, v20.16b
	mov v0.s[1], v10.s[0]
	fadd v12.4s, v9.4s, v2.4s
	fmla v11.4s, v6.4s, v9.4s
	mov v0.s[2], v10.s[0]
	ldr q10, [x9, #16]
	fmla v15.4s, v9.4s, v12.4s
	fmla v14.4s, v9.4s, v11.4s
	mov v11.16b, v17.16b
	fsqrt v12.4s, v31.4s
	fmul v10.4s, v10.4s, v0.4s
	mov v0.16b, v7.16b
	fmla v19.4s, v9.4s, v15.4s
	ldr q15, [x9]
	fmla v11.4s, v9.4s, v14.4s
	mov v14.16b, v23.16b
	fmul v8.4s, v15.4s, v8.4s
	mov v15.16b, v4.16b
	fmla v14.4s, v9.4s, v19.4s
	fmla v13.4s, v9.4s, v11.4s
	fmax v19.4s, v8.4s, v16.4s
	fdiv v9.4s, v13.4s, v14.4s
	fmla v0.4s, v6.4s, v12.4s
	fadd v14.4s, v12.4s, v2.4s
	fmin v11.4s, v19.4s, v1.4s
	fmax v19.4s, v10.4s, v16.4s
	mov v16.16b, v20.16b
	fmla v15.4s, v12.4s, v0.4s
	fmla v18.4s, v12.4s, v14.4s
	mov v0.16b, v17.16b
	fmin v19.4s, v19.4s, v1.4s
	mov v14.16b, v5.16b
	fmla v0.4s, v12.4s, v15.4s
	fmla v14.4s, v12.4s, v18.4s
	mov v18.16b, v7.16b
	fsqrt v13.4s, v11.4s
	fmla v16.4s, v12.4s, v0.4s
	fmla v21.4s, v12.4s, v14.4s
	mov v12.16b, v22.16b
	mov v0.16b, v4.16b
	mov v14.16b, v20.16b
	fsqrt v3.4s, v19.4s
	fadd v15.4s, v13.4s, v2.4s
	fmla v18.4s, v6.4s, v13.4s
	fmla v12.4s, v13.4s, v15.4s
	fmla v0.4s, v13.4s, v18.4s
	mov v18.16b, v17.16b
	mov v15.16b, v23.16b
	fdiv v16.4s, v16.4s, v21.4s
	mov v21.16b, v5.16b
	fmla v18.4s, v13.4s, v0.4s
	fmla v21.4s, v13.4s, v12.4s
	fmla v7.4s, v6.4s, v3.4s
	fadd v12.4s, v3.4s, v2.4s
	ldr q2, [sp]
	fmla v14.4s, v13.4s, v18.4s
	fmla v4.4s, v3.4s, v7.4s
	fmla v22.4s, v3.4s, v12.4s
	fmla v15.4s, v13.4s, v21.4s
	ldur q13, [x29, #-80]
	fmin v21.4s, v9.4s, v1.4s
	fmul v18.4s, v30.4s, v13.4s
	fcmgt v30.4s, v2.4s, v30.4s
	fmul v9.4s, v31.4s, v13.4s
	fmla v17.4s, v3.4s, v4.4s
	fmla v5.4s, v3.4s, v22.4s
	fcmgt v31.4s, v2.4s, v31.4s
	fdiv v0.4s, v14.4s, v15.4s
	fmul v12.4s, v11.4s, v13.4s
	fcmgt v11.4s, v2.4s, v11.4s
	fmin v16.4s, v16.4s, v1.4s
	fmul v13.4s, v19.4s, v13.4s
	fcmgt v19.4s, v2.4s, v19.4s
	bif v18.16b, v21.16b, v30.16b
	fcmge v21.4s, v28.4s, v1.4s
	fcmge v28.4s, v29.4s, v1.4s
	fmla v20.4s, v3.4s, v17.4s
	fmla v23.4s, v3.4s, v5.4s
	fcmge v29.4s, v8.4s, v1.4s
	bit v16.16b, v9.16b, v31.16b
	bit v18.16b, v1.16b, v21.16b
	bit v16.16b, v1.16b, v28.16b
	fdiv v3.4s, v20.4s, v23.4s
	fmin v0.4s, v0.4s, v1.4s
	stp q18, q16, [x9, #-32]
	stur s24, [x9, #-20]
	stur s25, [x9, #-4]
	bit v0.16b, v12.16b, v11.16b
	bit v0.16b, v1.16b, v29.16b
	fmin v3.4s, v3.4s, v1.4s
	bit v3.16b, v13.16b, v19.16b
	fcmge v19.4s, v10.4s, v1.4s
	bit v3.16b, v1.16b, v19.16b
	stp q0, q3, [x9]
	str s26, [x9, #12]
	str s27, [x9, #28]
	add x9, x9, #64
	b.ne .LBB16_2
.LBB16_3:
	tst x1, #0xc
	b.eq .LBB16_24
	mov w9, #981467136
	lsr x8, x1, #4
	mov w10, #20545
	fmov s9, w9
	mov w9, #47186
	movk w10, #15175, lsl #16
	movk w9, #16718, lsl #16
	fmov s10, w10
	add x8, x0, x8, lsl #6
	fmov s11, w9
	mov w9, #21845
	fmov s12, #1.00000000
	movk w9, #16085, lsl #16
	and x10, x1, #0xc
	add x20, x8, #8
	stur w9, [x29, #-80]
	mov w9, #21227
	neg x19, x10
	movk w9, #48481, lsl #16
	fmov s13, w9
	mov w9, #2711
	movk w9, #16263, lsl #16
	fmov s14, w9
	b .LBB16_7
.LBB16_5:
	stur xzr, [x20, #-8]
	str wzr, [x20]
.LBB16_6:
	adds x19, x19, #4
	add x20, x20, #16
	b.eq .LBB16_24
.LBB16_7:
	ldr s0, [x20, #4]
	fcmp s0, s9
	b.le .LBB16_5
	fdiv s15, s12, s0
	ldur s0, [x20, #-8]
	movi d8, #0000000000000000
	movi d1, #0000000000000000
	fmul s0, s15, s0
	fcmp s0, #0.0
	b.mi .LBB16_13
	fcmp s0, s10
	b.pl .LBB16_11
	fmul s1, s0, s11
	b .LBB16_13
.LBB16_11:
	fmov s1, #1.00000000
	fcmp s0, s1
	b.pl .LBB16_13
	ldur s1, [x29, #-80]
	bl powf
	fmadd s1, s0, s14, s13
.LBB16_13:
	ldur s0, [x20, #-4]
	stur s1, [x20, #-8]
	fmul s0, s15, s0
	fcmp s0, #0.0
	b.mi .LBB16_18
	fcmp s0, s10
	b.pl .LBB16_16
	fmul s8, s0, s11
	b .LBB16_18
.LBB16_16:
	fmov s8, #1.00000000
	fcmp s0, s8
	b.pl .LBB16_18
	ldur s1, [x29, #-80]
	bl powf
	fmadd s8, s0, s14, s13
.LBB16_18:
	ldr s0, [x20]
	movi d1, #0000000000000000
	stur s8, [x20, #-4]
	fmul s0, s15, s0
	fcmp s0, #0.0
	b.pl .LBB16_20
.LBB16_19:
	str s1, [x20]
	b .LBB16_6
.LBB16_20:
	fcmp s0, s10
	b.pl .LBB16_22
	fmul s1, s0, s11
	str s1, [x20]
	b .LBB16_6
.LBB16_22:
	fmov s1, #1.00000000
	fcmp s0, s1
	b.pl .LBB16_19
	ldur s1, [x29, #-80]
	bl powf
	fmadd s1, s0, s14, s13
	b .LBB16_19
.LBB16_24:
	.cfi_def_cfa wsp, 288
	ldp x20, x19, [sp, #272]
	ldr x28, [sp, #256]
	ldp x29, x30, [sp, #240]
	ldp d9, d8, [sp, #224]
	ldp d11, d10, [sp, #208]
	ldp d13, d12, [sp, #192]
	ldp d15, d14, [sp, #176]
	add sp, sp, #288
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
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
