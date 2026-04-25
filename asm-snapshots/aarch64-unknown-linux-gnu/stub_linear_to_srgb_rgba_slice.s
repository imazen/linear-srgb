.section .text.stub_linear_to_srgb_rgba_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_rgba_slice
	.p2align	2
.type	stub_linear_to_srgb_rgba_slice,@function
stub_linear_to_srgb_rgba_slice:
	.cfi_startproc
	str d14, [sp, #-96]!
	.cfi_def_cfa_offset 96
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	stp x29, x30, [sp, #64]
	stp x20, x19, [sp, #80]
	add x29, sp, #64
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	.cfi_offset b8, -40
	.cfi_offset b9, -48
	.cfi_offset b10, -56
	.cfi_offset b11, -64
	.cfi_offset b12, -72
	.cfi_offset b13, -80
	.cfi_offset b14, -96
	lsl x8, x1, #2
	ands x8, x8, #0x7fffffffffffffc0
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
	add x9, x0, #32
.LBB9_2:
	ldur q21, [x9, #-32]
	mov v26.16b, v4.16b
	mov v28.16b, v5.16b
	mov v30.16b, v17.16b
	mov v31.16b, v4.16b
	subs x8, x8, #64
	fmax v22.4s, v21.4s, v0.4s
	mov v10.16b, v5.16b
	mov v11.16b, v17.16b
	mov v13.16b, v7.16b
	mov v14.16b, v19.16b
	fcmge v21.4s, v21.4s, v1.4s
	fmin v23.4s, v22.4s, v1.4s
	ldur q22, [x9, #-16]
	fmax v24.4s, v22.4s, v0.4s
	fcmge v22.4s, v22.4s, v1.4s
	fsqrt v25.4s, v23.4s
	fmin v24.4s, v24.4s, v1.4s
	fsqrt v29.4s, v24.4s
	fmla v26.4s, v3.4s, v25.4s
	fadd v27.4s, v25.4s, v16.4s
	fmla v28.4s, v25.4s, v26.4s
	fmla v30.4s, v25.4s, v27.4s
	mov v26.16b, v6.16b
	mov v27.16b, v18.16b
	fmla v26.4s, v25.4s, v28.4s
	fmla v27.4s, v25.4s, v30.4s
	mov v28.16b, v7.16b
	mov v30.16b, v19.16b
	fmla v31.4s, v3.4s, v29.4s
	fadd v9.4s, v29.4s, v16.4s
	fmla v28.4s, v25.4s, v26.4s
	fmla v30.4s, v25.4s, v27.4s
	ldr q25, [x9]
	fmla v10.4s, v29.4s, v31.4s
	fmla v11.4s, v29.4s, v9.4s
	mov v31.16b, v6.16b
	fmax v27.4s, v25.4s, v0.4s
	mov v9.16b, v18.16b
	fcmge v25.4s, v25.4s, v1.4s
	fdiv v26.4s, v28.4s, v30.4s
	fmla v31.4s, v29.4s, v10.4s
	fmla v9.4s, v29.4s, v11.4s
	mov v10.16b, v4.16b
	fmin v28.4s, v27.4s, v1.4s
	ldr q27, [x9, #16]
	fmax v8.4s, v27.4s, v0.4s
	fcmge v27.4s, v27.4s, v1.4s
	fmla v13.4s, v29.4s, v31.4s
	fmla v14.4s, v29.4s, v9.4s
	mov v29.16b, v5.16b
	mov v31.16b, v17.16b
	fmin v8.4s, v8.4s, v1.4s
	fsqrt v30.4s, v28.4s
	fmin v26.4s, v26.4s, v1.4s
	fsqrt v12.4s, v8.4s
	fmla v10.4s, v3.4s, v30.4s
	fadd v11.4s, v30.4s, v16.4s
	fmla v29.4s, v30.4s, v10.4s
	fmla v31.4s, v30.4s, v11.4s
	mov v10.16b, v6.16b
	mov v11.16b, v18.16b
	fdiv v9.4s, v13.4s, v14.4s
	mov v13.16b, v7.16b
	mov v14.16b, v19.16b
	fmla v10.4s, v30.4s, v29.4s
	fmla v11.4s, v30.4s, v31.4s
	mov v29.16b, v4.16b
	fadd v31.4s, v12.4s, v16.4s
	fmla v29.4s, v3.4s, v12.4s
	fmla v13.4s, v30.4s, v10.4s
	fmla v14.4s, v30.4s, v11.4s
	mov v30.16b, v5.16b
	mov v10.16b, v17.16b
	mov v11.16b, v18.16b
	fmla v30.4s, v12.4s, v29.4s
	fmla v10.4s, v12.4s, v31.4s
	mov v31.16b, v6.16b
	fdiv v29.4s, v13.4s, v14.4s
	fmla v31.4s, v12.4s, v30.4s
	mov v30.16b, v7.16b
	fmin v9.4s, v9.4s, v1.4s
	fmla v11.4s, v12.4s, v10.4s
	mov v10.16b, v19.16b
	fmla v30.4s, v12.4s, v31.4s
	fmul v31.4s, v23.4s, v2.4s
	fcmgt v23.4s, v20.4s, v23.4s
	fmla v10.4s, v12.4s, v11.4s
	fmul v11.4s, v28.4s, v2.4s
	fcmgt v28.4s, v20.4s, v28.4s
	fmul v12.4s, v8.4s, v2.4s
	fcmgt v8.4s, v20.4s, v8.4s
	bsl v23.16b, v31.16b, v26.16b
	fdiv v30.4s, v30.4s, v10.4s
	fmul v10.4s, v24.4s, v2.4s
	fcmgt v24.4s, v20.4s, v24.4s
	fmin v29.4s, v29.4s, v1.4s
	mov v26.16b, v28.16b
	mov v28.16b, v8.16b
	bsl v21.16b, v1.16b, v23.16b
	mov v23.16b, v25.16b
	ldur s25, [x9, #-20]
	bsl v24.16b, v10.16b, v9.16b
	bsl v26.16b, v11.16b, v29.16b
	bsl v22.16b, v1.16b, v24.16b
	mov v24.16b, v27.16b
	bsl v23.16b, v1.16b, v26.16b
	ldur s26, [x9, #-4]
	stp q21, q22, [x9, #-32]
	ldr s21, [x9, #12]
	ldr s22, [x9, #28]
	fmin v30.4s, v30.4s, v1.4s
	stur s25, [x9, #-20]
	stur s26, [x9, #-4]
	bsl v28.16b, v12.16b, v30.16b
	bsl v24.16b, v1.16b, v28.16b
	stp q23, q24, [x9]
	str s21, [x9, #12]
	str s22, [x9, #28]
	add x9, x9, #64
	b.ne .LBB9_2
.LBB9_3:
	tst x1, #0xc
	b.eq .LBB9_21
	mov w10, #20545
	and x8, x1, #0x1ffffffffffffff0
	and x9, x1, #0xc
	movk w10, #15175, lsl #16
	mov w11, #47186
	add x19, x0, x8, lsl #2
	neg x20, x9
	fmov s9, w10
	mov w8, #21845
	mov w9, #21227
	mov w10, #2711
	movk w11, #16718, lsl #16
	movk w8, #16085, lsl #16
	movk w9, #48481, lsl #16
	movk w10, #16263, lsl #16
	fmov s10, w11
	fmov s8, w8
	fmov s11, w9
	fmov s12, w10
	b .LBB9_7
.LBB9_5:
	fmul s1, s0, s10
.LBB9_6:
	adds x20, x20, #4
	str s1, [x19, #8]
	add x19, x19, #16
	b.eq .LBB9_21
.LBB9_7:
	ldr s0, [x19]
	movi d13, #0000000000000000
	movi d1, #0000000000000000
	fcmp s0, #0.0
	b.mi .LBB9_12
	fcmp s0, s9
	b.pl .LBB9_10
	fmul s1, s0, s10
	b .LBB9_12
.LBB9_10:
	fmov s1, #1.00000000
	fcmp s0, s1
	b.pl .LBB9_12
	fmov s1, s8
	bl powf
	fmadd s1, s0, s12, s11
.LBB9_12:
	ldr s0, [x19, #4]
	str s1, [x19]
	fcmp s0, #0.0
	b.mi .LBB9_17
	fcmp s0, s9
	b.pl .LBB9_15
	fmul s13, s0, s10
	b .LBB9_17
.LBB9_15:
	fmov s13, #1.00000000
	fcmp s0, s13
	b.pl .LBB9_17
	fmov s1, s8
	bl powf
	fmadd s13, s0, s12, s11
.LBB9_17:
	ldr s0, [x19, #8]
	movi d1, #0000000000000000
	str s13, [x19, #4]
	fcmp s0, #0.0
	b.mi .LBB9_6
	fcmp s0, s9
	b.mi .LBB9_5
	fmov s1, #1.00000000
	fcmp s0, s1
	b.pl .LBB9_6
	fmov s1, s8
	bl powf
	fmadd s1, s0, s12, s11
	b .LBB9_6
.LBB9_21:
	.cfi_def_cfa wsp, 96
	ldp x20, x19, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldr d14, [sp], #96
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	.cfi_restore b13
	.cfi_restore b14
	ret
