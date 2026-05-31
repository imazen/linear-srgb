.section .text.stub_linear_to_srgb_rgba_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_rgba_slice
	.p2align	2
.type	stub_linear_to_srgb_rgba_slice,@function
stub_linear_to_srgb_rgba_slice:
	.cfi_startproc
	str d14, [sp, #-64]!
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
	.cfi_offset b14, -64
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
	adrp x11, .LCPI9_0
	and x8, x1, #0x1ffffffffffffff0
	movk w10, #15175, lsl #16
	ldr d1, [x11, :lo12:.LCPI9_0]
	adrp x11, .LCPI9_3
	fmov s0, w10
	adrp x10, .LCPI9_1
	ldr q4, [x11, :lo12:.LCPI9_3]
	ldr d2, [x10, :lo12:.LCPI9_1]
	adrp x10, .LCPI9_2
	mov x11, #260141874151424
	ldr q3, [x10, :lo12:.LCPI9_2]
	adrp x10, .LCPI9_4
	movk x11, #16443, lsl #48
	ldr q5, [x10, :lo12:.LCPI9_4]
	mov w10, #47186
	fmov d6, x11
	movk w10, #16718, lsl #16
	and x9, x1, #0xc
	add x8, x0, x8, lsl #2
	fmov s7, w10
	neg x9, x9
	b .LBB9_7
.LBB9_5:
	fsqrt s16, s18
	fcvt d16, s16
	fmadd d17, d16, d6, d1
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
.LBB9_6:
	adds x9, x9, #4
	str s17, [x8, #8]
	add x8, x8, #16
	b.eq .LBB9_21
.LBB9_7:
	ldr s17, [x8]
	movi d16, #0000000000000000
	movi d18, #0000000000000000
	fcmp s17, #0.0
	b.mi .LBB9_11
	fmov s18, #1.00000000
	fcmp s17, s18
	b.ge .LBB9_11
	fcmp s17, s0
	b.ls .LBB9_15
	fsqrt s17, s17
	fcvt d17, s17
	fmadd d18, d17, d6, d1
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
.LBB9_11:
	ldr s17, [x8, #4]
	str s18, [x8]
	fcmp s17, #0.0
	b.mi .LBB9_17
.LBB9_12:
	fmov s16, #1.00000000
	fcmp s17, s16
	b.ge .LBB9_17
	fcmp s17, s0
	b.ls .LBB9_16
	fsqrt s16, s17
	fcvt d16, s16
	fmadd d17, d16, d6, d1
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
	b .LBB9_17
.LBB9_15:
	fmul s18, s17, s7
	ldr s17, [x8, #4]
	str s18, [x8]
	fcmp s17, #0.0
	b.pl .LBB9_12
	b .LBB9_17
.LBB9_16:
	fmul s16, s17, s7
.LBB9_17:
	ldr s18, [x8, #8]
	movi d17, #0000000000000000
	str s16, [x8, #4]
	fcmp s18, #0.0
	b.mi .LBB9_6
	fmov s17, #1.00000000
	fcmp s18, s17
	b.ge .LBB9_6
	fcmp s18, s0
	b.hi .LBB9_5
	fmul s17, s18, s7
	b .LBB9_6
.LBB9_21:
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldr d14, [sp], #64
	.cfi_def_cfa_offset 0
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	.cfi_restore b13
	.cfi_restore b14
	ret
