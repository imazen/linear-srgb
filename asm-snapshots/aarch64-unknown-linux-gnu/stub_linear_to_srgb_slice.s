.section .text.stub_linear_to_srgb_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_slice
	.p2align	2
.type	stub_linear_to_srgb_slice,@function
stub_linear_to_srgb_slice:
	.cfi_startproc
	str d14, [sp, #-96]!
	.cfi_def_cfa_offset 96
	stp d13, d12, [sp, #8]
	stp d11, d10, [sp, #24]
	stp d9, d8, [sp, #40]
	stp x29, x30, [sp, #56]
	str x21, [sp, #72]
	stp x20, x19, [sp, #80]
	add x29, sp, #56
	.cfi_def_cfa w29, 40
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w30, -32
	.cfi_offset w29, -40
	.cfi_offset b8, -48
	.cfi_offset b9, -56
	.cfi_offset b10, -64
	.cfi_offset b11, -72
	.cfi_offset b12, -80
	.cfi_offset b13, -88
	.cfi_offset b14, -96
	lsl x8, x1, #2
	ands x9, x8, #0x7fffffffffffffc0
	b.eq .LBB10_3
	mov w10, #47186
	mov w11, #25800
	mov w12, #14394
	movk w10, #16718, lsl #16
	movk w11, #16863, lsl #16
	movk w12, #15807, lsl #16
	dup v2.4s, w10
	dup v3.4s, w11
	mov w10, #5570
	mov w11, #10701
	movk w10, #16968, lsl #16
	dup v6.4s, w12
	movk w11, #16697, lsl #16
	dup v4.4s, w10
	mov w10, #15682
	dup v5.4s, w11
	mov w11, #15285
	mov w12, #7182
	movk w10, #48222, lsl #16
	movk w11, #16906, lsl #16
	movk w12, #16947, lsl #16
	dup v7.4s, w10
	dup v16.4s, w11
	dup v17.4s, w12
	mov w10, #64401
	mov w11, #55785
	mov w12, #20545
	movi v0.2d, #0000000000000000
	fmov v1.4s, #1.00000000
	movk w10, #16655, lsl #16
	movk w11, #16006, lsl #16
	movk w12, #15175, lsl #16
	dup v18.4s, w10
	dup v19.4s, w11
	dup v20.4s, w12
	add x9, x0, x9
	mov x10, x0
.LBB10_2:
	ldp q22, q21, [x10]
	mov v26.16b, v4.16b
	mov v28.16b, v5.16b
	mov v30.16b, v17.16b
	mov v31.16b, v4.16b
	mov v10.16b, v5.16b
	mov v11.16b, v17.16b
	mov v13.16b, v7.16b
	fmax v23.4s, v22.4s, v0.4s
	fmax v24.4s, v21.4s, v0.4s
	mov v14.16b, v19.16b
	fcmge v22.4s, v22.4s, v1.4s
	fcmge v21.4s, v21.4s, v1.4s
	fmin v23.4s, v23.4s, v1.4s
	fmin v24.4s, v24.4s, v1.4s
	fsqrt v25.4s, v23.4s
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
	ldp q26, q25, [x10, #32]
	fmla v10.4s, v29.4s, v31.4s
	fmla v11.4s, v29.4s, v9.4s
	mov v31.16b, v6.16b
	mov v9.16b, v18.16b
	fdiv v27.4s, v28.4s, v30.4s
	fmax v28.4s, v26.4s, v0.4s
	fmax v8.4s, v25.4s, v0.4s
	fmla v31.4s, v29.4s, v10.4s
	mov v10.16b, v4.16b
	fcmge v26.4s, v26.4s, v1.4s
	fmla v9.4s, v29.4s, v11.4s
	fcmge v25.4s, v25.4s, v1.4s
	fmin v28.4s, v28.4s, v1.4s
	fmin v8.4s, v8.4s, v1.4s
	fmla v13.4s, v29.4s, v31.4s
	mov v31.16b, v17.16b
	fmla v14.4s, v29.4s, v9.4s
	mov v29.16b, v5.16b
	fsqrt v30.4s, v28.4s
	fmin v27.4s, v27.4s, v1.4s
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
	bsl v23.16b, v31.16b, v27.16b
	fdiv v30.4s, v30.4s, v10.4s
	fmul v10.4s, v24.4s, v2.4s
	fcmgt v24.4s, v20.4s, v24.4s
	fmin v29.4s, v29.4s, v1.4s
	mov v27.16b, v28.16b
	mov v28.16b, v8.16b
	bsl v22.16b, v1.16b, v23.16b
	mov v23.16b, v26.16b
	bsl v24.16b, v10.16b, v9.16b
	bsl v27.16b, v11.16b, v29.16b
	bsl v21.16b, v1.16b, v24.16b
	mov v24.16b, v25.16b
	bsl v23.16b, v1.16b, v27.16b
	stp q22, q21, [x10]
	fmin v30.4s, v30.4s, v1.4s
	bsl v28.16b, v12.16b, v30.16b
	bsl v24.16b, v1.16b, v28.16b
	stp q23, q24, [x10, #32]
	add x10, x10, #64
	cmp x10, x9
	b.ne .LBB10_2
.LBB10_3:
	ands x19, x8, #0x3c
	b.eq .LBB10_11
	and x8, x1, #0x1ffffffffffffff0
	mov w9, #20545
	mov w10, #47186
	add x20, x0, x8, lsl #2
	mov w8, #21845
	movk w9, #15175, lsl #16
	movk w8, #16085, lsl #16
	fmov s9, w9
	mov w9, #2711
	fmov s8, w8
	mov w8, #21227
	movk w10, #16718, lsl #16
	movk w8, #48481, lsl #16
	movk w9, #16263, lsl #16
	fmov s10, w10
	fmov s11, w8
	fmov s12, w9
	mov x21, x20
	b .LBB10_7
.LBB10_5:
	fmul s1, s0, s10
.LBB10_6:
	subs x19, x19, #4
	str s1, [x20]
	mov x20, x21
	b.eq .LBB10_11
.LBB10_7:
	ldr s0, [x21], #4
	movi d1, #0000000000000000
	fcmp s0, #0.0
	b.mi .LBB10_6
	fcmp s0, s9
	b.mi .LBB10_5
	fmov s1, #1.00000000
	fcmp s0, s1
	b.pl .LBB10_6
	fmov s1, s8
	bl powf
	fmadd s1, s0, s12, s11
	b .LBB10_6
.LBB10_11:
	.cfi_def_cfa wsp, 96
	ldp x20, x19, [sp, #80]
	ldr x21, [sp, #72]
	ldp x29, x30, [sp, #56]
	ldp d9, d8, [sp, #40]
	ldp d11, d10, [sp, #24]
	ldp d13, d12, [sp, #8]
	ldr d14, [sp], #96
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
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
