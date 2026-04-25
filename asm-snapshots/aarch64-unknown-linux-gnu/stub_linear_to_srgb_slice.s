.section .text.stub_linear_to_srgb_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_slice
	.p2align	2
.type	stub_linear_to_srgb_slice,@function
stub_linear_to_srgb_slice:
	.cfi_startproc
	str d12, [sp, #-80]!
	.cfi_def_cfa_offset 80
	stp d11, d10, [sp, #8]
	stp d9, d8, [sp, #24]
	stp x29, x30, [sp, #40]
	str x21, [sp, #56]
	stp x20, x19, [sp, #64]
	add x29, sp, #40
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
	lsl x8, x1, #2
	ands x9, x8, #0x7ffffffffffffff0
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
	mov x10, x0
.LBB10_2:
	ldr q21, [x10]
	mov v24.16b, v4.16b
	mov v26.16b, v5.16b
	mov v27.16b, v17.16b
	subs x9, x9, #16
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
	str q21, [x10], #16
	b.ne .LBB10_2
.LBB10_3:
	ands x19, x8, #0xc
	b.eq .LBB10_11
	and x8, x1, #0x1ffffffffffffffc
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
	.cfi_def_cfa wsp, 80
	ldp x20, x19, [sp, #64]
	ldr x21, [sp, #56]
	ldp x29, x30, [sp, #40]
	ldp d9, d8, [sp, #24]
	ldp d11, d10, [sp, #8]
	ldr d12, [sp], #80
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
	ret
