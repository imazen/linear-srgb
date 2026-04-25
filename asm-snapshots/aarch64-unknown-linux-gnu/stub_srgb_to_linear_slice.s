.section .text.stub_srgb_to_linear_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_slice
	.p2align	2
.type	stub_srgb_to_linear_slice,@function
stub_srgb_to_linear_slice:
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
	mov x10, x0
.LBB14_2:
	ldr q21, [x10]
	mov v23.16b, v4.16b
	mov v25.16b, v5.16b
	mov v26.16b, v17.16b
	subs x9, x9, #16
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
	str q21, [x10], #16
	b.ne .LBB14_2
.LBB14_3:
	ands x19, x8, #0xc
	b.eq .LBB14_11
	and x8, x1, #0x1ffffffffffffffc
	mov w9, #61974
	mov w10, #33681
	add x20, x0, x8, lsl #2
	mov w8, #21227
	movk w9, #15648, lsl #16
	movk w8, #15713, lsl #16
	fmov s9, w9
	mov w9, #39322
	fmov s11, w8
	mov w8, #2711
	movk w10, #15774, lsl #16
	movk w8, #16263, lsl #16
	movk w9, #16409, lsl #16
	fmov s10, w10
	fmov s12, w8
	fmov s8, w9
	mov x21, x20
	b .LBB14_7
.LBB14_5:
	fmul s0, s1, s10
.LBB14_6:
	subs x19, x19, #4
	str s0, [x20]
	mov x20, x21
	b.eq .LBB14_11
.LBB14_7:
	ldr s1, [x21], #4
	movi d0, #0000000000000000
	fcmp s1, #0.0
	b.mi .LBB14_6
	fcmp s1, s9
	b.mi .LBB14_5
	fmov s0, #1.00000000
	fcmp s1, s0
	b.pl .LBB14_6
	fadd s0, s1, s11
	fmov s1, s8
	fdiv s0, s0, s12
	bl powf
	b .LBB14_6
.LBB14_11:
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
