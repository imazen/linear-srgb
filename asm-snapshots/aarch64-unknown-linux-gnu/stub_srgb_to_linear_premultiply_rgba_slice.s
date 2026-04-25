.section .text.stub_srgb_to_linear_premultiply_rgba_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_premultiply_rgba_slice
	.p2align	2
.type	stub_srgb_to_linear_premultiply_rgba_slice,@function
stub_srgb_to_linear_premultiply_rgba_slice:
	.cfi_startproc
	sub sp, sp, #160
	.cfi_def_cfa_offset 160
	stp d15, d14, [sp, #64]
	stp d13, d12, [sp, #80]
	stp d11, d10, [sp, #96]
	stp d9, d8, [sp, #112]
	stp x29, x30, [sp, #128]
	stp x20, x19, [sp, #144]
	add x29, sp, #128
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
	.cfi_offset b14, -88
	.cfi_offset b15, -96
	lsl x8, x1, #2
	ands x8, x8, #0x7fffffffffffffc0
	b.eq .LBB12_3
	mov w9, #33681
	mov w10, #32277
	mov w11, #6038
	movk w9, #15774, lsl #16
	movk w10, #17060, lsl #16
	movk w11, #16207, lsl #16
	dup v2.4s, w9
	dup v1.4s, w10
	mov w9, #5734
	mov w10, #9853
	movk w9, #17033, lsl #16
	dup v6.4s, w11
	movk w10, #16718, lsl #16
	mov w11, #6038
	fmov v0.4s, #1.00000000
	stp q1, q2, [sp, #16]
	dup v1.4s, w9
	dup v5.4s, w10
	mov w9, #2423
	mov w10, #1338
	movk w11, #16983, lsl #16
	movk w9, #15497, lsl #16
	movk w10, #49380, lsl #16
	dup v17.4s, w11
	dup v7.4s, w9
	dup v16.4s, w10
	mov w9, #41246
	mov w10, #19964
	mov w11, #61974
	movk w9, #17089, lsl #16
	movk w10, #16800, lsl #16
	movk w11, #15648, lsl #16
	dup v18.4s, w9
	dup v19.4s, w10
	dup v20.4s, w11
	add x9, x0, #32
	str q1, [sp]
.LBB12_2:
	movi v1.2d, #0000000000000000
	ldur q2, [x9, #-32]
	mov v27.16b, v5.16b
	mov v28.16b, v17.16b
	mov v29.16b, v18.16b
	ldr q4, [sp, #32]
	str q2, [sp, #48]
	mov v30.16b, v7.16b
	mov v31.16b, v19.16b
	mov v9.16b, v7.16b
	mov v10.16b, v19.16b
	ldur s13, [x9, #-20]
	fmax v22.4s, v2.4s, v1.4s
	ldp q3, q2, [sp]
	mov v12.16b, v19.16b
	ldur s14, [x9, #-4]
	ldr s15, [x9, #12]
	subs x8, x8, #64
	mov v24.16b, v3.16b
	fmin v23.4s, v22.4s, v0.4s
	ldur q22, [x9, #-16]
	fmax v26.4s, v22.4s, v1.4s
	fcmge v22.4s, v22.4s, v0.4s
	fmla v24.4s, v2.4s, v23.4s
	fadd v25.4s, v23.4s, v16.4s
	fmla v27.4s, v23.4s, v24.4s
	fmla v28.4s, v23.4s, v25.4s
	fmin v25.4s, v26.4s, v0.4s
	mov v26.16b, v6.16b
	ldr q24, [x9]
	fmax v8.4s, v24.4s, v1.4s
	fcmge v24.4s, v24.4s, v0.4s
	fmla v26.4s, v23.4s, v27.4s
	fmla v29.4s, v23.4s, v28.4s
	mov v27.16b, v3.16b
	fadd v28.4s, v25.4s, v16.4s
	fmla v27.4s, v2.4s, v25.4s
	fmla v30.4s, v23.4s, v26.4s
	fmla v31.4s, v23.4s, v29.4s
	mov v26.16b, v5.16b
	mov v29.16b, v17.16b
	fmla v26.4s, v25.4s, v27.4s
	fmin v27.4s, v8.4s, v0.4s
	fmla v29.4s, v25.4s, v28.4s
	fdiv v28.4s, v30.4s, v31.4s
	mov v30.16b, v6.16b
	mov v31.16b, v18.16b
	fmla v30.4s, v25.4s, v26.4s
	fadd v8.4s, v27.4s, v16.4s
	ldr q26, [x9, #16]
	fmla v31.4s, v25.4s, v29.4s
	mov v29.16b, v3.16b
	fmul v21.4s, v27.4s, v4.4s
	fmax v11.4s, v26.4s, v1.4s
	ldr s1, [x9, #28]
	fmla v29.4s, v2.4s, v27.4s
	fmla v9.4s, v25.4s, v30.4s
	mov v30.16b, v5.16b
	fmla v10.4s, v25.4s, v31.4s
	mov v31.16b, v17.16b
	fmla v30.4s, v27.4s, v29.4s
	fmla v31.4s, v27.4s, v8.4s
	fmin v29.4s, v11.4s, v0.4s
	fdiv v8.4s, v9.4s, v10.4s
	mov v9.16b, v6.16b
	mov v10.16b, v18.16b
	mov v11.16b, v7.16b
	fmin v28.4s, v28.4s, v0.4s
	fmla v9.4s, v27.4s, v30.4s
	fmla v10.4s, v27.4s, v31.4s
	mov v30.16b, v3.16b
	fadd v31.4s, v29.4s, v16.4s
	fmul v3.4s, v25.4s, v4.4s
	fcmgt v25.4s, v20.4s, v25.4s
	fmla v30.4s, v2.4s, v29.4s
	fmul v2.4s, v23.4s, v4.4s
	fcmgt v23.4s, v20.4s, v23.4s
	fmla v11.4s, v27.4s, v9.4s
	fmla v12.4s, v27.4s, v10.4s
	mov v9.16b, v5.16b
	mov v10.16b, v17.16b
	fcmgt v27.4s, v20.4s, v27.4s
	fmul v4.4s, v29.4s, v4.4s
	fmla v9.4s, v29.4s, v30.4s
	bif v2.16b, v28.16b, v23.16b
	ldr q23, [sp, #48]
	fmla v10.4s, v29.4s, v31.4s
	fdiv v30.4s, v11.4s, v12.4s
	mov v31.16b, v6.16b
	mov v11.16b, v18.16b
	fmov v12.4s, #1.00000000
	fcmge v23.4s, v23.4s, v0.4s
	fmin v8.4s, v8.4s, v0.4s
	fmla v31.4s, v29.4s, v9.4s
	mov v9.16b, v7.16b
	fmla v11.4s, v29.4s, v10.4s
	mov v10.16b, v19.16b
	mov v12.s[0], v1.s[0]
	bit v2.16b, v0.16b, v23.16b
	bif v3.16b, v8.16b, v25.16b
	fcmge v25.4s, v26.4s, v0.4s
	fmla v9.4s, v29.4s, v31.4s
	fmla v10.4s, v29.4s, v11.4s
	fmov v11.4s, #1.00000000
	fcmgt v29.4s, v20.4s, v29.4s
	mov v12.s[1], v1.s[0]
	bit v3.16b, v0.16b, v22.16b
	fdiv v31.4s, v9.4s, v10.4s
	fmov v9.4s, #1.00000000
	fmov v10.4s, #1.00000000
	mov v11.s[0], v15.s[0]
	fmin v30.4s, v30.4s, v0.4s
	mov v12.s[2], v1.s[0]
	mov v9.s[0], v13.s[0]
	mov v10.s[0], v14.s[0]
	bif v21.16b, v30.16b, v27.16b
	mov v11.s[1], v15.s[0]
	mov v9.s[1], v13.s[0]
	mov v10.s[1], v14.s[0]
	mov v11.s[2], v15.s[0]
	bit v21.16b, v0.16b, v24.16b
	mov v9.s[2], v13.s[0]
	mov v10.s[2], v14.s[0]
	fmin v31.4s, v31.4s, v0.4s
	fmul v21.4s, v11.4s, v21.4s
	fmul v2.4s, v9.4s, v2.4s
	fmul v3.4s, v10.4s, v3.4s
	bif v4.16b, v31.16b, v29.16b
	stp q2, q3, [x9, #-32]
	bit v4.16b, v0.16b, v25.16b
	stur s13, [x9, #-20]
	stur s14, [x9, #-4]
	fmul v4.4s, v12.4s, v4.4s
	stp q21, q4, [x9]
	str s15, [x9, #12]
	str s1, [x9, #28]
	add x9, x9, #64
	b.ne .LBB12_2
.LBB12_3:
	tst x1, #0xc
	b.eq .LBB12_21
	lsr x8, x1, #4
	mov w10, #61974
	mov w11, #33681
	mov w12, #21227
	mov w13, #2711
	mov w14, #39322
	movk w10, #15648, lsl #16
	movk w11, #15774, lsl #16
	movk w12, #15713, lsl #16
	movk w13, #16263, lsl #16
	movk w14, #16409, lsl #16
	add x8, x0, x8, lsl #6
	fmov s10, w10
	fmov s11, w11
	fmov s12, w12
	fmov s13, w13
	fmov s8, w14
	and x9, x1, #0xc
	neg x19, x9
	add x20, x8, #8
	b .LBB12_7
.LBB12_5:
	fmul s0, s1, s11
.LBB12_6:
	fmul s0, s14, s0
	adds x19, x19, #4
	str s0, [x20], #16
	b.eq .LBB12_21
.LBB12_7:
	ldur s1, [x20, #-8]
	movi d9, #0000000000000000
	movi d0, #0000000000000000
	ldr s14, [x20, #4]
	fcmp s1, #0.0
	b.mi .LBB12_12
	fcmp s1, s10
	b.pl .LBB12_10
	fmul s0, s1, s11
	b .LBB12_12
.LBB12_10:
	fmov s0, #1.00000000
	fcmp s1, s0
	b.pl .LBB12_12
	fadd s0, s1, s12
	fmov s1, s8
	fdiv s0, s0, s13
	bl powf
.LBB12_12:
	ldur s1, [x20, #-4]
	fmul s0, s14, s0
	fcmp s1, #0.0
	stur s0, [x20, #-8]
	b.mi .LBB12_17
	fcmp s1, s10
	b.pl .LBB12_15
	fmul s9, s1, s11
	b .LBB12_17
.LBB12_15:
	fmov s9, #1.00000000
	fcmp s1, s9
	b.pl .LBB12_17
	fadd s0, s1, s12
	fmov s1, s8
	fdiv s0, s0, s13
	bl powf
	fmov s9, s0
.LBB12_17:
	ldr s1, [x20]
	fmul s2, s14, s9
	movi d0, #0000000000000000
	fcmp s1, #0.0
	stur s2, [x20, #-4]
	b.mi .LBB12_6
	fcmp s1, s10
	b.mi .LBB12_5
	fmov s0, #1.00000000
	fcmp s1, s0
	b.pl .LBB12_6
	fadd s0, s1, s12
	fmov s1, s8
	fdiv s0, s0, s13
	bl powf
	b .LBB12_6
.LBB12_21:
	.cfi_def_cfa wsp, 160
	ldp x20, x19, [sp, #144]
	ldp x29, x30, [sp, #128]
	ldp d9, d8, [sp, #112]
	ldp d11, d10, [sp, #96]
	ldp d13, d12, [sp, #80]
	ldp d15, d14, [sp, #64]
	add sp, sp, #160
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
	.cfi_restore b15
	ret
