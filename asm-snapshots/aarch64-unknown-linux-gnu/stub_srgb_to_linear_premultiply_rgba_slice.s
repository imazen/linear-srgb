.section .text.stub_srgb_to_linear_premultiply_rgba_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_premultiply_rgba_slice
	.p2align	2
.type	stub_srgb_to_linear_premultiply_rgba_slice,@function
stub_srgb_to_linear_premultiply_rgba_slice:
	.cfi_startproc
	sub sp, sp, #336
	.cfi_def_cfa_offset 336
	stp d15, d14, [sp, #208]
	stp d13, d12, [sp, #224]
	stp d11, d10, [sp, #240]
	stp d9, d8, [sp, #256]
	stp x29, x30, [sp, #272]
	stp x28, x23, [sp, #288]
	stp x22, x21, [sp, #304]
	stp x20, x19, [sp, #320]
	add x29, sp, #272
	.cfi_def_cfa w29, 64
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w28, -48
	.cfi_offset w30, -56
	.cfi_offset w29, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	.cfi_offset b10, -88
	.cfi_offset b11, -96
	.cfi_offset b12, -104
	.cfi_offset b13, -112
	.cfi_offset b14, -120
	.cfi_offset b15, -128
	.cfi_remember_state
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB12_15
	cmp w8, #2
	b.ne .LBB12_14
.LBB12_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x21, x1, #0xc
	lsr x22, x1, #4
	add x19, x0, x8, lsl #2
	lsl x20, x21, #2
	b.eq .LBB12_4
	mov w8, #64
	mov x23, x0
	mov w1, wzr
	sub x2, x8, x20
	sub x8, x29, #128
	add x0, x8, x20
	bl memset
	sub x0, x29, #128
	mov x1, x19
	mov x2, x20
	bl memcpy
	ldp q0, q1, [x29, #-128]
	mov x0, x23
	mov w9, #1
	stp q0, q1, [sp, #80]
	ldp q0, q2, [x29, #-96]
	stp q0, q2, [sp, #112]
	b .LBB12_5
.LBB12_4:
	mov w9, wzr
.LBB12_5:
	mov w8, #33681
	mov w10, #32277
	fmov v1.4s, #1.00000000
	movk w8, #15774, lsl #16
	movk w10, #17060, lsl #16
	cmp x21, #0
	dup v2.4s, w8
	dup v0.4s, w10
	mov w8, #5734
	movk w8, #17033, lsl #16
	mov w10, #6038
	str w9, [sp, #76]
	movk w10, #16207, lsl #16
	stp q0, q2, [sp, #32]
	dup v2.4s, w8
	mov w8, #9853
	movk w8, #16718, lsl #16
	dup v6.4s, w10
	mov w10, #6038
	dup v0.4s, w8
	mov w8, #2423
	movk w10, #16983, lsl #16
	movk w8, #15497, lsl #16
	dup v17.4s, w10
	mov w10, #61974
	dup v7.4s, w8
	mov w8, #1338
	movk w10, #15648, lsl #16
	movk w8, #49380, lsl #16
	dup v20.4s, w10
	add x10, sp, #76
	dup v16.4s, w8
	mov w8, #41246
	add x1, x10, #4
	movk w8, #17089, lsl #16
	csel x9, xzr, x1, eq
	stp q0, q2, [sp]
	dup v18.4s, w8
	mov w8, #19964
	movk w8, #16800, lsl #16
	dup v19.4s, w8
	add x8, x0, x22, lsl #6
	b .LBB12_8
.LBB12_6:
	add x10, x0, #64
.LBB12_7:
	movi v0.2d, #0000000000000000
	ldr q21, [x0]
	ldr q4, [sp]
	ldp q3, q2, [sp, #16]
	mov v28.16b, v17.16b
	mov v27.16b, v4.16b
	mov v29.16b, v18.16b
	mov v30.16b, v7.16b
	mov v31.16b, v19.16b
	mov v9.16b, v7.16b
	mov v10.16b, v19.16b
	fmax v22.4s, v21.4s, v0.4s
	mov v24.16b, v3.16b
	ldr s13, [x0, #12]
	mov v12.16b, v17.16b
	ldr s14, [x0, #28]
	ldr s15, [x0, #44]
	ldr q5, [sp, #48]
	fcmge v21.4s, v21.4s, v1.4s
	fmin v23.4s, v22.4s, v1.4s
	ldr q22, [x0, #16]
	fmax v26.4s, v22.4s, v0.4s
	fcmge v22.4s, v22.4s, v1.4s
	fmla v24.4s, v2.4s, v23.4s
	fadd v25.4s, v23.4s, v16.4s
	fmla v27.4s, v23.4s, v24.4s
	fmla v28.4s, v23.4s, v25.4s
	fmin v25.4s, v26.4s, v1.4s
	mov v26.16b, v6.16b
	ldr q24, [x0, #32]
	fmax v8.4s, v24.4s, v0.4s
	fmla v26.4s, v23.4s, v27.4s
	fmla v29.4s, v23.4s, v28.4s
	mov v27.16b, v3.16b
	fadd v28.4s, v25.4s, v16.4s
	fmla v27.4s, v2.4s, v25.4s
	fmla v30.4s, v23.4s, v26.4s
	fmla v31.4s, v23.4s, v29.4s
	mov v26.16b, v4.16b
	mov v29.16b, v17.16b
	fmla v26.4s, v25.4s, v27.4s
	fmin v27.4s, v8.4s, v1.4s
	fmla v29.4s, v25.4s, v28.4s
	fdiv v28.4s, v30.4s, v31.4s
	mov v30.16b, v6.16b
	mov v31.16b, v18.16b
	fmla v30.4s, v25.4s, v26.4s
	fadd v8.4s, v27.4s, v16.4s
	ldr q26, [x0, #48]
	fmla v31.4s, v25.4s, v29.4s
	mov v29.16b, v3.16b
	fmax v11.4s, v26.4s, v0.4s
	ldr s0, [x0, #60]
	fmla v29.4s, v2.4s, v27.4s
	fmla v9.4s, v25.4s, v30.4s
	fmla v10.4s, v25.4s, v31.4s
	mov v31.16b, v4.16b
	fmla v12.4s, v27.4s, v8.4s
	mov v8.16b, v6.16b
	fmla v31.4s, v27.4s, v29.4s
	fmin v29.4s, v11.4s, v1.4s
	mov v11.16b, v7.16b
	fdiv v30.4s, v9.4s, v10.4s
	mov v9.16b, v18.16b
	fmin v28.4s, v28.4s, v1.4s
	fmla v8.4s, v27.4s, v31.4s
	fmla v9.4s, v27.4s, v12.4s
	mov v31.16b, v3.16b
	mov v12.16b, v19.16b
	fadd v10.4s, v29.4s, v16.4s
	fmul v3.4s, v25.4s, v5.4s
	fcmgt v25.4s, v20.4s, v25.4s
	fmla v31.4s, v2.4s, v29.4s
	fmul v2.4s, v23.4s, v5.4s
	fcmgt v23.4s, v20.4s, v23.4s
	fmla v11.4s, v27.4s, v8.4s
	fmla v12.4s, v27.4s, v9.4s
	mov v8.16b, v4.16b
	mov v9.16b, v17.16b
	fmul v4.4s, v27.4s, v5.4s
	fcmgt v27.4s, v20.4s, v27.4s
	fmul v5.4s, v29.4s, v5.4s
	fmla v8.4s, v29.4s, v31.4s
	bif v2.16b, v28.16b, v23.16b
	fcmge v23.4s, v24.4s, v1.4s
	fmla v9.4s, v29.4s, v10.4s
	fdiv v31.4s, v11.4s, v12.4s
	mov v10.16b, v6.16b
	mov v11.16b, v18.16b
	fmov v12.4s, #1.00000000
	fcmge v24.4s, v26.4s, v1.4s
	fmin v30.4s, v30.4s, v1.4s
	fmla v10.4s, v29.4s, v8.4s
	mov v8.16b, v7.16b
	bit v2.16b, v1.16b, v21.16b
	fmla v11.4s, v29.4s, v9.4s
	mov v9.16b, v19.16b
	mov v12.s[0], v0.s[0]
	bif v3.16b, v30.16b, v25.16b
	fmla v8.4s, v29.4s, v10.4s
	fmov v10.4s, #1.00000000
	fmla v9.4s, v29.4s, v11.4s
	fmov v11.4s, #1.00000000
	fcmgt v29.4s, v20.4s, v29.4s
	mov v12.s[1], v0.s[0]
	bit v3.16b, v1.16b, v22.16b
	mov v10.s[0], v14.s[0]
	fdiv v8.4s, v8.4s, v9.4s
	fmov v9.4s, #1.00000000
	mov v11.s[0], v15.s[0]
	fmin v31.4s, v31.4s, v1.4s
	mov v12.s[2], v0.s[0]
	mov v10.s[1], v14.s[0]
	mov v9.s[0], v13.s[0]
	mov v11.s[1], v15.s[0]
	bif v4.16b, v31.16b, v27.16b
	mov v10.s[2], v14.s[0]
	mov v9.s[1], v13.s[0]
	mov v11.s[2], v15.s[0]
	bit v4.16b, v1.16b, v23.16b
	fmul v3.4s, v10.4s, v3.4s
	mov v9.s[2], v13.s[0]
	fmin v8.4s, v8.4s, v1.4s
	fmul v4.4s, v11.4s, v4.4s
	fmul v2.4s, v9.4s, v2.4s
	bif v5.16b, v8.16b, v29.16b
	stp q2, q3, [x0]
	bit v5.16b, v1.16b, v24.16b
	str s13, [x0, #12]
	str s14, [x0, #28]
	fmul v5.4s, v12.4s, v5.4s
	stp q4, q5, [x0, #32]
	str s15, [x0, #44]
	str s0, [x0, #60]
	mov x0, x10
.LBB12_8:
	cmp x0, x8
	ccmp x0, #0, #4, ne
	b.ne .LBB12_6
	cbz x9, .LBB12_11
	mov x10, xzr
	mov x0, x9
	mov x9, xzr
	b .LBB12_7
.LBB12_11:
	ldr w8, [sp, #76]
	cbz w8, .LBB12_13
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB12_13:
	.cfi_def_cfa wsp, 336
	ldp x20, x19, [sp, #320]
	ldp x22, x21, [sp, #304]
	ldp x28, x23, [sp, #288]
	ldp x29, x30, [sp, #272]
	ldp d9, d8, [sp, #256]
	ldp d11, d10, [sp, #240]
	ldp d13, d12, [sp, #224]
	ldp d15, d14, [sp, #208]
	add sp, sp, #336
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
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
.LBB12_14:
	.cfi_restore_state
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB12_2
.LBB12_15:
	.cfi_def_cfa wsp, 336
	ldp x20, x19, [sp, #320]
	ldp x22, x21, [sp, #304]
	ldp x28, x23, [sp, #288]
	ldp x29, x30, [sp, #272]
	ldp d9, d8, [sp, #256]
	ldp d11, d10, [sp, #240]
	ldp d13, d12, [sp, #224]
	ldp d15, d14, [sp, #208]
	add sp, sp, #336
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
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
	b linear_srgb::simd::srgb_to_linear_premultiply_rgba_slice_tier_scalar
