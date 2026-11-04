.section .text.stub_linear_to_srgb_rgba_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_rgba_slice
	.p2align	2
.type	stub_linear_to_srgb_rgba_slice,@function
stub_linear_to_srgb_rgba_slice:
	.cfi_startproc
	sub sp, sp, #256
	.cfi_def_cfa_offset 256
	str d14, [sp, #144]
	stp d13, d12, [sp, #152]
	stp d11, d10, [sp, #168]
	stp d9, d8, [sp, #184]
	stp x29, x30, [sp, #200]
	str x23, [sp, #216]
	stp x22, x21, [sp, #224]
	stp x20, x19, [sp, #240]
	add x29, sp, #200
	.cfi_def_cfa w29, 56
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w30, -48
	.cfi_offset w29, -56
	.cfi_offset b8, -64
	.cfi_offset b9, -72
	.cfi_offset b10, -80
	.cfi_offset b11, -88
	.cfi_offset b12, -96
	.cfi_offset b13, -104
	.cfi_offset b14, -112
	.cfi_remember_state
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB9_15
	cmp w8, #2
	b.ne .LBB9_14
.LBB9_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x21, x1, #0xc
	lsr x22, x1, #4
	add x19, x0, x8, lsl #2
	lsl x20, x21, #2
	b.eq .LBB9_4
	mov w8, #64
	mov x23, x0
	mov w1, wzr
	sub x2, x8, x20
	add x8, sp, #80
	add x0, x8, x20
	bl memset
	add x0, sp, #80
	mov x1, x19
	mov x2, x20
	bl memcpy
	ldp q0, q1, [sp, #80]
	mov x0, x23
	mov w9, #1
	stp q0, q1, [sp, #16]
	ldp q0, q2, [sp, #112]
	stp q0, q2, [sp, #48]
	b .LBB9_5
.LBB9_4:
	mov w9, wzr
.LBB9_5:
	mov w8, #47186
	mov w10, #25800
	movi v0.2d, #0000000000000000
	movk w8, #16718, lsl #16
	movk w10, #16863, lsl #16
	fmov v1.4s, #1.00000000
	dup v2.4s, w8
	mov w8, #5570
	dup v3.4s, w10
	movk w8, #16968, lsl #16
	mov w10, #14394
	cmp x21, #0
	dup v4.4s, w8
	mov w8, #10701
	movk w10, #15807, lsl #16
	movk w8, #16697, lsl #16
	dup v6.4s, w10
	mov w10, #7182
	dup v5.4s, w8
	mov w8, #15682
	movk w10, #16947, lsl #16
	movk w8, #48222, lsl #16
	dup v17.4s, w10
	mov w10, #20545
	dup v7.4s, w8
	mov w8, #15285
	movk w10, #15175, lsl #16
	movk w8, #16906, lsl #16
	dup v20.4s, w10
	add x10, sp, #12
	dup v16.4s, w8
	mov w8, #64401
	add x1, x10, #4
	movk w8, #16655, lsl #16
	str w9, [sp, #12]
	csel x9, xzr, x1, eq
	dup v18.4s, w8
	mov w8, #55785
	movk w8, #16006, lsl #16
	dup v19.4s, w8
	add x8, x0, x22, lsl #6
	b .LBB9_8
.LBB9_6:
	add x10, x0, #64
.LBB9_7:
	ldr q21, [x0]
	mov v26.16b, v4.16b
	mov v28.16b, v5.16b
	mov v29.16b, v17.16b
	mov v31.16b, v4.16b
	mov v10.16b, v5.16b
	fmax v22.4s, v21.4s, v0.4s
	mov v11.16b, v17.16b
	mov v12.16b, v18.16b
	mov v13.16b, v7.16b
	mov v14.16b, v19.16b
	fcmge v21.4s, v21.4s, v1.4s
	fmin v23.4s, v22.4s, v1.4s
	ldr q22, [x0, #16]
	fmax v24.4s, v22.4s, v0.4s
	fcmge v22.4s, v22.4s, v1.4s
	fsqrt v25.4s, v23.4s
	fmin v24.4s, v24.4s, v1.4s
	fsqrt v30.4s, v24.4s
	fmla v26.4s, v3.4s, v25.4s
	fadd v27.4s, v25.4s, v16.4s
	fmla v28.4s, v25.4s, v26.4s
	fmla v29.4s, v25.4s, v27.4s
	mov v26.16b, v6.16b
	mov v27.16b, v18.16b
	fmla v26.4s, v25.4s, v28.4s
	fmla v27.4s, v25.4s, v29.4s
	mov v28.16b, v7.16b
	mov v29.16b, v19.16b
	fmla v31.4s, v3.4s, v30.4s
	fadd v9.4s, v30.4s, v16.4s
	fmla v28.4s, v25.4s, v26.4s
	fmla v29.4s, v25.4s, v27.4s
	ldr q25, [x0, #32]
	fmla v10.4s, v30.4s, v31.4s
	fmla v11.4s, v30.4s, v9.4s
	mov v9.16b, v6.16b
	fmax v26.4s, v25.4s, v0.4s
	fcmge v25.4s, v25.4s, v1.4s
	fdiv v28.4s, v28.4s, v29.4s
	fmla v9.4s, v30.4s, v10.4s
	fmla v12.4s, v30.4s, v11.4s
	mov v10.16b, v4.16b
	fmin v27.4s, v26.4s, v1.4s
	ldr q26, [x0, #48]
	fmax v8.4s, v26.4s, v0.4s
	fcmge v26.4s, v26.4s, v1.4s
	fmla v13.4s, v30.4s, v9.4s
	fmla v14.4s, v30.4s, v12.4s
	mov v30.16b, v5.16b
	mov v9.16b, v17.16b
	mov v12.16b, v18.16b
	fmin v8.4s, v8.4s, v1.4s
	fsqrt v29.4s, v27.4s
	fmin v28.4s, v28.4s, v1.4s
	fsqrt v31.4s, v8.4s
	fmla v10.4s, v3.4s, v29.4s
	fadd v11.4s, v29.4s, v16.4s
	fmla v30.4s, v29.4s, v10.4s
	fmla v9.4s, v29.4s, v11.4s
	mov v11.16b, v6.16b
	fdiv v10.4s, v13.4s, v14.4s
	mov v13.16b, v7.16b
	mov v14.16b, v19.16b
	fmla v11.4s, v29.4s, v30.4s
	fmla v12.4s, v29.4s, v9.4s
	mov v30.16b, v4.16b
	fadd v9.4s, v31.4s, v16.4s
	fmla v30.4s, v3.4s, v31.4s
	fmla v13.4s, v29.4s, v11.4s
	fmla v14.4s, v29.4s, v12.4s
	mov v29.16b, v5.16b
	mov v11.16b, v17.16b
	mov v12.16b, v18.16b
	fmla v29.4s, v31.4s, v30.4s
	fmla v11.4s, v31.4s, v9.4s
	mov v9.16b, v6.16b
	fdiv v30.4s, v13.4s, v14.4s
	fmla v9.4s, v31.4s, v29.4s
	mov v29.16b, v7.16b
	fmin v10.4s, v10.4s, v1.4s
	fmla v12.4s, v31.4s, v11.4s
	mov v11.16b, v19.16b
	fmla v29.4s, v31.4s, v9.4s
	fmul v9.4s, v24.4s, v2.4s
	fcmgt v24.4s, v20.4s, v24.4s
	fmla v11.4s, v31.4s, v12.4s
	fmul v31.4s, v23.4s, v2.4s
	fcmgt v23.4s, v20.4s, v23.4s
	fmul v12.4s, v8.4s, v2.4s
	fcmgt v8.4s, v20.4s, v8.4s
	bsl v24.16b, v9.16b, v10.16b
	fdiv v29.4s, v29.4s, v11.4s
	fmul v11.4s, v27.4s, v2.4s
	fcmgt v27.4s, v20.4s, v27.4s
	fmin v30.4s, v30.4s, v1.4s
	bsl v23.16b, v31.16b, v28.16b
	mov v28.16b, v8.16b
	bsl v22.16b, v1.16b, v24.16b
	mov v24.16b, v26.16b
	ldr s26, [x0, #28]
	bsl v27.16b, v11.16b, v30.16b
	bsl v21.16b, v1.16b, v23.16b
	mov v23.16b, v25.16b
	ldr s25, [x0, #12]
	bsl v23.16b, v1.16b, v27.16b
	stp q21, q22, [x0]
	ldr s21, [x0, #44]
	ldr s22, [x0, #60]
	str s25, [x0, #12]
	str s26, [x0, #28]
	fmin v29.4s, v29.4s, v1.4s
	bsl v28.16b, v12.16b, v29.16b
	bsl v24.16b, v1.16b, v28.16b
	stp q23, q24, [x0, #32]
	str s21, [x0, #44]
	str s22, [x0, #60]
	mov x0, x10
.LBB9_8:
	cmp x0, x8
	ccmp x0, #0, #4, ne
	b.ne .LBB9_6
	cbz x9, .LBB9_11
	mov x10, xzr
	mov x0, x9
	mov x9, xzr
	b .LBB9_7
.LBB9_11:
	ldr w8, [sp, #12]
	cbz w8, .LBB9_13
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB9_13:
	.cfi_def_cfa wsp, 256
	ldp x20, x19, [sp, #240]
	ldr x23, [sp, #216]
	ldp x22, x21, [sp, #224]
	ldr d14, [sp, #144]
	ldp x29, x30, [sp, #200]
	ldp d9, d8, [sp, #184]
	ldp d11, d10, [sp, #168]
	ldp d13, d12, [sp, #152]
	add sp, sp, #256
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
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
.LBB9_14:
	.cfi_restore_state
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB9_2
.LBB9_15:
	.cfi_def_cfa wsp, 256
	ldp x20, x19, [sp, #240]
	ldr x23, [sp, #216]
	ldp x22, x21, [sp, #224]
	ldr d14, [sp, #144]
	ldp x29, x30, [sp, #200]
	ldp d9, d8, [sp, #184]
	ldp d11, d10, [sp, #168]
	ldp d13, d12, [sp, #152]
	add sp, sp, #256
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	.cfi_restore b13
	.cfi_restore b14
	b linear_srgb::simd::linear_to_srgb_rgba_slice_tier_scalar
