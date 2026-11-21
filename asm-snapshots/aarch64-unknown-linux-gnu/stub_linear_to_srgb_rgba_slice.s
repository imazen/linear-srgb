.section .text.stub_linear_to_srgb_rgba_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_rgba_slice
	.p2align	2
.type	stub_linear_to_srgb_rgba_slice,@function
stub_linear_to_srgb_rgba_slice:
	.cfi_startproc
	sub sp, sp, #144
	.cfi_def_cfa_offset 144
	stp d13, d12, [sp, #64]
	stp d11, d10, [sp, #80]
	stp d9, d8, [sp, #96]
	stp x29, x30, [sp, #112]
	stp x20, x19, [sp, #128]
	add x29, sp, #112
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
	.cfi_remember_state
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB12_9
	cmp w8, #2
	b.ne .LBB12_8
.LBB12_2:
	lsl x8, x1, #2
	ands x8, x8, #0x7fffffffffffffc0
	b.eq .LBB12_5
	mov w9, #47186
	mov w10, #25800
	mov w11, #5570
	movk w9, #16718, lsl #16
	movk w10, #16863, lsl #16
	movk w11, #16968, lsl #16
	dup v1.4s, w9
	dup v2.4s, w10
	mov w9, #10701
	mov w10, #14394
	movk w9, #16697, lsl #16
	dup v3.4s, w11
	movk w10, #15807, lsl #16
	dup v4.4s, w9
	mov w9, #15682
	dup v5.4s, w10
	mov w10, #15285
	mov w11, #7182
	movk w9, #48222, lsl #16
	movk w10, #16906, lsl #16
	movk w11, #16947, lsl #16
	dup v6.4s, w9
	dup v7.4s, w10
	dup v16.4s, w11
	mov w9, #64401
	mov w10, #55785
	mov w11, #20545
	fmov v0.4s, #1.00000000
	movk w9, #16655, lsl #16
	movk w10, #16006, lsl #16
	movk w11, #15175, lsl #16
	dup v17.4s, w9
	dup v18.4s, w10
	dup v19.4s, w11
	add x8, x0, x8
	mov x9, x0
.LBB12_4:
	ldp q20, q21, [x9]
	mov v26.16b, v3.16b
	mov v28.16b, v4.16b
	mov v29.16b, v16.16b
	mov v31.16b, v3.16b
	mov v9.16b, v4.16b
	mov v10.16b, v16.16b
	mov v11.16b, v17.16b
	fcmgt v22.4s, v20.4s, #0.0
	fcmgt v24.4s, v21.4s, #0.0
	mov v12.16b, v6.16b
	mov v13.16b, v18.16b
	and v22.16b, v22.16b, v20.16b
	and v24.16b, v24.16b, v21.16b
	fminnm v22.4s, v22.4s, v0.4s
	fminnm v25.4s, v24.4s, v0.4s
	mov v24.16b, v5.16b
	fsqrt v23.4s, v22.4s
	fmla v26.4s, v2.4s, v23.4s
	fadd v27.4s, v23.4s, v7.4s
	fmla v28.4s, v23.4s, v26.4s
	fmla v29.4s, v23.4s, v27.4s
	mov v26.16b, v17.16b
	fsqrt v27.4s, v25.4s
	fmla v24.4s, v23.4s, v28.4s
	fmla v26.4s, v23.4s, v29.4s
	mov v28.16b, v6.16b
	mov v29.16b, v18.16b
	fmla v28.4s, v23.4s, v24.4s
	fmla v29.4s, v23.4s, v26.4s
	ldp q24, q23, [x9, #32]
	fcmgt v26.4s, v24.4s, #0.0
	fdiv v28.4s, v28.4s, v29.4s
	fcmgt v29.4s, v23.4s, #0.0
	fmla v31.4s, v2.4s, v27.4s
	fadd v8.4s, v27.4s, v7.4s
	and v26.16b, v26.16b, v24.16b
	and v29.16b, v29.16b, v23.16b
	fmla v9.4s, v27.4s, v31.4s
	fmla v10.4s, v27.4s, v8.4s
	mov v8.16b, v5.16b
	fminnm v26.4s, v26.4s, v0.4s
	fminnm v29.4s, v29.4s, v0.4s
	fmla v8.4s, v27.4s, v9.4s
	fmla v11.4s, v27.4s, v10.4s
	mov v9.16b, v3.16b
	fsqrt v30.4s, v26.4s
	fmla v12.4s, v27.4s, v8.4s
	fmla v13.4s, v27.4s, v11.4s
	mov v27.16b, v4.16b
	mov v8.16b, v16.16b
	mov v11.16b, v17.16b
	fsqrt v31.4s, v29.4s
	fmla v9.4s, v2.4s, v30.4s
	fadd v10.4s, v30.4s, v7.4s
	fmla v27.4s, v30.4s, v9.4s
	fmla v8.4s, v30.4s, v10.4s
	mov v10.16b, v5.16b
	fdiv v9.4s, v12.4s, v13.4s
	mov v12.16b, v6.16b
	mov v13.16b, v18.16b
	fmla v10.4s, v30.4s, v27.4s
	fmla v11.4s, v30.4s, v8.4s
	mov v27.16b, v3.16b
	fadd v8.4s, v31.4s, v7.4s
	fmla v27.4s, v2.4s, v31.4s
	fmla v12.4s, v30.4s, v10.4s
	fmla v13.4s, v30.4s, v11.4s
	mov v30.16b, v4.16b
	mov v10.16b, v16.16b
	mov v11.16b, v17.16b
	fmla v30.4s, v31.4s, v27.4s
	fmla v10.4s, v31.4s, v8.4s
	mov v8.16b, v5.16b
	fdiv v27.4s, v12.4s, v13.4s
	fmla v8.4s, v31.4s, v30.4s
	mov v30.16b, v6.16b
	fmla v11.4s, v31.4s, v10.4s
	mov v10.16b, v18.16b
	fmla v30.4s, v31.4s, v8.4s
	fmul v8.4s, v25.4s, v1.4s
	fcmge v25.4s, v19.4s, v25.4s
	fmla v10.4s, v31.4s, v11.4s
	fmul v11.4s, v29.4s, v1.4s
	fcmge v29.4s, v19.4s, v29.4s
	fmul v31.4s, v22.4s, v1.4s
	fcmge v22.4s, v19.4s, v22.4s
	bsl v25.16b, v8.16b, v9.16b
	fdiv v30.4s, v30.4s, v10.4s
	fmul v10.4s, v26.4s, v1.4s
	fcmge v26.4s, v19.4s, v26.4s
	bsl v22.16b, v31.16b, v28.16b
	fcmge v28.4s, v20.4s, v0.4s
	fcmge v31.4s, v21.4s, v0.4s
	bsl v26.16b, v10.16b, v27.16b
	fcmge v27.4s, v24.4s, v0.4s
	bit v22.16b, v0.16b, v28.16b
	bit v25.16b, v0.16b, v31.16b
	bit v26.16b, v0.16b, v27.16b
	mov v22.s[3], v20.s[3]
	mov v25.s[3], v21.s[3]
	mov v26.s[3], v24.s[3]
	bsl v29.16b, v11.16b, v30.16b
	fcmge v30.4s, v23.4s, v0.4s
	stp q22, q25, [x9]
	mov v27.16b, v30.16b
	bsl v27.16b, v0.16b, v29.16b
	mov v27.s[3], v23.s[3]
	stp q26, q27, [x9, #32]
	add x9, x9, #64
	cmp x9, x8
	b.ne .LBB12_4
.LBB12_5:
	ands x8, x1, #0xc
	b.eq .LBB12_7
	lsl x19, x8, #2
	mov w8, #64
	and x9, x1, #0x1ffffffffffffff0
	add x20, x0, x9, lsl #2
	mov w1, wzr
	sub x2, x8, x19
	mov x8, sp
	add x0, x8, x19
	bl memset
	mov x0, sp
	mov x1, x20
	mov x2, x19
	bl memcpy
	ldp q0, q1, [sp]
	fmov v2.4s, #1.00000000
	mov w9, #5570
	mov w8, #25800
	mov x1, sp
	movk w9, #16968, lsl #16
	movk w8, #16863, lsl #16
	mov x0, x20
	fcmgt v3.4s, v0.4s, #0.0
	dup v4.4s, w9
	mov w9, #15285
	movk w9, #16906, lsl #16
	dup v16.4s, w8
	mov w8, #10701
	dup v17.4s, w9
	mov w9, #7182
	movk w8, #16697, lsl #16
	movk w9, #16947, lsl #16
	mov v22.16b, v4.16b
	fcmgt v18.4s, v1.4s, #0.0
	and v3.16b, v3.16b, v0.16b
	dup v6.4s, w8
	dup v7.4s, w9
	mov w8, #14394
	mov w9, #64401
	mov v29.16b, v4.16b
	movk w8, #15807, lsl #16
	movk w9, #16655, lsl #16
	mov x2, x19
	fminnm v3.4s, v3.4s, v2.4s
	mov v24.16b, v6.16b
	mov v25.16b, v7.16b
	and v18.16b, v18.16b, v1.16b
	dup v20.4s, w8
	dup v21.4s, w9
	mov w8, #15682
	mov w9, #55785
	mov v31.16b, v6.16b
	movk w8, #48222, lsl #16
	movk w9, #16006, lsl #16
	mov v8.16b, v7.16b
	fsqrt v5.4s, v3.4s
	fminnm v19.4s, v18.4s, v2.4s
	mov v18.16b, v20.16b
	dup v27.4s, w9
	mov v9.16b, v21.16b
	mov w9, #20545
	movk w9, #15175, lsl #16
	mov v11.16b, v27.16b
	fsqrt v26.4s, v19.4s
	fmla v22.4s, v16.4s, v5.4s
	fadd v23.4s, v5.4s, v17.4s
	fmla v24.4s, v5.4s, v22.4s
	fmla v25.4s, v5.4s, v23.4s
	mov v23.16b, v21.16b
	dup v22.4s, w8
	mov w8, #47186
	movk w8, #16718, lsl #16
	fmla v18.4s, v5.4s, v24.4s
	fmla v23.4s, v5.4s, v25.4s
	mov v24.16b, v22.16b
	mov v25.16b, v27.16b
	fmla v29.4s, v16.4s, v26.4s
	fadd v30.4s, v26.4s, v17.4s
	mov v10.16b, v22.16b
	fmla v24.4s, v5.4s, v18.4s
	fmla v25.4s, v5.4s, v23.4s
	ldp q18, q5, [sp, #32]
	fmla v31.4s, v26.4s, v29.4s
	fmla v8.4s, v26.4s, v30.4s
	mov v30.16b, v20.16b
	fcmgt v23.4s, v18.4s, #0.0
	fdiv v24.4s, v24.4s, v25.4s
	fcmgt v25.4s, v5.4s, #0.0
	fmla v30.4s, v26.4s, v31.4s
	mov v31.16b, v4.16b
	fmla v9.4s, v26.4s, v8.4s
	and v23.16b, v23.16b, v18.16b
	and v25.16b, v25.16b, v5.16b
	fmla v10.4s, v26.4s, v30.4s
	mov v30.16b, v7.16b
	fmla v11.4s, v26.4s, v9.4s
	mov v26.16b, v6.16b
	mov v9.16b, v21.16b
	fminnm v23.4s, v23.4s, v2.4s
	fminnm v25.4s, v25.4s, v2.4s
	fsqrt v28.4s, v23.4s
	fsqrt v29.4s, v25.4s
	fmla v31.4s, v16.4s, v28.4s
	fadd v8.4s, v28.4s, v17.4s
	fmla v26.4s, v28.4s, v31.4s
	fmla v30.4s, v28.4s, v8.4s
	mov v8.16b, v20.16b
	fdiv v31.4s, v10.4s, v11.4s
	fmla v8.4s, v28.4s, v26.4s
	fmla v9.4s, v28.4s, v30.4s
	mov v26.16b, v27.16b
	fmla v4.4s, v16.4s, v29.4s
	fadd v16.4s, v29.4s, v17.4s
	mov v17.16b, v22.16b
	fmla v17.4s, v28.4s, v8.4s
	fmla v26.4s, v28.4s, v9.4s
	fmla v6.4s, v29.4s, v4.4s
	fmla v7.4s, v29.4s, v16.4s
	dup v16.4s, w9
	fdiv v4.4s, v17.4s, v26.4s
	fmla v20.4s, v29.4s, v6.4s
	fmla v21.4s, v29.4s, v7.4s
	dup v7.4s, w8
	fmul v17.4s, v3.4s, v7.4s
	fcmge v3.4s, v16.4s, v3.4s
	fmla v22.4s, v29.4s, v20.4s
	fmla v27.4s, v29.4s, v21.4s
	fmul v20.4s, v19.4s, v7.4s
	fcmge v19.4s, v16.4s, v19.4s
	fmul v21.4s, v23.4s, v7.4s
	fmul v7.4s, v25.4s, v7.4s
	bsl v3.16b, v17.16b, v24.16b
	fcmge v17.4s, v0.4s, v2.4s
	fdiv v6.4s, v22.4s, v27.4s
	fcmge v22.4s, v16.4s, v23.4s
	fcmge v16.4s, v16.4s, v25.4s
	bsl v19.16b, v20.16b, v31.16b
	fcmge v20.4s, v1.4s, v2.4s
	bit v3.16b, v2.16b, v17.16b
	bit v4.16b, v21.16b, v22.16b
	fcmge v21.4s, v18.4s, v2.4s
	mov v3.s[3], v0.s[3]
	bit v4.16b, v2.16b, v21.16b
	mov v4.s[3], v18.s[3]
	bit v6.16b, v7.16b, v16.16b
	fcmge v7.4s, v5.4s, v2.4s
	mov v16.16b, v20.16b
	bsl v16.16b, v2.16b, v19.16b
	bif v2.16b, v6.16b, v7.16b
	mov v16.s[3], v1.s[3]
	mov v2.s[3], v5.s[3]
	stp q3, q16, [sp]
	stp q4, q2, [sp, #32]
	bl memcpy
.LBB12_7:
	.cfi_def_cfa wsp, 144
	ldp x20, x19, [sp, #128]
	ldp x29, x30, [sp, #112]
	ldp d9, d8, [sp, #96]
	ldp d11, d10, [sp, #80]
	ldp d13, d12, [sp, #64]
	add sp, sp, #144
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
	ret
.LBB12_8:
	.cfi_restore_state
	mov x20, x1
	mov x19, x0
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB12_2
.LBB12_9:
	.cfi_def_cfa wsp, 144
	ldp x20, x19, [sp, #128]
	ldp x29, x30, [sp, #112]
	ldp d9, d8, [sp, #96]
	ldp d11, d10, [sp, #80]
	ldp d13, d12, [sp, #64]
	add sp, sp, #144
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
	b linear_srgb::simd::linear_to_srgb_rgba_slice_tier_scalar
