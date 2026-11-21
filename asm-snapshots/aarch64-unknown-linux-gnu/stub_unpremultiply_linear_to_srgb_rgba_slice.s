.section .text.stub_unpremultiply_linear_to_srgb_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_srgb_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_srgb_rgba_slice,@function
stub_unpremultiply_linear_to_srgb_rgba_slice:
	.cfi_startproc
	sub sp, sp, #336
	.cfi_def_cfa_offset 336
	stp d15, d14, [sp, #224]
	stp d13, d12, [sp, #240]
	stp d11, d10, [sp, #256]
	stp d9, d8, [sp, #272]
	stp x29, x30, [sp, #288]
	str x28, [sp, #304]
	stp x20, x19, [sp, #320]
	add x29, sp, #288
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
	.cfi_remember_state
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB19_9
	cmp w8, #2
	b.ne .LBB19_8
.LBB19_2:
	lsl x8, x1, #2
	ands x8, x8, #0x7fffffffffffffc0
	b.eq .LBB19_5
	mov w10, #47186
	mov w9, #981467136
	mov w11, #25800
	movk w10, #16718, lsl #16
	dup v2.4s, w9
	mov w9, #5570
	dup v1.4s, w10
	movk w11, #16863, lsl #16
	movk w9, #16968, lsl #16
	mov w10, #10701
	fmov v0.4s, #1.00000000
	add x8, x0, x8
	movk w10, #16697, lsl #16
	stp q1, q2, [sp, #128]
	dup v2.4s, w11
	dup v1.4s, w9
	mov w11, #14394
	mov w9, #15682
	movk w11, #15807, lsl #16
	movk w9, #48222, lsl #16
	stp q1, q2, [sp, #96]
	dup v2.4s, w10
	dup v1.4s, w11
	mov w10, #15285
	mov w11, #7182
	movk w10, #16906, lsl #16
	movk w11, #16947, lsl #16
	stp q1, q2, [sp, #64]
	dup v2.4s, w9
	dup v1.4s, w10
	dup v4.4s, w11
	mov w11, #20545
	mov w9, #64401
	movk w11, #15175, lsl #16
	mov w10, #55785
	movk w9, #16655, lsl #16
	stp q2, q1, [sp, #32]
	dup v2.4s, w11
	movk w10, #16006, lsl #16
	dup v1.4s, w9
	dup v3.4s, w10
	mov x9, x0
	stp q2, q4, [sp]
	ldr q4, [sp, #112]
.LBB19_4:
	ldp q21, q22, [x9]
	mov v14.16b, v1.16b
	ldr q7, [sp, #144]
	ldp q6, q5, [sp, #80]
	ldp q2, q16, [sp, #48]
	mov v18.16b, v1.16b
	dup v23.4s, v21.s[3]
	dup v27.4s, v22.s[3]
	mov v19.16b, v3.16b
	mov v10.16b, v5.16b
	ldp q20, q17, [sp, #16]
	mov v12.16b, v6.16b
	fdiv v24.4s, v0.4s, v23.4s
	fcmgt v23.4s, v23.4s, v7.4s
	mov v13.16b, v20.16b
	mov v15.16b, v20.16b
	fdiv v28.4s, v0.4s, v27.4s
	fcmgt v27.4s, v27.4s, v7.4s
	and v23.16b, v23.16b, v24.16b
	fmul v25.4s, v21.4s, v23.4s
	ldp q24, q23, [x9, #32]
	dup v31.4s, v24.s[3]
	dup v9.4s, v23.s[3]
	fcmgt v26.4s, v25.4s, #0.0
	fdiv v29.4s, v0.4s, v31.4s
	fcmgt v31.4s, v31.4s, v7.4s
	and v27.16b, v27.16b, v28.16b
	and v26.16b, v26.16b, v25.16b
	fcmge v25.4s, v25.4s, v0.4s
	fmul v27.4s, v22.4s, v27.4s
	fminnm v26.4s, v26.4s, v0.4s
	fcmgt v28.4s, v27.4s, #0.0
	fsqrt v8.4s, v26.4s
	and v28.16b, v28.16b, v27.16b
	fcmge v27.4s, v27.4s, v0.4s
	and v29.16b, v31.16b, v29.16b
	mov v31.16b, v17.16b
	fminnm v28.4s, v28.4s, v0.4s
	fmul v29.4s, v24.4s, v29.4s
	fdiv v30.4s, v0.4s, v9.4s
	fmla v10.4s, v4.4s, v8.4s
	fadd v11.4s, v8.4s, v2.4s
	fmla v12.4s, v8.4s, v10.4s
	fmla v13.4s, v8.4s, v11.4s
	mov v11.16b, v16.16b
	fsqrt v10.4s, v28.4s
	fmla v11.4s, v8.4s, v12.4s
	fmla v14.4s, v8.4s, v13.4s
	mov v12.16b, v3.16b
	fmla v31.4s, v8.4s, v11.4s
	fmla v12.4s, v8.4s, v14.4s
	fcmgt v8.4s, v9.4s, v7.4s
	fcmgt v9.4s, v29.4s, #0.0
	mov v14.16b, v6.16b
	mov v7.16b, v17.16b
	and v30.16b, v8.16b, v30.16b
	fdiv v31.4s, v31.4s, v12.4s
	mov v12.16b, v5.16b
	and v8.16b, v9.16b, v29.16b
	fmul v30.4s, v23.4s, v30.4s
	fmla v12.4s, v4.4s, v10.4s
	fadd v13.4s, v10.4s, v2.4s
	fminnm v8.4s, v8.4s, v0.4s
	fcmgt v9.4s, v30.4s, #0.0
	fmla v14.4s, v10.4s, v12.4s
	fmla v15.4s, v10.4s, v13.4s
	mov v13.16b, v16.16b
	fsqrt v11.4s, v8.4s
	and v9.16b, v9.16b, v30.16b
	fmla v13.4s, v10.4s, v14.4s
	fmla v18.4s, v10.4s, v15.4s
	mov v14.16b, v5.16b
	fminnm v9.4s, v9.4s, v0.4s
	fmla v7.4s, v10.4s, v13.4s
	fmla v19.4s, v10.4s, v18.4s
	mov v10.16b, v20.16b
	mov v13.16b, v1.16b
	mov v18.16b, v6.16b
	fsqrt v12.4s, v9.4s
	fadd v15.4s, v11.4s, v2.4s
	fmla v14.4s, v4.4s, v11.4s
	fmla v10.4s, v11.4s, v15.4s
	fmla v18.4s, v11.4s, v14.4s
	mov v15.16b, v3.16b
	mov v14.16b, v17.16b
	fdiv v7.4s, v7.4s, v19.4s
	mov v19.16b, v16.16b
	fmla v13.4s, v11.4s, v10.4s
	fmla v5.4s, v4.4s, v12.4s
	fadd v10.4s, v12.4s, v2.4s
	fmla v19.4s, v11.4s, v18.4s
	ldr q2, [sp]
	fmla v15.4s, v11.4s, v13.4s
	mov v13.16b, v1.16b
	fmla v6.4s, v12.4s, v5.4s
	fmla v20.4s, v12.4s, v10.4s
	fmla v14.4s, v11.4s, v19.4s
	mov v11.16b, v3.16b
	fmla v16.4s, v12.4s, v6.4s
	fmla v13.4s, v12.4s, v20.4s
	fdiv v18.4s, v14.4s, v15.4s
	fmla v17.4s, v12.4s, v16.4s
	fmla v11.4s, v12.4s, v13.4s
	ldr q13, [sp, #128]
	fmul v10.4s, v26.4s, v13.4s
	fcmge v26.4s, v2.4s, v26.4s
	fmul v12.4s, v8.4s, v13.4s
	fcmge v8.4s, v2.4s, v8.4s
	bsl v26.16b, v10.16b, v31.16b
	fdiv v19.4s, v17.4s, v11.4s
	fmul v11.4s, v28.4s, v13.4s
	fcmge v28.4s, v2.4s, v28.4s
	fmul v13.4s, v9.4s, v13.4s
	fcmge v9.4s, v2.4s, v9.4s
	bit v18.16b, v12.16b, v8.16b
	bsl v25.16b, v0.16b, v26.16b
	bit v7.16b, v11.16b, v28.16b
	fcmge v28.4s, v29.4s, v0.4s
	fcmge v29.4s, v30.4s, v0.4s
	mov v25.s[3], v21.s[3]
	bit v7.16b, v0.16b, v27.16b
	bit v18.16b, v0.16b, v28.16b
	mov v7.s[3], v22.s[3]
	mov v18.s[3], v24.s[3]
	bit v19.16b, v13.16b, v9.16b
	stp q25, q7, [x9]
	bit v19.16b, v0.16b, v29.16b
	mov v19.s[3], v23.s[3]
	stp q18, q19, [x9, #32]
	add x9, x9, #64
	cmp x9, x8
	b.ne .LBB19_4
.LBB19_5:
	ands x8, x1, #0xc
	b.eq .LBB19_7
	lsl x19, x8, #2
	mov w8, #64
	and x9, x1, #0x1ffffffffffffff0
	add x20, x0, x9, lsl #2
	mov w1, wzr
	sub x2, x8, x19
	sub x8, x29, #128
	add x0, x8, x19
	bl memset
	sub x0, x29, #128
	mov x1, x20
	mov x2, x19
	bl memcpy
	ldp q0, q1, [x29, #-128]
	fmov v2.4s, #1.00000000
	mov w8, #981467136
	mov w9, #25800
	sub x1, x29, #128
	dup v17.4s, w8
	mov w8, #5570
	movk w9, #16863, lsl #16
	dup v3.4s, v0.s[3]
	dup v7.4s, v1.s[3]
	movk w8, #16968, lsl #16
	dup v18.4s, w9
	mov w9, #10701
	mov x0, x20
	movk w9, #16697, lsl #16
	mov x2, x19
	fdiv v4.4s, v2.4s, v3.4s
	fcmgt v3.4s, v3.4s, v17.4s
	dup v21.4s, w9
	mov w9, #14394
	movk w9, #15807, lsl #16
	mov v31.16b, v21.16b
	dup v27.4s, w9
	mov w9, #55785
	movk w9, #16006, lsl #16
	fdiv v16.4s, v2.4s, v7.4s
	fcmgt v7.4s, v7.4s, v17.4s
	and v3.16b, v3.16b, v4.16b
	fmul v5.4s, v0.4s, v3.4s
	ldp q4, q3, [x29, #-96]
	dup v23.4s, v4.s[3]
	dup v24.4s, v3.s[3]
	fcmgt v6.4s, v5.4s, #0.0
	fdiv v20.4s, v2.4s, v23.4s
	fcmgt v8.4s, v23.4s, v17.4s
	fcmgt v17.4s, v24.4s, v17.4s
	and v7.16b, v7.16b, v16.16b
	dup v16.4s, w8
	mov w8, #15285
	and v6.16b, v6.16b, v5.16b
	movk w8, #16906, lsl #16
	fcmge v5.4s, v5.4s, v2.4s
	dup v25.4s, w8
	mov w8, #7182
	fmul v7.4s, v1.4s, v7.4s
	mov v28.16b, v16.16b
	movk w8, #16947, lsl #16
	fminnm v6.4s, v6.4s, v2.4s
	fcmgt v26.4s, v7.4s, #0.0
	fsqrt v19.4s, v6.4s
	and v29.16b, v26.16b, v7.16b
	dup v26.4s, w8
	mov w8, #64401
	movk w8, #16655, lsl #16
	and v20.16b, v8.16b, v20.16b
	fcmge v7.4s, v7.4s, v2.4s
	mov v9.16b, v26.16b
	fminnm v23.4s, v29.4s, v2.4s
	mov v12.16b, v26.16b
	fmul v20.4s, v4.4s, v20.4s
	fdiv v22.4s, v2.4s, v24.4s
	fmla v28.4s, v18.4s, v19.4s
	fadd v30.4s, v19.4s, v25.4s
	fmla v31.4s, v19.4s, v28.4s
	dup v28.4s, w8
	fmla v9.4s, v19.4s, v30.4s
	mov v30.16b, v27.16b
	mov w8, #15682
	movk w8, #48222, lsl #16
	dup v29.4s, w8
	fsqrt v8.4s, v23.4s
	mov v13.16b, v28.16b
	fmla v30.4s, v19.4s, v31.4s
	mov v31.16b, v28.16b
	mov w8, #47186
	movk w8, #16718, lsl #16
	and v17.16b, v17.16b, v22.16b
	mov v10.16b, v29.16b
	mov v14.16b, v29.16b
	fmla v31.4s, v19.4s, v9.4s
	dup v9.4s, w9
	mov w9, #20545
	movk w9, #15175, lsl #16
	fmul v17.4s, v3.4s, v17.4s
	fmla v10.4s, v19.4s, v30.4s
	mov v11.16b, v9.16b
	mov v15.16b, v9.16b
	fmla v11.4s, v19.4s, v31.4s
	fcmgt v19.4s, v20.4s, #0.0
	fcmgt v22.4s, v17.4s, #0.0
	mov v31.16b, v16.16b
	and v19.16b, v19.16b, v20.16b
	and v22.16b, v22.16b, v17.16b
	fmla v31.4s, v18.4s, v8.4s
	fdiv v24.4s, v10.4s, v11.4s
	mov v11.16b, v21.16b
	fadd v10.4s, v8.4s, v25.4s
	fcmge v17.4s, v17.4s, v2.4s
	fminnm v19.4s, v19.4s, v2.4s
	fminnm v22.4s, v22.4s, v2.4s
	fmla v11.4s, v8.4s, v31.4s
	fmla v12.4s, v8.4s, v10.4s
	mov v10.16b, v27.16b
	fmla v10.4s, v8.4s, v11.4s
	mov v11.16b, v16.16b
	fmla v13.4s, v8.4s, v12.4s
	fsqrt v30.4s, v19.4s
	fmla v14.4s, v8.4s, v10.4s
	mov v10.16b, v26.16b
	fmla v15.4s, v8.4s, v13.4s
	mov v8.16b, v21.16b
	mov v13.16b, v28.16b
	fsqrt v31.4s, v22.4s
	fmla v11.4s, v18.4s, v30.4s
	fadd v12.4s, v30.4s, v25.4s
	fmla v8.4s, v30.4s, v11.4s
	fmla v10.4s, v30.4s, v12.4s
	mov v12.16b, v27.16b
	fdiv v11.4s, v14.4s, v15.4s
	fmla v12.4s, v30.4s, v8.4s
	fmla v13.4s, v30.4s, v10.4s
	mov v8.16b, v9.16b
	fmla v16.4s, v18.4s, v31.4s
	fadd v18.4s, v31.4s, v25.4s
	mov v25.16b, v29.16b
	fmla v25.4s, v30.4s, v12.4s
	fmla v8.4s, v30.4s, v13.4s
	fmla v21.4s, v31.4s, v16.4s
	fmla v26.4s, v31.4s, v18.4s
	fdiv v16.4s, v25.4s, v8.4s
	dup v25.4s, w9
	fmla v27.4s, v31.4s, v21.4s
	fmla v28.4s, v31.4s, v26.4s
	dup v21.4s, w8
	fmul v26.4s, v6.4s, v21.4s
	fcmge v6.4s, v25.4s, v6.4s
	fmla v29.4s, v31.4s, v27.4s
	fmla v9.4s, v31.4s, v28.4s
	fmul v28.4s, v19.4s, v21.4s
	fcmge v19.4s, v25.4s, v19.4s
	fmul v27.4s, v23.4s, v21.4s
	fcmge v23.4s, v25.4s, v23.4s
	fmul v21.4s, v22.4s, v21.4s
	fcmge v22.4s, v25.4s, v22.4s
	bsl v6.16b, v26.16b, v24.16b
	fdiv v18.4s, v29.4s, v9.4s
	bsl v23.16b, v27.16b, v11.16b
	bsl v5.16b, v2.16b, v6.16b
	mov v6.16b, v7.16b
	bit v16.16b, v28.16b, v19.16b
	fcmge v19.4s, v20.4s, v2.4s
	bsl v6.16b, v2.16b, v23.16b
	mov v5.s[3], v0.s[3]
	mov v7.16b, v19.16b
	mov v6.s[3], v1.s[3]
	bsl v7.16b, v2.16b, v16.16b
	bit v18.16b, v21.16b, v22.16b
	stp q5, q6, [x29, #-128]
	mov v7.s[3], v4.s[3]
	bif v2.16b, v18.16b, v17.16b
	mov v2.s[3], v3.s[3]
	stp q7, q2, [x29, #-96]
	bl memcpy
.LBB19_7:
	.cfi_def_cfa wsp, 336
	ldp x20, x19, [sp, #320]
	ldr x28, [sp, #304]
	ldp x29, x30, [sp, #288]
	ldp d9, d8, [sp, #272]
	ldp d11, d10, [sp, #256]
	ldp d13, d12, [sp, #240]
	ldp d15, d14, [sp, #224]
	add sp, sp, #336
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
.LBB19_8:
	.cfi_restore_state
	mov x20, x1
	mov x19, x0
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB19_2
.LBB19_9:
	.cfi_def_cfa wsp, 336
	ldp x20, x19, [sp, #320]
	ldr x28, [sp, #304]
	ldp x29, x30, [sp, #288]
	ldp d9, d8, [sp, #272]
	ldp d11, d10, [sp, #256]
	ldp d13, d12, [sp, #240]
	ldp d15, d14, [sp, #224]
	add sp, sp, #336
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
	b linear_srgb::simd::unpremultiply_linear_to_srgb_rgba_slice_tier_scalar
