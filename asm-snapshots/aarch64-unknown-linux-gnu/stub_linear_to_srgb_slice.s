.section .text.stub_linear_to_srgb_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_slice
	.p2align	2
.type	stub_linear_to_srgb_slice,@function
stub_linear_to_srgb_slice:
	.cfi_startproc
	sub sp, sp, #208
	.cfi_def_cfa_offset 208
	stp d13, d12, [sp, #128]
	stp d11, d10, [sp, #144]
	stp d9, d8, [sp, #160]
	stp x29, x30, [sp, #176]
	stp x20, x19, [sp, #192]
	add x29, sp, #176
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
	b.eq .LBB13_9
	cmp w8, #2
	b.ne .LBB13_8
.LBB13_2:
	lsl x8, x1, #2
	ands x9, x8, #0x7fffffffffffffc0
	and x8, x1, #0xf
	b.eq .LBB13_5
	mov w10, #47186
	mov w11, #25800
	mov w12, #5570
	movk w10, #16718, lsl #16
	movk w11, #16863, lsl #16
	movk w12, #16968, lsl #16
	dup v1.4s, w10
	dup v2.4s, w11
	mov w10, #10701
	mov w11, #14394
	movk w10, #16697, lsl #16
	dup v3.4s, w12
	movk w11, #15807, lsl #16
	dup v4.4s, w10
	mov w10, #15682
	dup v5.4s, w11
	mov w11, #15285
	mov w12, #7182
	movk w10, #48222, lsl #16
	movk w11, #16906, lsl #16
	movk w12, #16947, lsl #16
	dup v6.4s, w10
	dup v7.4s, w11
	dup v16.4s, w12
	mov w10, #64401
	mov w11, #55785
	mov w12, #20545
	fmov v0.4s, #1.00000000
	movk w10, #16655, lsl #16
	movk w11, #16006, lsl #16
	movk w12, #15175, lsl #16
	dup v17.4s, w10
	dup v18.4s, w11
	dup v19.4s, w12
	add x9, x0, x9
	mov x10, x0
.LBB13_4:
	ldp q21, q20, [x10]
	mov v25.16b, v3.16b
	mov v27.16b, v4.16b
	mov v28.16b, v16.16b
	mov v31.16b, v3.16b
	mov v9.16b, v4.16b
	mov v10.16b, v16.16b
	mov v11.16b, v17.16b
	fcmgt v22.4s, v21.4s, #0.0
	fcmgt v23.4s, v20.4s, #0.0
	mov v12.16b, v6.16b
	mov v13.16b, v18.16b
	and v22.16b, v22.16b, v21.16b
	and v23.16b, v23.16b, v20.16b
	fcmge v21.4s, v21.4s, v0.4s
	fcmge v20.4s, v20.4s, v0.4s
	fminnm v22.4s, v22.4s, v0.4s
	fminnm v23.4s, v23.4s, v0.4s
	fsqrt v24.4s, v22.4s
	fsqrt v29.4s, v23.4s
	fmla v25.4s, v2.4s, v24.4s
	fadd v26.4s, v24.4s, v7.4s
	fmla v27.4s, v24.4s, v25.4s
	fmla v28.4s, v24.4s, v26.4s
	mov v25.16b, v5.16b
	mov v26.16b, v17.16b
	fmla v25.4s, v24.4s, v27.4s
	fmla v26.4s, v24.4s, v28.4s
	mov v27.16b, v6.16b
	mov v28.16b, v18.16b
	fmla v31.4s, v2.4s, v29.4s
	fadd v8.4s, v29.4s, v7.4s
	fmla v27.4s, v24.4s, v25.4s
	fmla v28.4s, v24.4s, v26.4s
	ldp q25, q24, [x10, #32]
	fmla v9.4s, v29.4s, v31.4s
	fmla v10.4s, v29.4s, v8.4s
	mov v8.16b, v5.16b
	fcmgt v26.4s, v25.4s, #0.0
	fdiv v27.4s, v27.4s, v28.4s
	fcmgt v28.4s, v24.4s, #0.0
	fmla v8.4s, v29.4s, v9.4s
	mov v9.16b, v3.16b
	fmla v11.4s, v29.4s, v10.4s
	and v26.16b, v26.16b, v25.16b
	and v28.16b, v28.16b, v24.16b
	fcmge v25.4s, v25.4s, v0.4s
	fmla v12.4s, v29.4s, v8.4s
	mov v8.16b, v16.16b
	fcmge v24.4s, v24.4s, v0.4s
	fmla v13.4s, v29.4s, v11.4s
	mov v29.16b, v4.16b
	mov v11.16b, v17.16b
	fminnm v26.4s, v26.4s, v0.4s
	fminnm v28.4s, v28.4s, v0.4s
	fsqrt v30.4s, v26.4s
	fsqrt v31.4s, v28.4s
	fmla v9.4s, v2.4s, v30.4s
	fadd v10.4s, v30.4s, v7.4s
	fmla v29.4s, v30.4s, v9.4s
	fmla v8.4s, v30.4s, v10.4s
	mov v10.16b, v5.16b
	fdiv v9.4s, v12.4s, v13.4s
	mov v12.16b, v6.16b
	mov v13.16b, v18.16b
	fmla v10.4s, v30.4s, v29.4s
	fmla v11.4s, v30.4s, v8.4s
	mov v29.16b, v3.16b
	fadd v8.4s, v31.4s, v7.4s
	fmla v29.4s, v2.4s, v31.4s
	fmla v12.4s, v30.4s, v10.4s
	fmla v13.4s, v30.4s, v11.4s
	mov v30.16b, v4.16b
	mov v10.16b, v16.16b
	mov v11.16b, v17.16b
	fmla v30.4s, v31.4s, v29.4s
	fmla v10.4s, v31.4s, v8.4s
	mov v8.16b, v5.16b
	fdiv v29.4s, v12.4s, v13.4s
	fmla v8.4s, v31.4s, v30.4s
	mov v30.16b, v6.16b
	fmla v11.4s, v31.4s, v10.4s
	mov v10.16b, v18.16b
	fmla v30.4s, v31.4s, v8.4s
	fmul v8.4s, v23.4s, v1.4s
	fcmge v23.4s, v19.4s, v23.4s
	fmla v10.4s, v31.4s, v11.4s
	fmul v31.4s, v22.4s, v1.4s
	fcmge v22.4s, v19.4s, v22.4s
	fmul v11.4s, v28.4s, v1.4s
	fcmge v28.4s, v19.4s, v28.4s
	bsl v23.16b, v8.16b, v9.16b
	fdiv v30.4s, v30.4s, v10.4s
	fmul v10.4s, v26.4s, v1.4s
	fcmge v26.4s, v19.4s, v26.4s
	bsl v22.16b, v31.16b, v27.16b
	mov v27.16b, v28.16b
	bsl v20.16b, v0.16b, v23.16b
	mov v23.16b, v24.16b
	bsl v26.16b, v10.16b, v29.16b
	bsl v21.16b, v0.16b, v22.16b
	mov v22.16b, v25.16b
	bsl v22.16b, v0.16b, v26.16b
	stp q21, q20, [x10]
	bsl v27.16b, v11.16b, v30.16b
	bsl v23.16b, v0.16b, v27.16b
	stp q22, q23, [x10, #32]
	add x10, x10, #64
	cmp x10, x9
	b.ne .LBB13_4
.LBB13_5:
	cbz x8, .LBB13_7
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
	ldp q2, q0, [sp]
	fmov v1.4s, #1.00000000
	mov w9, #5570
	mov w8, #25800
	add x1, sp, #64
	movk w9, #16968, lsl #16
	movk w8, #16863, lsl #16
	mov x0, x20
	fcmgt v3.4s, v2.4s, #0.0
	dup v4.4s, w9
	mov w9, #15285
	movk w9, #16906, lsl #16
	dup v16.4s, w8
	mov w8, #10701
	dup v17.4s, w9
	mov w9, #7182
	movk w8, #16697, lsl #16
	movk w9, #16947, lsl #16
	mov v21.16b, v4.16b
	dup v6.4s, w8
	and v3.16b, v3.16b, v2.16b
	dup v7.4s, w9
	mov w8, #14394
	mov w9, #64401
	movk w8, #15807, lsl #16
	fcmgt v18.4s, v0.4s, #0.0
	movk w9, #16655, lsl #16
	mov v23.16b, v6.16b
	dup v19.4s, w8
	fminnm v3.4s, v3.4s, v1.4s
	mov v24.16b, v7.16b
	dup v20.4s, w9
	mov w8, #15682
	mov w9, #55785
	mov v29.16b, v4.16b
	movk w8, #48222, lsl #16
	movk w9, #16006, lsl #16
	and v18.16b, v18.16b, v0.16b
	mov v25.16b, v20.16b
	dup v27.4s, w9
	mov v31.16b, v6.16b
	fsqrt v5.4s, v3.4s
	mov v8.16b, v7.16b
	mov v9.16b, v20.16b
	fminnm v18.4s, v18.4s, v1.4s
	mov w9, #20545
	fcmge v2.4s, v2.4s, v1.4s
	mov v11.16b, v27.16b
	movk w9, #15175, lsl #16
	fcmge v0.4s, v0.4s, v1.4s
	mov x2, x19
	fsqrt v26.4s, v18.4s
	fmla v21.4s, v16.4s, v5.4s
	fadd v22.4s, v5.4s, v17.4s
	fmla v23.4s, v5.4s, v21.4s
	fmla v24.4s, v5.4s, v22.4s
	mov v21.16b, v19.16b
	dup v22.4s, w8
	mov w8, #47186
	movk w8, #16718, lsl #16
	fmla v21.4s, v5.4s, v23.4s
	fmla v25.4s, v5.4s, v24.4s
	mov v23.16b, v22.16b
	mov v24.16b, v27.16b
	fmla v29.4s, v16.4s, v26.4s
	fadd v30.4s, v26.4s, v17.4s
	mov v10.16b, v22.16b
	fmla v23.4s, v5.4s, v21.4s
	fmla v24.4s, v5.4s, v25.4s
	ldp q21, q5, [sp, #32]
	fmla v31.4s, v26.4s, v29.4s
	fmla v8.4s, v26.4s, v30.4s
	mov v30.16b, v19.16b
	fcmgt v25.4s, v21.4s, #0.0
	fdiv v23.4s, v23.4s, v24.4s
	fmla v30.4s, v26.4s, v31.4s
	mov v31.16b, v4.16b
	fmla v9.4s, v26.4s, v8.4s
	and v24.16b, v25.16b, v21.16b
	fcmgt v25.4s, v5.4s, #0.0
	fmla v10.4s, v26.4s, v30.4s
	mov v30.16b, v7.16b
	fmla v11.4s, v26.4s, v9.4s
	mov v26.16b, v6.16b
	mov v9.16b, v20.16b
	fminnm v24.4s, v24.4s, v1.4s
	and v25.16b, v25.16b, v5.16b
	fcmge v5.4s, v5.4s, v1.4s
	fminnm v25.4s, v25.4s, v1.4s
	fsqrt v28.4s, v24.4s
	fsqrt v29.4s, v25.4s
	fmla v31.4s, v16.4s, v28.4s
	fadd v8.4s, v28.4s, v17.4s
	fmla v26.4s, v28.4s, v31.4s
	fmla v30.4s, v28.4s, v8.4s
	mov v8.16b, v19.16b
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
	fmla v19.4s, v29.4s, v6.4s
	fmla v20.4s, v29.4s, v7.4s
	dup v7.4s, w8
	fmul v17.4s, v3.4s, v7.4s
	fcmge v3.4s, v16.4s, v3.4s
	fmla v22.4s, v29.4s, v19.4s
	fmla v27.4s, v29.4s, v20.4s
	fmul v19.4s, v18.4s, v7.4s
	fcmge v18.4s, v16.4s, v18.4s
	fmul v20.4s, v24.4s, v7.4s
	fmul v7.4s, v25.4s, v7.4s
	bsl v3.16b, v17.16b, v23.16b
	fdiv v6.4s, v22.4s, v27.4s
	fcmge v22.4s, v16.4s, v24.4s
	fcmge v16.4s, v16.4s, v25.4s
	mov v17.16b, v18.16b
	fcmge v18.4s, v21.4s, v1.4s
	bsl v2.16b, v1.16b, v3.16b
	bsl v17.16b, v19.16b, v31.16b
	bit v4.16b, v20.16b, v22.16b
	mov v3.16b, v18.16b
	bsl v0.16b, v1.16b, v17.16b
	bsl v3.16b, v1.16b, v4.16b
	stp q2, q0, [sp, #64]
	bit v6.16b, v7.16b, v16.16b
	bif v1.16b, v6.16b, v5.16b
	stp q3, q1, [sp, #96]
	bl memcpy
.LBB13_7:
	.cfi_def_cfa wsp, 208
	ldp x20, x19, [sp, #192]
	ldp x29, x30, [sp, #176]
	ldp d9, d8, [sp, #160]
	ldp d11, d10, [sp, #144]
	ldp d13, d12, [sp, #128]
	add sp, sp, #208
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
.LBB13_8:
	.cfi_restore_state
	mov x20, x1
	mov x19, x0
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB13_2
.LBB13_9:
	.cfi_def_cfa wsp, 208
	ldp x20, x19, [sp, #192]
	ldp x29, x30, [sp, #176]
	ldp d9, d8, [sp, #160]
	ldp d11, d10, [sp, #144]
	ldp d13, d12, [sp, #128]
	add sp, sp, #208
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
	b linear_srgb::simd::linear_to_srgb_slice_tier_scalar_t
