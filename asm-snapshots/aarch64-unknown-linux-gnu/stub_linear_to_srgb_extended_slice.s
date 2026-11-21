.section .text.stub_linear_to_srgb_extended_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_extended_slice
	.p2align	2
.type	stub_linear_to_srgb_extended_slice,@function
stub_linear_to_srgb_extended_slice:
	.cfi_startproc
	sub sp, sp, #480
	.cfi_def_cfa_offset 480
	stp d15, d14, [sp, #368]
	stp d13, d12, [sp, #384]
	stp d11, d10, [sp, #400]
	stp d9, d8, [sp, #416]
	stp x29, x30, [sp, #432]
	stp x28, x21, [sp, #448]
	stp x20, x19, [sp, #464]
	add x29, sp, #432
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
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
	b.eq .LBB11_13
	cmp w8, #2
	b.ne .LBB11_12
.LBB11_2:
	and x8, x1, #0x1ffffffffffffff0
	mov x9, x1
	lsr x1, x1, #4
	add x19, x0, x8, lsl #2
	and x20, x9, #0xf
	cbz x1, .LBB11_6
	mov w8, #1115684864
	mov w9, #20545
	dup v0.4s, w8
	mov w8, #59837
	movk w9, #15175, lsl #16
	movk w8, #17200, lsl #16
	dup v1.4s, w9
	mov w9, #60277
	movk w9, #17737, lsl #16
	stur q0, [x29, #-208]
	dup v0.4s, w8
	mov w8, #21186
	movk w8, #17949, lsl #16
	stp q0, q1, [sp, #192]
	dup v1.4s, w9
	dup v0.4s, w8
	mov w8, #44370
	mov w9, #26919
	ldr q3, [sp, #192]
	movk w8, #17885, lsl #16
	movk w9, #17536, lsl #16
	stp q0, q1, [sp, #160]
	dup v1.4s, w8
	dup v0.4s, w9
	mov w8, #30049
	mov w9, #63323
	movk w8, #48797, lsl #16
	movk w9, #17310, lsl #16
	stp q0, q1, [sp, #128]
	dup v1.4s, w8
	mov w8, #17028
	movk w8, #49027, lsl #16
	dup v0.4s, w9
	mov w9, #18007
	dup v2.4s, w8
	mov w8, #60486
	movk w9, #17852, lsl #16
	movk w8, #17793, lsl #16
	stp q0, q1, [sp, #96]
	dup v0.4s, w8
	mov w8, #4173
	movk w8, #17952, lsl #16
	ldr q16, [sp, #96]
	dup v1.4s, w8
	mov w8, #46319
	stp q0, q2, [sp, #64]
	dup v0.4s, w9
	movk w8, #17487, lsl #16
	mov w9, #47186
	movk w9, #16718, lsl #16
	stp q0, q1, [sp, #32]
	dup v1.4s, w8
	mov w8, #12900
	movk w8, #16798, lsl #16
	dup v0.4s, w9
	dup v22.4s, w8
	stp q0, q1, [sp]
.LBB11_4:
	ldp q25, q26, [x0]
	ldur q0, [x29, #-208]
	ldp q27, q28, [x0, #32]
	fabs v8.4s, v25.4s
	fabs v31.4s, v26.4s
	fabs v30.4s, v27.4s
	fabs v29.4s, v28.4s
	fmaxnm v9.4s, v8.4s, v31.4s
	fmaxnm v10.4s, v30.4s, v29.4s
	fmaxnm v9.4s, v9.4s, v10.4s
	fcmgt v9.4s, v9.4s, v0.4s
	addp d9, v9.2d
	fmov x8, d9
	cbnz x8, .LBB11_10
	fsqrt v10.4s, v30.4s
	ldp q5, q4, [sp, #160]
	ldp q24, q19, [sp, #48]
	ldr q17, [sp, #112]
	ldp q2, q6, [sp, #128]
	ldr q1, [sp, #80]
	mov v11.16b, v4.16b
	mov v13.16b, v5.16b
	mov v7.16b, v17.16b
	mov v14.16b, v19.16b
	ldp q23, q21, [sp, #16]
	mov v15.16b, v2.16b
	mov v18.16b, v19.16b
	subs x1, x1, #1
	mov v20.16b, v21.16b
	fsqrt v9.4s, v31.4s
	fmla v11.4s, v3.4s, v10.4s
	fadd v12.4s, v10.4s, v16.4s
	fmla v13.4s, v10.4s, v11.4s
	fmla v14.4s, v10.4s, v12.4s
	mov v12.16b, v6.16b
	fsqrt v11.4s, v8.4s
	fmla v12.4s, v10.4s, v13.4s
	mov v13.16b, v24.16b
	fadd v0.4s, v9.4s, v16.4s
	fmla v13.4s, v10.4s, v14.4s
	mov v14.16b, v4.16b
	fmla v15.4s, v10.4s, v12.4s
	fmla v18.4s, v9.4s, v0.4s
	mov v0.16b, v23.16b
	fmla v14.4s, v3.4s, v9.4s
	fmla v20.4s, v10.4s, v13.4s
	mov v13.16b, v5.16b
	fsqrt v12.4s, v29.4s
	fmla v7.4s, v10.4s, v15.4s
	mov v15.16b, v1.16b
	fmla v13.4s, v9.4s, v14.4s
	mov v14.16b, v24.16b
	fmla v0.4s, v10.4s, v20.4s
	mov v20.16b, v6.16b
	fmla v15.4s, v10.4s, v7.4s
	mov v7.16b, v22.16b
	fmla v14.4s, v9.4s, v18.4s
	mov v18.16b, v4.16b
	fmla v20.4s, v9.4s, v13.4s
	mov v13.16b, v21.16b
	fmla v7.4s, v10.4s, v0.4s
	fadd v0.4s, v11.4s, v16.4s
	mov v10.16b, v2.16b
	fmla v18.4s, v3.4s, v11.4s
	fmla v13.4s, v9.4s, v14.4s
	mov v14.16b, v19.16b
	fmla v10.4s, v9.4s, v20.4s
	mov v20.16b, v5.16b
	fmla v4.4s, v3.4s, v12.4s
	fdiv v7.4s, v15.4s, v7.4s
	fmla v14.4s, v11.4s, v0.4s
	mov v0.16b, v17.16b
	fmla v20.4s, v11.4s, v18.4s
	mov v18.16b, v23.16b
	fmla v0.4s, v9.4s, v10.4s
	mov v10.16b, v6.16b
	fmla v5.4s, v12.4s, v4.4s
	fmla v18.4s, v9.4s, v13.4s
	mov v13.16b, v24.16b
	fmla v10.4s, v11.4s, v20.4s
	mov v20.16b, v1.16b
	fmla v13.4s, v11.4s, v14.4s
	mov v14.16b, v22.16b
	fmla v6.4s, v12.4s, v5.4s
	fmla v20.4s, v9.4s, v0.4s
	fadd v0.4s, v12.4s, v16.4s
	fmla v14.4s, v9.4s, v18.4s
	mov v18.16b, v2.16b
	mov v9.16b, v21.16b
	fmla v2.4s, v12.4s, v6.4s
	fmla v19.4s, v12.4s, v0.4s
	fmla v18.4s, v11.4s, v10.4s
	mov v10.16b, v1.16b
	fmla v9.4s, v11.4s, v13.4s
	fdiv v0.4s, v20.4s, v14.4s
	mov v20.16b, v17.16b
	mov v14.16b, v23.16b
	mov v13.16b, v22.16b
	fmla v17.4s, v12.4s, v2.4s
	fmla v24.4s, v12.4s, v19.4s
	fmla v20.4s, v11.4s, v18.4s
	fmla v14.4s, v11.4s, v9.4s
	fmla v1.4s, v12.4s, v17.4s
	fmla v21.4s, v12.4s, v24.4s
	fmla v10.4s, v11.4s, v20.4s
	fmla v13.4s, v11.4s, v14.4s
	mov v11.16b, v22.16b
	fmla v23.4s, v12.4s, v21.4s
	fdiv v18.4s, v10.4s, v13.4s
	fmla v11.4s, v12.4s, v23.4s
	ldr q12, [sp, #208]
	fcmgt v9.4s, v12.4s, v8.4s
	fcmgt v10.4s, v12.4s, v31.4s
	fdiv v20.4s, v1.4s, v11.4s
	ldr q1, [sp]
	fcmgt v11.4s, v12.4s, v30.4s
	fcmgt v12.4s, v12.4s, v29.4s
	fmul v8.4s, v8.4s, v1.4s
	fmul v31.4s, v31.4s, v1.4s
	fmul v30.4s, v30.4s, v1.4s
	fmul v29.4s, v29.4s, v1.4s
	mvni v1.4s, #128, lsl #24
	bit v18.16b, v8.16b, v9.16b
	bit v0.16b, v31.16b, v10.16b
	bit v7.16b, v30.16b, v11.16b
	bif v18.16b, v25.16b, v1.16b
	bif v0.16b, v26.16b, v1.16b
	bif v7.16b, v27.16b, v1.16b
	bit v20.16b, v29.16b, v12.16b
	stp q18, q0, [x0]
	bif v20.16b, v28.16b, v1.16b
	stp q7, q20, [x0, #32]
	add x0, x0, #64
	b.ne .LBB11_4
.LBB11_6:
	cbz x20, .LBB11_9
	lsl x21, x20, #2
	mov w8, #64
	mov w1, wzr
	sub x2, x8, x21
	sub x8, x29, #192
	add x0, x8, x21
	bl memset
	sub x0, x29, #192
	mov x1, x19
	mov x2, x21
	bl memcpy
	ldp q0, q1, [x29, #-192]
	mov w8, #1115684864
	ldp q2, q3, [x29, #-160]
	fabs v7.4s, v0.4s
	fabs v6.4s, v1.4s
	fabs v5.4s, v2.4s
	fabs v4.4s, v3.4s
	fmaxnm v16.4s, v7.4s, v6.4s
	fmaxnm v17.4s, v5.4s, v4.4s
	fmaxnm v16.4s, v16.4s, v17.4s
	dup v17.4s, w8
	fcmgt v16.4s, v16.4s, v17.4s
	addp d16, v16.2d
	fmov x8, d16
	cbnz x8, .LBB11_11
	fsqrt v19.4s, v5.4s
	mov w8, #60277
	mov w9, #59837
	movk w8, #17737, lsl #16
	movk w9, #17200, lsl #16
	sub x1, x29, #128
	dup v22.4s, w8
	mov w8, #63323
	dup v17.4s, w9
	movk w8, #17310, lsl #16
	mov w9, #30049
	mov x0, x19
	dup v18.4s, w8
	mov w8, #21186
	movk w9, #48797, lsl #16
	movk w8, #17949, lsl #16
	mov v24.16b, v22.16b
	mov v9.16b, v22.16b
	dup v20.4s, w8
	mov w8, #60486
	mov x2, x21
	movk w8, #17793, lsl #16
	dup v21.4s, w8
	mov w8, #44370
	fsqrt v16.4s, v6.4s
	mov v27.16b, v20.16b
	movk w8, #17885, lsl #16
	dup v26.4s, w8
	mov w8, #4173
	mov v13.16b, v20.16b
	movk w8, #17952, lsl #16
	fmla v24.4s, v17.4s, v19.4s
	fadd v25.4s, v19.4s, v18.4s
	dup v28.4s, w8
	mov w8, #26919
	mov v30.16b, v26.16b
	movk w8, #17536, lsl #16
	mov v8.16b, v28.16b
	fmla v27.4s, v19.4s, v24.4s
	mov v24.16b, v21.16b
	fsqrt v23.4s, v7.4s
	fmla v24.4s, v19.4s, v25.4s
	dup v25.4s, w8
	mov w8, #18007
	movk w8, #17852, lsl #16
	fmla v30.4s, v19.4s, v27.4s
	dup v27.4s, w9
	dup v29.4s, w8
	fmla v9.4s, v17.4s, v16.4s
	mov w8, #46319
	mov v10.16b, v25.16b
	movk w8, #17487, lsl #16
	mov w9, #17028
	fmla v8.4s, v19.4s, v24.4s
	fadd v12.4s, v16.4s, v18.4s
	movk w9, #49027, lsl #16
	mov v11.16b, v29.16b
	dup v31.4s, w8
	mov v14.16b, v27.16b
	fmla v10.4s, v19.4s, v30.4s
	dup v30.4s, w9
	fmla v13.4s, v16.4s, v9.4s
	mov v9.16b, v21.16b
	mov w8, #12900
	mov w9, #47186
	fmla v11.4s, v19.4s, v8.4s
	movk w8, #16798, lsl #16
	fsqrt v24.4s, v4.4s
	dup v8.4s, w8
	mov w8, #20545
	movk w9, #16718, lsl #16
	fmla v14.4s, v19.4s, v10.4s
	mov v10.16b, v31.16b
	fmla v9.4s, v16.4s, v12.4s
	mov v12.16b, v30.16b
	movk w8, #15175, lsl #16
	fmla v10.4s, v19.4s, v11.4s
	mov v11.16b, v26.16b
	fmla v12.4s, v19.4s, v14.4s
	mov v14.16b, v8.16b
	fmla v11.4s, v16.4s, v13.4s
	mov v13.16b, v28.16b
	fmla v14.4s, v19.4s, v10.4s
	mov v19.16b, v25.16b
	fadd v10.4s, v23.4s, v18.4s
	fmla v13.4s, v16.4s, v9.4s
	mov v9.16b, v22.16b
	fmla v22.4s, v17.4s, v23.4s
	fmla v19.4s, v16.4s, v11.4s
	mov v11.16b, v29.16b
	fdiv v12.4s, v12.4s, v14.4s
	fmla v9.4s, v17.4s, v24.4s
	mov v17.16b, v25.16b
	fmla v11.4s, v16.4s, v13.4s
	mov v13.16b, v20.16b
	fmla v20.4s, v23.4s, v22.4s
	mov v22.16b, v21.16b
	fmla v21.4s, v23.4s, v10.4s
	mov v10.16b, v27.16b
	fmla v13.4s, v24.4s, v9.4s
	fmla v10.4s, v16.4s, v19.4s
	mov v19.16b, v31.16b
	fmla v19.4s, v16.4s, v11.4s
	mov v11.16b, v26.16b
	fmla v26.4s, v23.4s, v20.4s
	mov v20.16b, v28.16b
	fmla v28.4s, v23.4s, v21.4s
	mov v21.16b, v30.16b
	fmla v11.4s, v24.4s, v13.4s
	fmla v21.4s, v16.4s, v10.4s
	mov v10.16b, v8.16b
	fmla v25.4s, v23.4s, v26.4s
	fmla v10.4s, v16.4s, v19.4s
	fadd v16.4s, v24.4s, v18.4s
	mov v18.16b, v29.16b
	fmla v29.4s, v23.4s, v28.4s
	fmla v17.4s, v24.4s, v11.4s
	mov v19.16b, v27.16b
	fmla v27.4s, v23.4s, v25.4s
	mov v25.16b, v8.16b
	fmla v22.4s, v24.4s, v16.4s
	fdiv v16.4s, v21.4s, v10.4s
	mov v21.16b, v31.16b
	fmla v31.4s, v23.4s, v29.4s
	fmla v19.4s, v24.4s, v17.4s
	fmla v20.4s, v24.4s, v22.4s
	mov v22.16b, v30.16b
	fmla v30.4s, v23.4s, v27.4s
	fmla v8.4s, v23.4s, v31.4s
	fmla v22.4s, v24.4s, v19.4s
	dup v19.4s, w9
	fmla v18.4s, v24.4s, v20.4s
	fdiv v20.4s, v30.4s, v8.4s
	fmla v21.4s, v24.4s, v18.4s
	dup v18.4s, w8
	fcmgt v23.4s, v18.4s, v5.4s
	fmul v5.4s, v5.4s, v19.4s
	fmla v25.4s, v24.4s, v21.4s
	fcmgt v21.4s, v18.4s, v7.4s
	fmul v7.4s, v7.4s, v19.4s
	bif v5.16b, v12.16b, v23.16b
	fdiv v17.4s, v22.4s, v25.4s
	fcmgt v22.4s, v18.4s, v6.4s
	fmul v6.4s, v6.4s, v19.4s
	fcmgt v18.4s, v18.4s, v4.4s
	fmul v4.4s, v4.4s, v19.4s
	mvni v19.4s, #128, lsl #24
	bif v7.16b, v20.16b, v21.16b
	bif v6.16b, v16.16b, v22.16b
	bit v2.16b, v5.16b, v19.16b
	bit v0.16b, v7.16b, v19.16b
	bit v1.16b, v6.16b, v19.16b
	stp q0, q1, [x29, #-128]
	bif v4.16b, v17.16b, v18.16b
	bit v3.16b, v4.16b, v19.16b
	stp q2, q3, [x29, #-96]
	bl memcpy
.LBB11_9:
	.cfi_def_cfa wsp, 480
	ldp x20, x19, [sp, #464]
	ldp x28, x21, [sp, #448]
	ldp x29, x30, [sp, #432]
	ldp d9, d8, [sp, #416]
	ldp d11, d10, [sp, #400]
	ldp d13, d12, [sp, #384]
	ldp d15, d14, [sp, #368]
	add sp, sp, #480
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
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
.LBB11_10:
	.cfi_restore_state
	.cfi_remember_state
	adrp x4, linear_srgb::kernels::float::x16::linear_to_srgb_extended_slice::slow::<archmage::tokens::generated::arm::NeonToken>
	add x4, x4, :lo12:linear_srgb::kernels::float::x16::linear_to_srgb_extended_slice::slow::<archmage::tokens::generated::arm::NeonToken>
	mov x2, x19
	mov x3, x20
	.cfi_def_cfa wsp, 480
	ldp x20, x19, [sp, #464]
	ldp x28, x21, [sp, #448]
	ldp x29, x30, [sp, #432]
	ldp d9, d8, [sp, #416]
	ldp d11, d10, [sp, #400]
	ldp d13, d12, [sp, #384]
	ldp d15, d14, [sp, #368]
	add sp, sp, #480
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
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
	b linear_srgb::kernels::finish_slice_cold::<archmage::tokens::generated::arm::NeonToken, 16>
.LBB11_11:
	.cfi_restore_state
	adrp x4, linear_srgb::kernels::float::x16::linear_to_srgb_extended_slice::slow::<archmage::tokens::generated::arm::NeonToken>
	add x4, x4, :lo12:linear_srgb::kernels::float::x16::linear_to_srgb_extended_slice::slow::<archmage::tokens::generated::arm::NeonToken>
	mov w0, #4
	mov x1, xzr
	mov x2, x19
	mov x3, x20
	bl linear_srgb::kernels::finish_slice_cold::<archmage::tokens::generated::arm::NeonToken, 16>
	b .LBB11_9
.LBB11_12:
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB11_2
.LBB11_13:
	.cfi_def_cfa wsp, 480
	ldp x20, x19, [sp, #464]
	ldp x28, x21, [sp, #448]
	ldp x29, x30, [sp, #432]
	ldp d9, d8, [sp, #416]
	ldp d11, d10, [sp, #400]
	ldp d13, d12, [sp, #384]
	ldp d15, d14, [sp, #368]
	add sp, sp, #480
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
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
	b linear_srgb::simd::linear_to_srgb_extended_slice_tier_scalar_t
