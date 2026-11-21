.section .text.stub_srgb_to_linear_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_slice
	.p2align	2
.type	stub_srgb_to_linear_slice,@function
stub_srgb_to_linear_slice:
	.cfi_startproc
	sub sp, sp, #192
	.cfi_def_cfa_offset 192
	stp d11, d10, [sp, #128]
	stp d9, d8, [sp, #144]
	stp x29, x30, [sp, #160]
	stp x20, x19, [sp, #176]
	add x29, sp, #160
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	.cfi_offset b8, -40
	.cfi_offset b9, -48
	.cfi_offset b10, -56
	.cfi_offset b11, -64
	.cfi_remember_state
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB17_9
	cmp w8, #2
	b.ne .LBB17_8
.LBB17_2:
	lsl x8, x1, #2
	ands x9, x8, #0x7fffffffffffffc0
	and x8, x1, #0xf
	b.eq .LBB17_5
	mov w10, #33681
	mov w11, #32277
	mov w12, #5734
	movk w10, #15774, lsl #16
	movk w11, #17060, lsl #16
	movk w12, #17033, lsl #16
	dup v1.4s, w10
	dup v2.4s, w11
	mov w10, #9853
	mov w11, #6038
	movk w10, #16718, lsl #16
	dup v3.4s, w12
	movk w11, #16207, lsl #16
	dup v4.4s, w10
	mov w10, #2423
	dup v5.4s, w11
	mov w11, #1338
	mov w12, #6038
	movk w10, #15497, lsl #16
	movk w11, #49380, lsl #16
	movk w12, #16983, lsl #16
	dup v6.4s, w10
	dup v7.4s, w11
	dup v16.4s, w12
	mov w10, #41246
	mov w11, #19964
	mov w12, #61974
	fmov v0.4s, #1.00000000
	movk w10, #17089, lsl #16
	movk w11, #16800, lsl #16
	movk w12, #15648, lsl #16
	dup v17.4s, w10
	dup v18.4s, w11
	dup v19.4s, w12
	add x9, x0, x9
	mov x10, x0
.LBB17_4:
	ldp q21, q20, [x10]
	mov v23.16b, v3.16b
	mov v26.16b, v4.16b
	mov v27.16b, v16.16b
	mov v28.16b, v5.16b
	mov v29.16b, v17.16b
	mov v31.16b, v6.16b
	mov v8.16b, v18.16b
	fcmgt v22.4s, v21.4s, #0.0
	fcmgt v24.4s, v20.4s, #0.0
	mov v9.16b, v6.16b
	mov v10.16b, v18.16b
	mov v11.16b, v18.16b
	and v22.16b, v22.16b, v21.16b
	and v24.16b, v24.16b, v20.16b
	fcmge v21.4s, v21.4s, v0.4s
	fcmge v20.4s, v20.4s, v0.4s
	fminnm v22.4s, v22.4s, v0.4s
	fmla v23.4s, v2.4s, v22.4s
	fadd v25.4s, v22.4s, v7.4s
	fmla v26.4s, v22.4s, v23.4s
	fmla v27.4s, v22.4s, v25.4s
	fminnm v25.4s, v24.4s, v0.4s
	ldp q23, q24, [x10, #32]
	fmla v28.4s, v22.4s, v26.4s
	fmla v29.4s, v22.4s, v27.4s
	mov v26.16b, v3.16b
	fcmgt v27.4s, v23.4s, #0.0
	fadd v30.4s, v25.4s, v7.4s
	fmla v26.4s, v2.4s, v25.4s
	fmla v31.4s, v22.4s, v28.4s
	fmla v8.4s, v22.4s, v29.4s
	mov v28.16b, v4.16b
	mov v29.16b, v16.16b
	and v27.16b, v27.16b, v23.16b
	fcmge v23.4s, v23.4s, v0.4s
	fmla v28.4s, v25.4s, v26.4s
	fmla v29.4s, v25.4s, v30.4s
	fminnm v26.4s, v27.4s, v0.4s
	fdiv v27.4s, v31.4s, v8.4s
	mov v30.16b, v5.16b
	mov v31.16b, v17.16b
	fmla v30.4s, v25.4s, v28.4s
	fmla v31.4s, v25.4s, v29.4s
	mov v28.16b, v3.16b
	fcmgt v29.4s, v24.4s, #0.0
	fadd v8.4s, v26.4s, v7.4s
	fmla v28.4s, v2.4s, v26.4s
	fmla v9.4s, v25.4s, v30.4s
	fmla v10.4s, v25.4s, v31.4s
	mov v30.16b, v4.16b
	mov v31.16b, v16.16b
	and v29.16b, v29.16b, v24.16b
	fcmge v24.4s, v24.4s, v0.4s
	fmla v30.4s, v26.4s, v28.4s
	fmla v31.4s, v26.4s, v8.4s
	fminnm v28.4s, v29.4s, v0.4s
	fdiv v29.4s, v9.4s, v10.4s
	mov v8.16b, v5.16b
	mov v9.16b, v17.16b
	mov v10.16b, v6.16b
	fmla v8.4s, v26.4s, v30.4s
	fmla v9.4s, v26.4s, v31.4s
	mov v30.16b, v3.16b
	fadd v31.4s, v28.4s, v7.4s
	fmla v30.4s, v2.4s, v28.4s
	fmla v10.4s, v26.4s, v8.4s
	fmla v11.4s, v26.4s, v9.4s
	mov v8.16b, v4.16b
	mov v9.16b, v16.16b
	fmla v8.4s, v28.4s, v30.4s
	fmla v9.4s, v28.4s, v31.4s
	fdiv v30.4s, v10.4s, v11.4s
	mov v31.16b, v5.16b
	mov v10.16b, v17.16b
	fmul v11.4s, v28.4s, v1.4s
	fmla v31.4s, v28.4s, v8.4s
	mov v8.16b, v6.16b
	fmla v10.4s, v28.4s, v9.4s
	mov v9.16b, v18.16b
	fmla v8.4s, v28.4s, v31.4s
	fmla v9.4s, v28.4s, v10.4s
	fcmge v28.4s, v19.4s, v28.4s
	fmul v10.4s, v26.4s, v1.4s
	fcmge v26.4s, v19.4s, v26.4s
	fdiv v31.4s, v8.4s, v9.4s
	fmul v8.4s, v22.4s, v1.4s
	fcmge v22.4s, v19.4s, v22.4s
	fmul v9.4s, v25.4s, v1.4s
	fcmge v25.4s, v19.4s, v25.4s
	bsl v26.16b, v10.16b, v30.16b
	bsl v22.16b, v8.16b, v27.16b
	mov v27.16b, v28.16b
	bsl v25.16b, v9.16b, v29.16b
	bsl v21.16b, v0.16b, v22.16b
	mov v22.16b, v23.16b
	mov v23.16b, v24.16b
	bsl v20.16b, v0.16b, v25.16b
	bsl v22.16b, v0.16b, v26.16b
	bsl v27.16b, v11.16b, v31.16b
	stp q21, q20, [x10]
	bsl v23.16b, v0.16b, v27.16b
	stp q22, q23, [x10, #32]
	add x10, x10, #64
	cmp x10, x9
	b.ne .LBB17_4
.LBB17_5:
	cbz x8, .LBB17_7
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
	ldp q2, q1, [sp]
	fmov v0.4s, #1.00000000
	mov w9, #5734
	mov w8, #32277
	add x1, sp, #64
	movk w9, #17033, lsl #16
	movk w8, #17060, lsl #16
	mov x0, x20
	fcmgt v3.4s, v2.4s, #0.0
	dup v7.4s, w8
	mov w8, #9853
	movk w8, #16718, lsl #16
	fcmgt v5.4s, v1.4s, #0.0
	mov x2, x19
	dup v6.4s, w8
	mov w8, #6038
	movk w8, #16207, lsl #16
	and v4.16b, v3.16b, v2.16b
	dup v3.4s, w9
	mov w9, #1338
	movk w9, #49380, lsl #16
	mov v22.16b, v6.16b
	dup v19.4s, w8
	dup v16.4s, w9
	mov w9, #6038
	and v24.16b, v5.16b, v1.16b
	fminnm v4.4s, v4.4s, v0.4s
	movk w9, #16983, lsl #16
	mov v18.16b, v3.16b
	dup v17.4s, w9
	mov w9, #41246
	mov w8, #2423
	movk w9, #17089, lsl #16
	movk w8, #15497, lsl #16
	mov v25.16b, v19.16b
	dup v20.4s, w9
	mov w9, #19964
	fcmge v2.4s, v2.4s, v0.4s
	fmla v18.4s, v7.4s, v4.4s
	fadd v21.4s, v4.4s, v16.4s
	mov v23.16b, v17.16b
	movk w9, #16800, lsl #16
	fcmge v1.4s, v1.4s, v0.4s
	mov v26.16b, v20.16b
	dup v27.4s, w9
	mov w9, #61974
	movk w9, #15648, lsl #16
	fmla v22.4s, v4.4s, v18.4s
	fmla v23.4s, v4.4s, v21.4s
	fminnm v21.4s, v24.4s, v0.4s
	dup v24.4s, w8
	mov v30.16b, v27.16b
	mov v8.16b, v27.16b
	ldp q5, q18, [sp, #32]
	mov w8, #33681
	movk w8, #15774, lsl #16
	fmla v25.4s, v4.4s, v22.4s
	fmla v26.4s, v4.4s, v23.4s
	mov v23.16b, v3.16b
	mov v29.16b, v24.16b
	fadd v28.4s, v21.4s, v16.4s
	fcmgt v22.4s, v5.4s, #0.0
	mov v31.16b, v24.16b
	fmla v23.4s, v7.4s, v21.4s
	fmla v29.4s, v4.4s, v25.4s
	mov v25.16b, v6.16b
	fmla v30.4s, v4.4s, v26.4s
	mov v26.16b, v17.16b
	and v22.16b, v22.16b, v5.16b
	fcmge v5.4s, v5.4s, v0.4s
	fmla v25.4s, v21.4s, v23.4s
	fmla v26.4s, v21.4s, v28.4s
	mov v28.16b, v19.16b
	fdiv v23.4s, v29.4s, v30.4s
	mov v29.16b, v20.16b
	fminnm v22.4s, v22.4s, v0.4s
	fmla v28.4s, v21.4s, v25.4s
	fcmgt v25.4s, v18.4s, #0.0
	fmla v29.4s, v21.4s, v26.4s
	mov v26.16b, v3.16b
	fadd v30.4s, v22.4s, v16.4s
	and v25.16b, v25.16b, v18.16b
	fmla v26.4s, v7.4s, v22.4s
	fmla v31.4s, v21.4s, v28.4s
	fmla v8.4s, v21.4s, v29.4s
	mov v28.16b, v6.16b
	mov v29.16b, v17.16b
	fcmge v18.4s, v18.4s, v0.4s
	fminnm v25.4s, v25.4s, v0.4s
	fmla v28.4s, v22.4s, v26.4s
	fmla v29.4s, v22.4s, v30.4s
	mov v30.16b, v19.16b
	fdiv v26.4s, v31.4s, v8.4s
	mov v31.16b, v20.16b
	fmla v3.4s, v7.4s, v25.4s
	fadd v7.4s, v25.4s, v16.4s
	mov v16.16b, v24.16b
	fmla v30.4s, v22.4s, v28.4s
	fmla v31.4s, v22.4s, v29.4s
	mov v28.16b, v27.16b
	fmla v6.4s, v25.4s, v3.4s
	fmla v17.4s, v25.4s, v7.4s
	dup v7.4s, w8
	fmla v16.4s, v22.4s, v30.4s
	fmla v28.4s, v22.4s, v31.4s
	fmla v19.4s, v25.4s, v6.4s
	fmla v20.4s, v25.4s, v17.4s
	fmul v17.4s, v4.4s, v7.4s
	fdiv v3.4s, v16.4s, v28.4s
	dup v16.4s, w9
	fmla v24.4s, v25.4s, v19.4s
	fmla v27.4s, v25.4s, v20.4s
	fcmge v4.4s, v16.4s, v4.4s
	fcmge v20.4s, v16.4s, v21.4s
	fmul v19.4s, v21.4s, v7.4s
	fmul v21.4s, v22.4s, v7.4s
	fcmge v22.4s, v16.4s, v22.4s
	fmul v7.4s, v25.4s, v7.4s
	fcmge v16.4s, v16.4s, v25.4s
	bsl v4.16b, v17.16b, v23.16b
	mov v17.16b, v20.16b
	fdiv v6.4s, v24.4s, v27.4s
	bsl v17.16b, v19.16b, v26.16b
	bsl v2.16b, v0.16b, v4.16b
	bit v3.16b, v21.16b, v22.16b
	bsl v1.16b, v0.16b, v17.16b
	bit v3.16b, v0.16b, v5.16b
	stp q2, q1, [sp, #64]
	bit v6.16b, v7.16b, v16.16b
	bif v0.16b, v6.16b, v18.16b
	stp q3, q0, [sp, #96]
	bl memcpy
.LBB17_7:
	.cfi_def_cfa wsp, 192
	ldp x20, x19, [sp, #176]
	ldp x29, x30, [sp, #160]
	ldp d9, d8, [sp, #144]
	ldp d11, d10, [sp, #128]
	add sp, sp, #192
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	ret
.LBB17_8:
	.cfi_restore_state
	mov x20, x1
	mov x19, x0
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB17_2
.LBB17_9:
	.cfi_def_cfa wsp, 192
	ldp x20, x19, [sp, #176]
	ldp x29, x30, [sp, #160]
	ldp d9, d8, [sp, #144]
	ldp d11, d10, [sp, #128]
	add sp, sp, #192
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	b linear_srgb::simd::srgb_to_linear_slice_tier_scalar_t
