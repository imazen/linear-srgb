.section .text.stub_srgb_to_linear_premultiply_rgba_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_premultiply_rgba_slice
	.p2align	2
.type	stub_srgb_to_linear_premultiply_rgba_slice,@function
stub_srgb_to_linear_premultiply_rgba_slice:
	.cfi_startproc
	sub sp, sp, #240
	.cfi_def_cfa_offset 240
	stp d11, d10, [sp, #144]
	stp d9, d8, [sp, #160]
	stp x29, x30, [sp, #176]
	str x23, [sp, #192]
	stp x22, x21, [sp, #208]
	stp x20, x19, [sp, #224]
	add x29, sp, #176
	.cfi_def_cfa w29, 64
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -48
	.cfi_offset w30, -56
	.cfi_offset w29, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	.cfi_offset b10, -88
	.cfi_offset b11, -96
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
	b .LBB12_5
.LBB12_4:
	mov w9, wzr
.LBB12_5:
	mov w8, #33681
	mov w10, #32277
	fmov v0.4s, #1.00000000
	movk w8, #15774, lsl #16
	movk w10, #17060, lsl #16
	cmp x21, #0
	dup v1.4s, w8
	mov w8, #5734
	dup v2.4s, w10
	movk w8, #17033, lsl #16
	mov w10, #6038
	str w9, [sp, #12]
	dup v3.4s, w8
	mov w8, #9853
	movk w10, #16207, lsl #16
	movk w8, #16718, lsl #16
	dup v5.4s, w10
	mov w10, #6038
	dup v4.4s, w8
	mov w8, #2423
	movk w10, #16983, lsl #16
	movk w8, #15497, lsl #16
	dup v16.4s, w10
	mov w10, #61974
	dup v6.4s, w8
	mov w8, #1338
	movk w10, #15648, lsl #16
	movk w8, #49380, lsl #16
	dup v19.4s, w10
	add x10, sp, #12
	dup v7.4s, w8
	mov w8, #41246
	add x1, x10, #4
	movk w8, #17089, lsl #16
	csel x9, xzr, x1, eq
	dup v17.4s, w8
	mov w8, #19964
	movk w8, #16800, lsl #16
	dup v18.4s, w8
	add x8, x0, x22, lsl #6
	b .LBB12_8
.LBB12_6:
	add x10, x0, #64
.LBB12_7:
	ldp q21, q20, [x0]
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
	fminnm v22.4s, v22.4s, v0.4s
	fmla v23.4s, v2.4s, v22.4s
	fadd v25.4s, v22.4s, v7.4s
	fmla v26.4s, v22.4s, v23.4s
	fmla v27.4s, v22.4s, v25.4s
	fminnm v25.4s, v24.4s, v0.4s
	ldp q23, q24, [x0, #32]
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
	fmla v30.4s, v26.4s, v28.4s
	fmla v31.4s, v26.4s, v8.4s
	fminnm v28.4s, v29.4s, v0.4s
	fdiv v29.4s, v9.4s, v10.4s
	mov v8.16b, v5.16b
	mov v9.16b, v17.16b
	mov v10.16b, v6.16b
	fminnm v27.4s, v27.4s, v0.4s
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
	fminnm v29.4s, v29.4s, v0.4s
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
	fminnm v30.4s, v30.4s, v0.4s
	bsl v22.16b, v8.16b, v27.16b
	fcmge v27.4s, v21.4s, v0.4s
	bsl v25.16b, v9.16b, v29.16b
	fcmge v29.4s, v20.4s, v0.4s
	bsl v26.16b, v10.16b, v30.16b
	fcmge v30.4s, v23.4s, v0.4s
	bit v22.16b, v0.16b, v27.16b
	bit v25.16b, v0.16b, v29.16b
	bit v26.16b, v0.16b, v30.16b
	fmul v22.4s, v22.4s, v21.s[3]
	fminnm v31.4s, v31.4s, v0.4s
	fmul v25.4s, v25.4s, v20.s[3]
	fmul v26.4s, v26.4s, v23.s[3]
	mov v22.s[3], v21.s[3]
	bsl v28.16b, v11.16b, v31.16b
	fcmge v31.4s, v24.4s, v0.4s
	mov v25.s[3], v20.s[3]
	mov v26.s[3], v23.s[3]
	mov v27.16b, v31.16b
	stp q22, q25, [x0]
	bsl v27.16b, v0.16b, v28.16b
	fmul v27.4s, v27.4s, v24.s[3]
	mov v27.s[3], v24.s[3]
	stp q26, q27, [x0, #32]
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
	ldr w8, [sp, #12]
	cbz w8, .LBB12_13
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB12_13:
	.cfi_def_cfa wsp, 240
	ldp x20, x19, [sp, #224]
	ldr x23, [sp, #192]
	ldp x22, x21, [sp, #208]
	ldp x29, x30, [sp, #176]
	ldp d9, d8, [sp, #160]
	ldp d11, d10, [sp, #144]
	add sp, sp, #240
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
	.cfi_def_cfa wsp, 240
	ldp x20, x19, [sp, #224]
	ldr x23, [sp, #192]
	ldp x22, x21, [sp, #208]
	ldp x29, x30, [sp, #176]
	ldp d9, d8, [sp, #160]
	ldp d11, d10, [sp, #144]
	add sp, sp, #240
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
	b linear_srgb::simd::srgb_to_linear_premultiply_rgba_slice_tier_scalar
