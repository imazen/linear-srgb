.section .text.stub_srgb_to_linear_extended_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_extended_slice
	.p2align	2
.type	stub_srgb_to_linear_extended_slice,@function
stub_srgb_to_linear_extended_slice:
	.cfi_startproc
	sub sp, sp, #272
	.cfi_def_cfa_offset 272
	stp d15, d14, [sp, #144]
	stp d13, d12, [sp, #160]
	stp d11, d10, [sp, #176]
	stp d9, d8, [sp, #192]
	stp x29, x30, [sp, #208]
	stp x28, x23, [sp, #224]
	stp x22, x21, [sp, #240]
	stp x20, x19, [sp, #256]
	add x29, sp, #208
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
	b.eq .LBB11_15
	cmp w8, #2
	b.ne .LBB11_14
.LBB11_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x21, x1, #0xf
	lsr x22, x1, #4
	add x19, x0, x8, lsl #2
	lsl x20, x21, #2
	b.eq .LBB11_4
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
	b .LBB11_5
.LBB11_4:
	mov w9, wzr
.LBB11_5:
	mov w8, #33681
	mov w10, #21091
	cmp x21, #0
	movk w8, #15774, lsl #16
	movk w10, #18053, lsl #16
	str w9, [sp, #12]
	dup v0.4s, w8
	mov w8, #45243
	dup v1.4s, w10
	movk w8, #18434, lsl #16
	mov w10, #26646
	dup v2.4s, w8
	mov w8, #54856
	movk w10, #18037, lsl #16
	movk w8, #18518, lsl #16
	dup v5.4s, w10
	mov w10, #23593
	dup v3.4s, w8
	mov w8, #22505
	movk w10, #49790, lsl #16
	movk w8, #18375, lsl #16
	dup v16.4s, w10
	mov w10, #61050
	dup v4.4s, w8
	mov w8, #49826
	movk w10, #18528, lsl #16
	movk w8, #17507, lsl #16
	dup v19.4s, w10
	mov w10, #61974
	dup v6.4s, w8
	mov w8, #11201
	movk w10, #15648, lsl #16
	movk w8, #16784, lsl #16
	dup v22.4s, w10
	add x10, sp, #12
	dup v7.4s, w8
	mov w8, #51710
	add x1, x10, #4
	movk w8, #17803, lsl #16
	csel x9, xzr, x1, eq
	dup v17.4s, w8
	mov w8, #60686
	movk w8, #18336, lsl #16
	dup v18.4s, w8
	mov w8, #20961
	movk w8, #18451, lsl #16
	dup v20.4s, w8
	mov w8, #46089
	movk w8, #18088, lsl #16
	dup v21.4s, w8
	add x8, x0, x22, lsl #6
	b .LBB11_8
.LBB11_6:
	add x10, x0, #64
.LBB11_7:
	ldp q24, q23, [x0]
	mov v26.16b, v2.16b
	mov v28.16b, v3.16b
	mov v29.16b, v17.16b
	mov v30.16b, v18.16b
	mov v8.16b, v5.16b
	mov v9.16b, v19.16b
	mov v10.16b, v17.16b
	fabs v25.4s, v24.4s
	mov v11.16b, v20.16b
	mov v12.16b, v2.16b
	mov v14.16b, v2.16b
	fcmlt v24.4s, v24.4s, #0.0
	fmla v26.4s, v1.4s, v25.4s
	fadd v27.4s, v25.4s, v16.4s
	fmla v28.4s, v25.4s, v26.4s
	fmla v29.4s, v25.4s, v27.4s
	fabs v26.4s, v23.4s
	mov v27.16b, v4.16b
	fcmlt v23.4s, v23.4s, #0.0
	fmla v27.4s, v25.4s, v28.4s
	fmla v30.4s, v25.4s, v29.4s
	mov v29.16b, v2.16b
	fadd v31.4s, v26.4s, v16.4s
	fmla v29.4s, v1.4s, v26.4s
	fmla v8.4s, v25.4s, v27.4s
	fmla v9.4s, v25.4s, v30.4s
	mov v30.16b, v3.16b
	fmla v10.4s, v26.4s, v31.4s
	mov v31.16b, v6.16b
	ldp q28, q27, [x0, #32]
	fmla v30.4s, v26.4s, v29.4s
	fmla v31.4s, v25.4s, v8.4s
	fmla v11.4s, v25.4s, v9.4s
	mov v8.16b, v4.16b
	mov v9.16b, v18.16b
	fabs v29.4s, v28.4s
	fcmlt v28.4s, v28.4s, #0.0
	fmla v8.4s, v26.4s, v30.4s
	mov v30.16b, v7.16b
	fmla v9.4s, v26.4s, v10.4s
	mov v10.16b, v21.16b
	fmla v12.4s, v1.4s, v29.4s
	fadd v13.4s, v29.4s, v16.4s
	fmla v30.4s, v25.4s, v31.4s
	mov v31.16b, v5.16b
	fmla v10.4s, v25.4s, v11.4s
	mov v11.16b, v19.16b
	fmla v31.4s, v26.4s, v8.4s
	mov v8.16b, v3.16b
	fmla v11.4s, v26.4s, v9.4s
	mov v9.16b, v17.16b
	fdiv v30.4s, v30.4s, v10.4s
	mov v10.16b, v6.16b
	fmla v8.4s, v29.4s, v12.4s
	mov v12.16b, v20.16b
	fmla v9.4s, v29.4s, v13.4s
	fabs v13.4s, v27.4s
	fcmlt v27.4s, v27.4s, #0.0
	fmla v10.4s, v26.4s, v31.4s
	mov v31.16b, v4.16b
	fmla v12.4s, v26.4s, v11.4s
	mov v11.16b, v18.16b
	fmla v31.4s, v29.4s, v8.4s
	mov v8.16b, v7.16b
	fmla v14.4s, v1.4s, v13.4s
	fmla v11.4s, v29.4s, v9.4s
	mov v9.16b, v21.16b
	fadd v15.4s, v13.4s, v16.4s
	fmla v8.4s, v26.4s, v10.4s
	mov v10.16b, v5.16b
	fmla v9.4s, v26.4s, v12.4s
	mov v12.16b, v19.16b
	fmla v10.4s, v29.4s, v31.4s
	mov v31.16b, v3.16b
	fmla v12.4s, v29.4s, v11.4s
	mov v11.16b, v17.16b
	fdiv v8.4s, v8.4s, v9.4s
	mov v9.16b, v6.16b
	fmla v31.4s, v13.4s, v14.4s
	mov v14.16b, v20.16b
	fmla v11.4s, v13.4s, v15.4s
	fmla v9.4s, v29.4s, v10.4s
	mov v10.16b, v4.16b
	fmla v14.4s, v29.4s, v12.4s
	mov v12.16b, v18.16b
	fmla v10.4s, v13.4s, v31.4s
	mov v31.16b, v7.16b
	fmla v12.4s, v13.4s, v11.4s
	mov v11.16b, v21.16b
	fmla v31.4s, v29.4s, v9.4s
	mov v9.16b, v5.16b
	fmla v11.4s, v29.4s, v14.4s
	mov v14.16b, v19.16b
	fmla v9.4s, v13.4s, v10.4s
	mov v10.16b, v6.16b
	fmla v14.4s, v13.4s, v12.4s
	mov v12.16b, v21.16b
	fdiv v31.4s, v31.4s, v11.4s
	mov v11.16b, v20.16b
	fmla v10.4s, v13.4s, v9.4s
	mov v9.16b, v7.16b
	fmla v11.4s, v13.4s, v14.4s
	fmul v14.4s, v13.4s, v0.4s
	fmla v9.4s, v13.4s, v10.4s
	fmul v10.4s, v25.4s, v0.4s
	fcmgt v25.4s, v22.4s, v25.4s
	fmla v12.4s, v13.4s, v11.4s
	fmul v11.4s, v26.4s, v0.4s
	fcmgt v26.4s, v22.4s, v26.4s
	fcmgt v13.4s, v22.4s, v13.4s
	bsl v25.16b, v10.16b, v30.16b
	fdiv v9.4s, v9.4s, v12.4s
	fmul v12.4s, v29.4s, v0.4s
	fcmgt v29.4s, v22.4s, v29.4s
	bsl v26.16b, v11.16b, v8.16b
	mov v30.16b, v13.16b
	bsl v29.16b, v12.16b, v31.16b
	fneg v31.4s, v25.4s
	fneg v8.4s, v26.4s
	bsl v24.16b, v31.16b, v25.16b
	mov v25.16b, v28.16b
	bsl v23.16b, v8.16b, v26.16b
	mov v26.16b, v27.16b
	bsl v30.16b, v14.16b, v9.16b
	fneg v9.4s, v29.4s
	stp q24, q23, [x0]
	fneg v10.4s, v30.4s
	bsl v25.16b, v9.16b, v29.16b
	bsl v26.16b, v10.16b, v30.16b
	stp q25, q26, [x0, #32]
	mov x0, x10
.LBB11_8:
	cmp x0, x8
	ccmp x0, #0, #4, ne
	b.ne .LBB11_6
	cbz x9, .LBB11_11
	mov x10, xzr
	mov x0, x9
	mov x9, xzr
	b .LBB11_7
.LBB11_11:
	ldr w8, [sp, #12]
	cbz w8, .LBB11_13
	mov x0, x19
	mov x2, x20
	bl memcpy
.LBB11_13:
	.cfi_def_cfa wsp, 272
	ldp x20, x19, [sp, #256]
	ldp x22, x21, [sp, #240]
	ldp x28, x23, [sp, #224]
	ldp x29, x30, [sp, #208]
	ldp d9, d8, [sp, #192]
	ldp d11, d10, [sp, #176]
	ldp d13, d12, [sp, #160]
	ldp d15, d14, [sp, #144]
	add sp, sp, #272
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
.LBB11_14:
	.cfi_restore_state
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB11_2
.LBB11_15:
	.cfi_def_cfa wsp, 272
	ldp x20, x19, [sp, #256]
	ldp x22, x21, [sp, #240]
	ldp x28, x23, [sp, #224]
	ldp x29, x30, [sp, #208]
	ldp d9, d8, [sp, #192]
	ldp d11, d10, [sp, #176]
	ldp d13, d12, [sp, #160]
	ldp d15, d14, [sp, #144]
	add sp, sp, #272
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
	b linear_srgb::simd::srgb_to_linear_extended_slice_tier_scalar_t
