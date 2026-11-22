.section .text.stub_srgb_to_linear_extended_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_extended_slice
	.p2align	2
.type	stub_srgb_to_linear_extended_slice,@function
stub_srgb_to_linear_extended_slice:
	.cfi_startproc
	stp x29, x30, [sp, #-32]!
	.cfi_def_cfa_offset 32
	stp x20, x19, [sp, #16]
	mov x29, sp
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	.cfi_remember_state
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB20_10
	cmp w8, #2
	b.ne .LBB20_9
.LBB20_2:
	and x8, x1, #0x1ffffffffffffffc
	mov x9, x1
	lsr x1, x1, #2
	add x2, x0, x8, lsl #2
	and x3, x9, #0x3
	cbz x1, .LBB20_6
	mov w8, #61974
	mov w9, #21091
	movi v0.4s, #65, lsl #24
	movk w8, #15648, lsl #16
	movk w9, #18053, lsl #16
	mvni v24.4s, #128, lsl #24
	dup v1.4s, w8
	mov w8, #45243
	dup v2.4s, w9
	movk w8, #18434, lsl #16
	mov w9, #26646
	dup v3.4s, w8
	mov w8, #54856
	movk w9, #18037, lsl #16
	movk w8, #18518, lsl #16
	dup v6.4s, w9
	mov w9, #23593
	dup v4.4s, w8
	mov w8, #22505
	movk w9, #49790, lsl #16
	movk w8, #18375, lsl #16
	dup v17.4s, w9
	mov w9, #61050
	dup v5.4s, w8
	mov w8, #49826
	movk w9, #18528, lsl #16
	movk w8, #17507, lsl #16
	dup v20.4s, w9
	mov w9, #33681
	dup v7.4s, w8
	mov w8, #11201
	movk w9, #15774, lsl #16
	movk w8, #16784, lsl #16
	dup v23.4s, w9
	dup v16.4s, w8
	mov w8, #51710
	movk w8, #17803, lsl #16
	dup v18.4s, w8
	mov w8, #60686
	movk w8, #18336, lsl #16
	dup v19.4s, w8
	mov w8, #20961
	movk w8, #18451, lsl #16
	dup v21.4s, w8
	mov w8, #46089
	movk w8, #18088, lsl #16
	dup v22.4s, w8
.LBB20_4:
	ldr q25, [x0]
	fabs v26.4s, v25.4s
	fcmgt v27.4s, v26.4s, v0.4s
	addp d27, v27.2d
	fmov x8, d27
	cbnz x8, .LBB20_8
	mov v27.16b, v3.16b
	fadd v28.4s, v26.4s, v17.4s
	subs x1, x1, #1
	mov v29.16b, v4.16b
	mov v30.16b, v18.16b
	fmla v27.4s, v2.4s, v26.4s
	fmla v30.4s, v26.4s, v28.4s
	mov v28.16b, v19.16b
	fmla v29.4s, v26.4s, v27.4s
	mov v27.16b, v5.16b
	fmla v28.4s, v26.4s, v30.4s
	mov v30.16b, v20.16b
	fmla v27.4s, v26.4s, v29.4s
	mov v29.16b, v6.16b
	fmla v30.4s, v26.4s, v28.4s
	mov v28.16b, v21.16b
	fmla v29.4s, v26.4s, v27.4s
	mov v27.16b, v7.16b
	fmla v28.4s, v26.4s, v30.4s
	mov v30.16b, v22.16b
	fmla v27.4s, v26.4s, v29.4s
	mov v29.16b, v16.16b
	fmla v30.4s, v26.4s, v28.4s
	fcmgt v28.4s, v1.4s, v26.4s
	fmla v29.4s, v26.4s, v27.4s
	fmul v26.4s, v26.4s, v23.4s
	fdiv v27.4s, v29.4s, v30.4s
	bif v26.16b, v27.16b, v28.16b
	bit v25.16b, v26.16b, v24.16b
	str q25, [x0], #16
	b.ne .LBB20_4
.LBB20_6:
	cbnz x3, .LBB20_11
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB20_8:
	.cfi_restore_state
	.cfi_remember_state
	adrp x4, linear_srgb::kernels::float::x4::srgb_to_linear_extended_slice_cfg::slow::<archmage::tokens::generated::arm::NeonToken>
	add x4, x4, :lo12:linear_srgb::kernels::float::x4::srgb_to_linear_extended_slice_cfg::slow::<archmage::tokens::generated::arm::NeonToken>
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	b linear_srgb::kernels::finish_slice_cold::<archmage::tokens::generated::arm::NeonToken, 4>
.LBB20_9:
	.cfi_restore_state
	.cfi_remember_state
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB20_2
.LBB20_10:
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	b linear_srgb::simd::srgb_to_linear_extended_slice_tier_scalar_t
.LBB20_11:
	.cfi_restore_state
	adrp x4, linear_srgb::kernels::float::x4::srgb_to_linear_extended_slice_cfg::slow::<archmage::tokens::generated::arm::NeonToken>
	add x4, x4, :lo12:linear_srgb::kernels::float::x4::srgb_to_linear_extended_slice_cfg::slow::<archmage::tokens::generated::arm::NeonToken>
	mov w0, #4
	mov x1, xzr
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	b linear_srgb::kernels::finish_slice_cold::<archmage::tokens::generated::arm::NeonToken, 4>
