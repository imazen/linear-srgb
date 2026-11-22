.section .text.stub_linear_to_srgb_extended_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_extended_slice
	.p2align	2
.type	stub_linear_to_srgb_extended_slice,@function
stub_linear_to_srgb_extended_slice:
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
	b.eq .LBB17_10
	cmp w8, #2
	b.ne .LBB17_9
.LBB17_2:
	and x8, x1, #0x1ffffffffffffffc
	mov x9, x1
	lsr x1, x1, #2
	add x2, x0, x8, lsl #2
	and x3, x9, #0x3
	cbz x1, .LBB17_6
	mov w8, #1115684864
	mov w9, #20545
	mvni v24.4s, #128, lsl #24
	dup v0.4s, w8
	mov w8, #59837
	movk w9, #15175, lsl #16
	movk w8, #17200, lsl #16
	dup v1.4s, w9
	mov w9, #60277
	dup v2.4s, w8
	mov w8, #21186
	movk w9, #17737, lsl #16
	movk w8, #17949, lsl #16
	dup v3.4s, w9
	mov w9, #26919
	dup v4.4s, w8
	mov w8, #44370
	movk w9, #17536, lsl #16
	movk w8, #17885, lsl #16
	dup v6.4s, w9
	mov w9, #63323
	dup v5.4s, w8
	mov w8, #30049
	movk w9, #17310, lsl #16
	movk w8, #48797, lsl #16
	dup v17.4s, w9
	mov w9, #18007
	dup v7.4s, w8
	mov w8, #17028
	movk w9, #17852, lsl #16
	movk w8, #49027, lsl #16
	dup v20.4s, w9
	mov w9, #47186
	dup v16.4s, w8
	mov w8, #60486
	movk w9, #16718, lsl #16
	movk w8, #17793, lsl #16
	dup v23.4s, w9
	dup v18.4s, w8
	mov w8, #4173
	movk w8, #17952, lsl #16
	dup v19.4s, w8
	mov w8, #46319
	movk w8, #17487, lsl #16
	dup v21.4s, w8
	mov w8, #12900
	movk w8, #16798, lsl #16
	dup v22.4s, w8
.LBB17_4:
	ldr q25, [x0]
	fabs v26.4s, v25.4s
	fcmgt v27.4s, v26.4s, v0.4s
	addp d27, v27.2d
	fmov x8, d27
	cbnz x8, .LBB17_8
	fsqrt v27.4s, v26.4s
	mov v28.16b, v3.16b
	mov v30.16b, v4.16b
	mov v31.16b, v18.16b
	subs x1, x1, #1
	fmla v28.4s, v2.4s, v27.4s
	fadd v29.4s, v27.4s, v17.4s
	fmla v30.4s, v27.4s, v28.4s
	fmla v31.4s, v27.4s, v29.4s
	mov v28.16b, v5.16b
	mov v29.16b, v19.16b
	fmla v28.4s, v27.4s, v30.4s
	fmla v29.4s, v27.4s, v31.4s
	mov v30.16b, v6.16b
	mov v31.16b, v20.16b
	fmla v30.4s, v27.4s, v28.4s
	fmla v31.4s, v27.4s, v29.4s
	mov v28.16b, v7.16b
	mov v29.16b, v21.16b
	fmla v28.4s, v27.4s, v30.4s
	fmla v29.4s, v27.4s, v31.4s
	mov v30.16b, v16.16b
	mov v31.16b, v22.16b
	fmla v30.4s, v27.4s, v28.4s
	fmla v31.4s, v27.4s, v29.4s
	fcmgt v28.4s, v1.4s, v26.4s
	fmul v26.4s, v26.4s, v23.4s
	fdiv v27.4s, v30.4s, v31.4s
	bif v26.16b, v27.16b, v28.16b
	bit v25.16b, v26.16b, v24.16b
	str q25, [x0], #16
	b.ne .LBB17_4
.LBB17_6:
	cbnz x3, .LBB17_11
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB17_8:
	.cfi_restore_state
	.cfi_remember_state
	adrp x4, linear_srgb::kernels::float::x4::linear_to_srgb_extended_slice_cfg::slow::<archmage::tokens::generated::arm::NeonToken>
	add x4, x4, :lo12:linear_srgb::kernels::float::x4::linear_to_srgb_extended_slice_cfg::slow::<archmage::tokens::generated::arm::NeonToken>
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	b linear_srgb::kernels::finish_slice_cold::<archmage::tokens::generated::arm::NeonToken, 4>
.LBB17_9:
	.cfi_restore_state
	.cfi_remember_state
	mov x19, x0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB17_2
.LBB17_10:
	.cfi_def_cfa wsp, 32
	ldp x20, x19, [sp, #16]
	ldp x29, x30, [sp], #32
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	b linear_srgb::simd::linear_to_srgb_extended_slice_tier_scalar_t
.LBB17_11:
	.cfi_restore_state
	adrp x4, linear_srgb::kernels::float::x4::linear_to_srgb_extended_slice_cfg::slow::<archmage::tokens::generated::arm::NeonToken>
	add x4, x4, :lo12:linear_srgb::kernels::float::x4::linear_to_srgb_extended_slice_cfg::slow::<archmage::tokens::generated::arm::NeonToken>
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
