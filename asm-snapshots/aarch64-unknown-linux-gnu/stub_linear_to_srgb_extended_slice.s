.section .text.stub_linear_to_srgb_extended_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_extended_slice
	.p2align	2
.type	stub_linear_to_srgb_extended_slice,@function
stub_linear_to_srgb_extended_slice:
	.cfi_startproc
	sub sp, sp, #352
	.cfi_def_cfa_offset 352
	stp x29, x30, [sp, #272]
	str x28, [sp, #288]
	stp x24, x23, [sp, #304]
	stp x22, x21, [sp, #320]
	stp x20, x19, [sp, #336]
	add x29, sp, #272
	.cfi_def_cfa w29, 80
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w28, -64
	.cfi_offset w30, -72
	.cfi_offset w29, -80
	.cfi_remember_state
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB16_10
	cmp w8, #2
	b.ne .LBB16_9
.LBB16_2:
	lsl x8, x1, #2
	and x19, x1, #0x3
	ands x21, x8, #0x7ffffffffffffff0
	b.eq .LBB16_7
	mov w8, #47186
	mov w9, #59837
	mvni v30.4s, #128, lsl #24
	movk w8, #16718, lsl #16
	movk w9, #17200, lsl #16
	mov x20, x0
	dup v7.4s, w8
	mov w8, #60277
	dup v16.4s, w9
	movk w8, #17737, lsl #16
	mov w9, #21186
	mov x22, x0
	dup v17.4s, w8
	mov w8, #44370
	movk w9, #17949, lsl #16
	movk w8, #17885, lsl #16
	dup v18.4s, w9
	mov w9, #30049
	dup v19.4s, w8
	mov w8, #26919
	movk w9, #48797, lsl #16
	movk w8, #17536, lsl #16
	dup v21.4s, w9
	mov w9, #60486
	dup v20.4s, w8
	mov w8, #17028
	movk w9, #17793, lsl #16
	movk w8, #49027, lsl #16
	dup v24.4s, w9
	mov w9, #46319
	dup v22.4s, w8
	mov w8, #63323
	movk w9, #17487, lsl #16
	movk w8, #17310, lsl #16
	dup v27.4s, w9
	mov w9, #1115684864
	dup v23.4s, w8
	mov w8, #4173
	dup v31.4s, w9
	movk w8, #17952, lsl #16
	dup v25.4s, w8
	mov w8, #18007
	movk w8, #17852, lsl #16
	dup v26.4s, w8
	mov w8, #12900
	movk w8, #16798, lsl #16
	dup v28.4s, w8
	mov w8, #20545
	movk w8, #15175, lsl #16
	dup v29.4s, w8
.LBB16_4:
	ldr q0, [x22], #16
	mov v3.16b, v17.16b
	mov v5.16b, v18.16b
	mov v6.16b, v24.16b
	fabs v1.4s, v0.4s
	fsqrt v2.4s, v1.4s
	fmla v3.4s, v16.4s, v2.4s
	fadd v4.4s, v2.4s, v23.4s
	fmla v5.4s, v2.4s, v3.4s
	fmla v6.4s, v2.4s, v4.4s
	mov v3.16b, v19.16b
	mov v4.16b, v25.16b
	fmla v3.4s, v2.4s, v5.4s
	fmla v4.4s, v2.4s, v6.4s
	mov v5.16b, v20.16b
	mov v6.16b, v26.16b
	fmla v5.4s, v2.4s, v3.4s
	fmla v6.4s, v2.4s, v4.4s
	mov v3.16b, v21.16b
	mov v4.16b, v27.16b
	fmla v3.4s, v2.4s, v5.4s
	fmla v4.4s, v2.4s, v6.4s
	mov v5.16b, v22.16b
	mov v6.16b, v28.16b
	fmla v5.4s, v2.4s, v3.4s
	fmla v6.4s, v2.4s, v4.4s
	fmul v3.4s, v1.4s, v7.4s
	fcmgt v4.4s, v29.4s, v1.4s
	fcmgt v1.4s, v1.4s, v31.4s
	fdiv v2.4s, v5.4s, v6.4s
	addp d1, v1.2d
	fmov x8, d1
	bit v2.16b, v3.16b, v4.16b
	bit v0.16b, v2.16b, v30.16b
	cbnz x8, .LBB16_6
	str q0, [x20]
	subs x21, x21, #16
	mov x20, x22
	b.ne .LBB16_4
	b .LBB16_7
.LBB16_6:
	mov x23, x1
	sub x1, x29, #16
	mov x24, x0
	mov x0, x20
	stp q21, q20, [x29, #-128]
	stp q19, q18, [x29, #-96]
	stp q17, q16, [x29, #-64]
	stp q7, q0, [x29, #-32]
	stp q29, q28, [sp, #16]
	stp q27, q26, [sp, #48]
	stp q25, q24, [sp, #80]
	stp q23, q22, [sp, #112]
	str q31, [sp]
	bl linear_srgb::kernels::float::linear_to_srgb_extended_patch::<archmage::tokens::generated::arm::NeonToken, 4>
	ldp q31, q29, [sp]
	mvni v30.4s, #128, lsl #24
	ldp q28, q27, [sp, #32]
	mov x0, x24
	ldp q26, q25, [sp, #64]
	mov x1, x23
	ldp q24, q23, [sp, #96]
	ldr q22, [sp, #128]
	ldp q21, q20, [x29, #-128]
	ldp q19, q18, [x29, #-96]
	ldp q17, q16, [x29, #-64]
	ldp q7, q0, [x29, #-32]
	str q0, [x20]
	subs x21, x21, #16
	mov x20, x22
	b.ne .LBB16_4
.LBB16_7:
	cbnz x19, .LBB16_11
	.cfi_def_cfa wsp, 352
	ldp x20, x19, [sp, #336]
	ldr x28, [sp, #288]
	ldp x22, x21, [sp, #320]
	ldp x24, x23, [sp, #304]
	ldp x29, x30, [sp, #272]
	add sp, sp, #352
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w28
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB16_9:
	.cfi_restore_state
	.cfi_remember_state
	mov x20, x1
	mov x19, x0
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB16_2
.LBB16_10:
	.cfi_def_cfa wsp, 352
	ldp x20, x19, [sp, #336]
	ldr x28, [sp, #288]
	ldp x22, x21, [sp, #320]
	ldp x24, x23, [sp, #304]
	ldp x29, x30, [sp, #272]
	add sp, sp, #352
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w28
	.cfi_restore w30
	.cfi_restore w29
	b linear_srgb::simd::linear_to_srgb_extended_slice_tier_scalar_t
.LBB16_11:
	.cfi_restore_state
	and x8, x1, #0x1ffffffffffffffc
	adrp x2, linear_srgb::kernels::float::x4::linear_to_srgb_extended_slice_cfg::tail::<archmage::tokens::generated::arm::NeonToken>
	add x2, x2, :lo12:linear_srgb::kernels::float::x4::linear_to_srgb_extended_slice_cfg::tail::<archmage::tokens::generated::arm::NeonToken>
	add x0, x0, x8, lsl #2
	mov x1, x19
	.cfi_def_cfa wsp, 352
	ldp x20, x19, [sp, #336]
	ldr x28, [sp, #288]
	ldp x22, x21, [sp, #320]
	ldp x24, x23, [sp, #304]
	ldp x29, x30, [sp, #272]
	add sp, sp, #352
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w28
	.cfi_restore w30
	.cfi_restore w29
	b linear_srgb::kernels::finish_tail_cold::<archmage::tokens::generated::arm::NeonToken, 4>
