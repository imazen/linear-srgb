.section .text.stub_srgb_to_linear_extended_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_extended_slice
	.p2align	2
.type	stub_srgb_to_linear_extended_slice,@function
stub_srgb_to_linear_extended_slice:
	.cfi_startproc
	sub sp, sp, #336
	.cfi_def_cfa_offset 336
	stp x29, x30, [sp, #256]
	str x28, [sp, #272]
	stp x24, x23, [sp, #288]
	stp x22, x21, [sp, #304]
	stp x20, x19, [sp, #320]
	add x29, sp, #256
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
	b.eq .LBB19_10
	cmp w8, #2
	b.ne .LBB19_9
.LBB19_2:
	lsl x8, x1, #2
	and x19, x1, #0x3
	ands x21, x8, #0x7ffffffffffffff0
	b.eq .LBB19_7
	mov w8, #33681
	mov w9, #21091
	mov w10, #45243
	movk w8, #15774, lsl #16
	movk w9, #18053, lsl #16
	movk w10, #18434, lsl #16
	dup v6.4s, w8
	mov w8, #54856
	dup v7.4s, w9
	movk w8, #18518, lsl #16
	mov w9, #26646
	dup v16.4s, w10
	dup v17.4s, w8
	mov w8, #22505
	movk w9, #18037, lsl #16
	movk w8, #18375, lsl #16
	mov w10, #49826
	dup v19.4s, w9
	dup v18.4s, w8
	mov w8, #11201
	mov w9, #23593
	movk w10, #17507, lsl #16
	movk w8, #16784, lsl #16
	movk w9, #49790, lsl #16
	dup v20.4s, w10
	mov w10, #51710
	dup v21.4s, w8
	dup v22.4s, w9
	mov w8, #60686
	mov w9, #61050
	movk w10, #17803, lsl #16
	movk w8, #18336, lsl #16
	movk w9, #18528, lsl #16
	dup v23.4s, w10
	mov w10, #20961
	dup v24.4s, w8
	dup v25.4s, w9
	mov w8, #46089
	mov w9, #61974
	movk w10, #18451, lsl #16
	movk w8, #18088, lsl #16
	movk w9, #15648, lsl #16
	movi v27.4s, #65, lsl #24
	dup v26.4s, w10
	dup v28.4s, w8
	dup v29.4s, w9
	mvni v30.4s, #128, lsl #24
	mov x20, x0
	mov x22, x0
.LBB19_4:
	ldr q0, [x22], #16
	mov v2.16b, v16.16b
	mov v4.16b, v17.16b
	mov v5.16b, v23.16b
	fabs v1.4s, v0.4s
	fmla v2.4s, v7.4s, v1.4s
	fadd v3.4s, v1.4s, v22.4s
	fmla v4.4s, v1.4s, v2.4s
	fmla v5.4s, v1.4s, v3.4s
	mov v2.16b, v18.16b
	mov v3.16b, v24.16b
	fmla v2.4s, v1.4s, v4.4s
	fmla v3.4s, v1.4s, v5.4s
	mov v4.16b, v19.16b
	mov v5.16b, v25.16b
	fmla v4.4s, v1.4s, v2.4s
	fmla v5.4s, v1.4s, v3.4s
	mov v2.16b, v20.16b
	mov v3.16b, v26.16b
	fmla v2.4s, v1.4s, v4.4s
	fmla v3.4s, v1.4s, v5.4s
	mov v4.16b, v21.16b
	mov v5.16b, v28.16b
	fmla v4.4s, v1.4s, v2.4s
	fmla v5.4s, v1.4s, v3.4s
	fmul v3.4s, v1.4s, v6.4s
	fdiv v2.4s, v4.4s, v5.4s
	fcmgt v4.4s, v29.4s, v1.4s
	fcmgt v1.4s, v1.4s, v27.4s
	addp d1, v1.2d
	fmov x8, d1
	bit v2.16b, v3.16b, v4.16b
	bit v0.16b, v2.16b, v30.16b
	cbnz x8, .LBB19_6
	str q0, [x20]
	subs x21, x21, #16
	mov x20, x22
	b.ne .LBB19_4
	b .LBB19_7
.LBB19_6:
	mov x23, x1
	sub x1, x29, #16
	mov x24, x0
	mov x0, x20
	stp q18, q17, [x29, #-96]
	stp q16, q7, [x29, #-64]
	stp q6, q0, [x29, #-32]
	stur q19, [x29, #-112]
	stp q28, q26, [sp, #16]
	stp q25, q24, [sp, #48]
	stp q23, q22, [sp, #80]
	stp q21, q20, [sp, #112]
	str q29, [sp]
	bl linear_srgb::kernels::float::srgb_to_linear_extended_patch::<4>
	movi v27.4s, #65, lsl #24
	mvni v30.4s, #128, lsl #24
	ldr q20, [sp, #128]
	ldp q29, q28, [sp]
	ldur q0, [x29, #-16]
	ldp q26, q25, [sp, #32]
	mov x0, x24
	ldp q24, q23, [sp, #64]
	mov x1, x23
	ldp q22, q21, [sp, #96]
	ldp q19, q18, [x29, #-112]
	ldp q17, q16, [x29, #-80]
	ldp q7, q6, [x29, #-48]
	str q0, [x20]
	subs x21, x21, #16
	mov x20, x22
	b.ne .LBB19_4
.LBB19_7:
	cbnz x19, .LBB19_11
.LBB19_8:
	.cfi_def_cfa wsp, 336
	ldp x20, x19, [sp, #320]
	ldr x28, [sp, #272]
	ldp x22, x21, [sp, #304]
	ldp x24, x23, [sp, #288]
	ldp x29, x30, [sp, #256]
	add sp, sp, #336
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
.LBB19_9:
	.cfi_restore_state
	.cfi_remember_state
	mov x20, x1
	mov x19, x0
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB19_2
.LBB19_10:
	.cfi_def_cfa wsp, 336
	ldp x20, x19, [sp, #320]
	ldr x28, [sp, #272]
	ldp x22, x21, [sp, #304]
	ldp x24, x23, [sp, #288]
	ldp x29, x30, [sp, #256]
	add sp, sp, #336
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
	b linear_srgb::simd::srgb_to_linear_extended_slice_tier_scalar_t
.LBB19_11:
	.cfi_restore_state
	and x8, x1, #0x1ffffffffffffffc
	adrp x2, linear_srgb::kernels::float::x4::srgb_to_linear_extended_slice_cfg::tail::<archmage::tokens::generated::arm::NeonToken>
	add x2, x2, :lo12:linear_srgb::kernels::float::x4::srgb_to_linear_extended_slice_cfg::tail::<archmage::tokens::generated::arm::NeonToken>
	add x0, x0, x8, lsl #2
	mov x1, x19
	bl linear_srgb::kernels::finish_tail_cold::<archmage::tokens::generated::arm::NeonToken, 4>
	b .LBB19_8
