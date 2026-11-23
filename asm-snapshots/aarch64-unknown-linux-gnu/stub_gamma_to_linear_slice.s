.section .text.stub_gamma_to_linear_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_slice
	.p2align	2
.type	stub_gamma_to_linear_slice,@function
stub_gamma_to_linear_slice:
	.cfi_startproc
	stp d15, d14, [sp, #-96]!
	.cfi_def_cfa_offset 96
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	stp x29, x30, [sp, #64]
	stp x20, x19, [sp, #80]
	add x29, sp, #64
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
	.cfi_offset b14, -88
	.cfi_offset b15, -96
	.cfi_remember_state
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB14_16
	cmp w8, #2
	b.ne .LBB14_15
.LBB14_2:
	fcmp s0, #0.0
	b.lt .LBB14_17
	fmov s1, #4.00000000
	fcmp s0, s1
	b.hi .LBB14_17
	fcmp s0, #0.0
	and x8, x1, #0x1ffffffffffffffc
	lsl x9, x1, #2
	add x8, x0, x8, lsl #2
	and x1, x1, #0x3
	and x9, x9, #0x7ffffffffffffff0
	b.ne .LBB14_10
	cbz x9, .LBB14_8
	fmov v0.4s, #1.00000000
.LBB14_7:
	ldr q1, [x0]
	subs x9, x9, #16
	fcmgt v2.4s, v1.4s, #0.0
	fcmeq v3.4s, v1.4s, v1.4s
	bic v1.16b, v1.16b, v3.16b
	and v2.16b, v2.16b, v0.16b
	orr v1.16b, v2.16b, v1.16b
	str q1, [x0], #16
	b.ne .LBB14_7
.LBB14_8:
	cbz x1, .LBB14_14
	adrp x2, linear_srgb::tf::math::gamma_slice_x4_cfg::tail_zero::<archmage::tokens::generated::arm::NeonToken>
	add x2, x2, :lo12:linear_srgb::tf::math::gamma_slice_x4_cfg::tail_zero::<archmage::tokens::generated::arm::NeonToken>
	mov x0, x8
	.cfi_def_cfa wsp, 96
	ldp x20, x19, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #96
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
	.cfi_restore b14
	.cfi_restore b15
	b linear_srgb::kernels::finish_tail_cold::<archmage::tokens::generated::arm::NeonToken, 4>
.LBB14_10:
	.cfi_restore_state
	.cfi_remember_state
	cbz x9, .LBB14_13
	fcmp s0, s1
	mov w10, #8388607
	movi d24, #0000000000000000
	fmov s25, #1.00000000
	mvni v28.4s, #127, msl #16
	fmov v2.4s, #1.00000000
	movi v3.4s, #127, msl #16
	fmov v5.4s, #-1.00000000
	fcsel s23, s1, s0, gt
	movi v1.4s, #128, lsl #16
	fneg v28.4s, v28.4s
	fmov w11, s23
	sub w11, w11, #1
	cmp w11, w10
	mov w10, #64269
	mov w11, #64269
	movk w10, #74, lsl #16
	movk w11, #16330, lsl #16
	fcsel s23, s24, s23, lo
	dup v4.4s, w10
	mov w10, #1267
	dup v6.4s, w11
	movk w10, #16181, lsl #16
	mov w11, #27455
	fcsel s29, s25, s24, lo
	dup v7.4s, w10
	mov w10, #46233
	movk w11, #16094, lsl #16
	movk w10, #16147, lsl #16
	dup v16.4s, w11
	mov w11, #14483
	dup v17.4s, w10
	mov w10, #43579
	movk w11, #16246, lsl #16
	movk w10, #16440, lsl #16
	dup v18.4s, w11
	mov w11, #-1023672320
	dup v19.4s, w10
	mov w10, #1123942400
	dup v20.4s, w11
	mov w11, #63802
	dup v21.4s, w10
	mov w10, #50297
	movk w11, #14625, lsl #16
	movk w10, #15022, lsl #16
	movi v25.4s, #67, lsl #24
	dup v22.4s, w11
	dup v24.4s, w10
	mov w10, #38156
	mov w11, #22716
	movk w10, #15389, lsl #16
	dup v29.4s, v29.s[0]
	movk w11, #15715, lsl #16
	dup v26.4s, w10
	mov w10, #65005
	dup v27.4s, w11
	mov w11, #29208
	movk w10, #15989, lsl #16
	movk w11, #16177, lsl #16
	dup v30.4s, w10
	dup v31.4s, w11
.LBB14_12:
	ldr q8, [x0]
	mov v13.16b, v17.16b
	mov v15.16b, v18.16b
	mvni v14.4s, #126
	subs x9, x9, #16
	fcmge v9.4s, v8.4s, v1.4s
	and v9.16b, v9.16b, v8.16b
	fminnm v9.4s, v9.4s, v2.4s
	cmeq v10.4s, v9.4s, #0
	add v9.4s, v9.4s, v4.4s
	bit v9.16b, v6.16b, v10.16b
	and v11.16b, v9.16b, v3.16b
	ssra v14.4s, v9.4s, #23
	mov v9.16b, v19.16b
	add v11.4s, v11.4s, v7.4s
	fadd v12.4s, v11.4s, v5.4s
	fadd v11.4s, v11.4s, v2.4s
	fdiv v11.4s, v12.4s, v11.4s
	fmul v12.4s, v11.4s, v11.4s
	fmla v13.4s, v16.4s, v12.4s
	fmla v15.4s, v12.4s, v13.4s
	mov v13.16b, v24.16b
	fmla v9.4s, v12.4s, v15.4s
	scvtf v12.4s, v14.4s
	mov v14.16b, v26.16b
	fmla v12.4s, v11.4s, v9.4s
	fmul v9.4s, v12.4s, v23.s[0]
	fmaxnm v11.4s, v9.4s, v20.4s
	fcmge v15.4s, v9.4s, v25.4s
	fcmgt v9.4s, v20.4s, v9.4s
	fminnm v11.4s, v11.4s, v25.4s
	frintn v12.4s, v11.4s
	fminnm v12.4s, v12.4s, v21.4s
	fsub v11.4s, v11.4s, v12.4s
	fcvtns v12.4s, v12.4s
	fmla v13.4s, v22.4s, v11.4s
	shl v12.4s, v12.4s, #23
	fmla v14.4s, v11.4s, v13.4s
	mov v13.16b, v27.16b
	fmla v13.4s, v11.4s, v14.4s
	mov v14.16b, v30.16b
	fmla v14.4s, v11.4s, v13.4s
	mov v13.16b, v31.16b
	fmla v13.4s, v11.4s, v14.4s
	fmov v14.4s, #1.00000000
	fmla v14.4s, v11.4s, v13.4s
	add v11.4s, v12.4s, v2.4s
	mvn v12.16b, v15.16b
	bic v9.16b, v12.16b, v9.16b
	fmul v11.4s, v14.4s, v11.4s
	and v9.16b, v9.16b, v11.16b
	and v11.16b, v15.16b, v28.16b
	orr v9.16b, v9.16b, v11.16b
	bit v9.16b, v29.16b, v10.16b
	fcmeq v10.4s, v8.4s, v8.4s
	bit v8.16b, v9.16b, v10.16b
	str q8, [x0], #16
	b.ne .LBB14_12
.LBB14_13:
	cbnz x1, .LBB14_18
.LBB14_14:
	.cfi_def_cfa wsp, 96
	ldp x20, x19, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #96
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
	.cfi_restore b14
	.cfi_restore b15
	ret
.LBB14_15:
	.cfi_restore_state
	.cfi_remember_state
	mov x19, x0
	fmov s8, s0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	fmov s0, s8
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB14_2
.LBB14_16:
	.cfi_def_cfa wsp, 96
	ldp x20, x19, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #96
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
	.cfi_restore b14
	.cfi_restore b15
	b linear_srgb::simd::gamma_to_linear_slice_tier_scalar_t
.LBB14_17:
	.cfi_restore_state
	.cfi_remember_state
	.cfi_def_cfa wsp, 96
	ldp x20, x19, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #96
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
	.cfi_restore b14
	.cfi_restore b15
	b linear_srgb::tf::math::gamma_outside_fit_slice
.LBB14_18:
	.cfi_restore_state
	mov x0, x8
	.cfi_def_cfa wsp, 96
	ldp x20, x19, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #96
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
	.cfi_restore b14
	.cfi_restore b15
	b linear_srgb::kernels::finish_tail_cold::<(archmage::tokens::generated::arm::NeonToken, f32), 4>
