.section .text.stub_gamma_to_linear_premultiply_rgba_slice,"ax",@progbits
	.globl	stub_gamma_to_linear_premultiply_rgba_slice
	.p2align	2
.type	stub_gamma_to_linear_premultiply_rgba_slice,@function
stub_gamma_to_linear_premultiply_rgba_slice:
	.cfi_startproc
	sub sp, sp, #144
	.cfi_def_cfa_offset 144
	str d14, [sp, #48]
	stp d13, d12, [sp, #64]
	stp d11, d10, [sp, #80]
	stp d9, d8, [sp, #96]
	stp x29, x30, [sp, #112]
	stp x20, x19, [sp, #128]
	add x29, sp, #112
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
	.cfi_offset b14, -96
	.cfi_remember_state
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB13_15
	cmp w8, #2
	b.ne .LBB13_14
.LBB13_2:
	fcmp s0, #0.0
	b.lt .LBB13_8
	fmov s1, #4.00000000
	fcmp s0, s1
	b.hi .LBB13_8
	fcmp s0, #0.0
	lsl x8, x1, #2
	and x8, x8, #0x7ffffffffffffff0
	b.ne .LBB13_10
	cbz x8, .LBB13_13
	fmov v0.4s, #1.00000000
.LBB13_7:
	ldr q1, [x0]
	subs x8, x8, #16
	fcmgt v2.4s, v1.4s, #0.0
	fcmeq v3.4s, v1.4s, v1.4s
	bic v3.16b, v1.16b, v3.16b
	and v2.16b, v2.16b, v0.16b
	orr v2.16b, v2.16b, v3.16b
	fmul v2.4s, v2.4s, v1.s[3]
	mov v2.s[3], v1.s[3]
	str q2, [x0], #16
	b.ne .LBB13_7
	b .LBB13_13
.LBB13_8:
	lsl x8, x1, #2
	ands x19, x8, #0x7ffffffffffffff0
	b.eq .LBB13_13
.LBB13_9:
	ldr q1, [x0]
	add x8, sp, #16
	mov x20, x0
	add x0, sp, #32
	fmov s8, s0
	str q1, [sp]
	str q1, [sp, #32]
	bl linear_srgb::tf::math::gamma_outside_fit::<4>
	ldp q2, q1, [sp]
	fmov s0, s8
	mov x0, x20
	subs x19, x19, #16
	fmul v1.4s, v1.4s, v2.s[3]
	mov v1.s[3], v2.s[3]
	str q1, [x0], #16
	b.ne .LBB13_9
	b .LBB13_13
.LBB13_10:
	cbz x8, .LBB13_13
	fcmp s0, s1
	mov w9, #8388607
	movi d27, #0000000000000000
	fmov s23, #1.00000000
	movi v2.4s, #127, msl #16
	fmov v4.4s, #-1.00000000
	movi v17.4s, #67, lsl #24
	fcsel s26, s1, s0, gt
	movi v0.4s, #128, lsl #16
	fmov v1.4s, #1.00000000
	fmov w10, s26
	sub w10, w10, #1
	cmp w10, w9
	mov w9, #64269
	mov w10, #64269
	movk w9, #74, lsl #16
	movk w10, #16330, lsl #16
	fcsel s28, s23, s27, lo
	dup v3.4s, w9
	mov w9, #1267
	dup v5.4s, w10
	movk w9, #16181, lsl #16
	mov w10, #27455
	fcsel s26, s27, s26, lo
	dup v6.4s, w9
	mov w9, #46233
	movk w10, #16094, lsl #16
	movk w9, #16147, lsl #16
	dup v7.4s, w10
	mov w10, #14483
	dup v16.4s, w9
	mov w9, #43579
	movk w10, #16246, lsl #16
	movk w9, #16440, lsl #16
	dup v18.4s, w10
	mov w10, #-1023672320
	dup v19.4s, w9
	mov w9, #1123942400
	dup v20.4s, w10
	mov w10, #63802
	dup v21.4s, w9
	mov w9, #50297
	movk w10, #14625, lsl #16
	movk w9, #15022, lsl #16
	dup v27.4s, v28.s[0]
	dup v22.4s, w10
	mov w10, #38156
	dup v23.4s, w9
	mov w9, #22716
	movk w10, #15389, lsl #16
	movk w9, #15715, lsl #16
	dup v24.4s, w10
	mov w10, #65005
	dup v25.4s, w9
	mov w9, #29208
	movk w10, #15989, lsl #16
	movk w9, #16177, lsl #16
	dup v28.4s, w10
	dup v29.4s, w9
	adrp x9, .LCPI13_0
	ldr q30, [x9, :lo12:.LCPI13_0]
.LBB13_12:
	ldr q31, [x0]
	mov v12.16b, v16.16b
	mov v14.16b, v18.16b
	mvni v13.4s, #126
	subs x8, x8, #16
	fcmge v8.4s, v31.4s, v0.4s
	and v8.16b, v8.16b, v31.16b
	fminnm v8.4s, v8.4s, v1.4s
	cmeq v9.4s, v8.4s, #0
	add v8.4s, v8.4s, v3.4s
	bit v8.16b, v5.16b, v9.16b
	and v10.16b, v8.16b, v2.16b
	ssra v13.4s, v8.4s, #23
	mov v8.16b, v19.16b
	add v10.4s, v10.4s, v6.4s
	fadd v11.4s, v10.4s, v4.4s
	fadd v10.4s, v10.4s, v1.4s
	fdiv v10.4s, v11.4s, v10.4s
	fmul v11.4s, v10.4s, v10.4s
	fmla v12.4s, v7.4s, v11.4s
	fmla v14.4s, v11.4s, v12.4s
	mov v12.16b, v23.16b
	fmla v8.4s, v11.4s, v14.4s
	scvtf v11.4s, v13.4s
	mov v13.16b, v24.16b
	fmla v11.4s, v10.4s, v8.4s
	fmul v8.4s, v11.4s, v26.s[0]
	fmaxnm v10.4s, v8.4s, v20.4s
	fcmge v14.4s, v8.4s, v17.4s
	fcmgt v8.4s, v20.4s, v8.4s
	fminnm v10.4s, v10.4s, v17.4s
	frintn v11.4s, v10.4s
	fminnm v11.4s, v11.4s, v21.4s
	fsub v10.4s, v10.4s, v11.4s
	fcvtns v11.4s, v11.4s
	fmla v12.4s, v22.4s, v10.4s
	shl v11.4s, v11.4s, #23
	fmla v13.4s, v10.4s, v12.4s
	mov v12.16b, v25.16b
	fmla v12.4s, v10.4s, v13.4s
	mov v13.16b, v28.16b
	fmla v13.4s, v10.4s, v12.4s
	mov v12.16b, v29.16b
	fmla v12.4s, v10.4s, v13.4s
	fmov v13.4s, #1.00000000
	fmla v13.4s, v10.4s, v12.4s
	add v10.4s, v11.4s, v1.4s
	mvn v11.16b, v14.16b
	bic v8.16b, v11.16b, v8.16b
	fmul v10.4s, v13.4s, v10.4s
	and v8.16b, v8.16b, v10.16b
	and v10.16b, v14.16b, v30.16b
	orr v8.16b, v8.16b, v10.16b
	bit v8.16b, v27.16b, v9.16b
	fcmeq v9.4s, v31.4s, v31.4s
	bif v8.16b, v31.16b, v9.16b
	fmul v8.4s, v8.4s, v31.s[3]
	mov v8.s[3], v31.s[3]
	str q8, [x0], #16
	b.ne .LBB13_12
.LBB13_13:
	.cfi_def_cfa wsp, 144
	ldp x20, x19, [sp, #128]
	ldr d14, [sp, #48]
	ldp x29, x30, [sp, #112]
	ldp d9, d8, [sp, #96]
	ldp d11, d10, [sp, #80]
	ldp d13, d12, [sp, #64]
	add sp, sp, #144
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
	ret
.LBB13_14:
	.cfi_restore_state
	mov x19, x0
	fmov s8, s0
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	fmov s0, s8
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB13_2
.LBB13_15:
	.cfi_def_cfa wsp, 144
	ldp x20, x19, [sp, #128]
	ldr d14, [sp, #48]
	ldp x29, x30, [sp, #112]
	ldp d9, d8, [sp, #96]
	ldp d11, d10, [sp, #80]
	ldp d13, d12, [sp, #64]
	add sp, sp, #144
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
	b linear_srgb::simd::gamma_to_linear_premultiply_rgba_slice_tier_scalar
