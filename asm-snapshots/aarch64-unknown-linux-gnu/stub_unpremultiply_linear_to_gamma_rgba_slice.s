.section .text.stub_unpremultiply_linear_to_gamma_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_gamma_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_gamma_rgba_slice,@function
stub_unpremultiply_linear_to_gamma_rgba_slice:
	.cfi_startproc
	sub sp, sp, #160
	.cfi_def_cfa_offset 160
	stp d15, d14, [sp, #64]
	stp d13, d12, [sp, #80]
	stp d11, d10, [sp, #96]
	stp d9, d8, [sp, #112]
	stp x29, x30, [sp, #128]
	stp x20, x19, [sp, #144]
	add x29, sp, #128
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
	b.eq .LBB23_15
	cmp w8, #2
	b.ne .LBB23_14
.LBB23_2:
	fmov s1, #1.00000000
	fdiv s8, s1, s0
	fmov s0, #4.00000000
	fcmp s8, #0.0
	fccmp s8, s0, #2, ge
	b.ls .LBB23_6
	lsl x8, x1, #2
	ands x19, x8, #0x7ffffffffffffff0
	b.eq .LBB23_13
	mov w8, #981467136
	dup v0.4s, w8
	str q0, [sp]
.LBB23_5:
	ldr q3, [x0]
	fmov v1.4s, #1.00000000
	ldr q2, [sp]
	add x8, sp, #32
	mov x20, x0
	add x0, sp, #48
	dup v0.4s, v3.s[3]
	str q3, [sp, #16]
	fdiv v1.4s, v1.4s, v0.4s
	fcmgt v0.4s, v0.4s, v2.4s
	and v0.16b, v0.16b, v1.16b
	fmul v0.4s, v3.4s, v0.4s
	str q0, [sp, #48]
	fmov s0, s8
	bl linear_srgb::tf::math::gamma_outside_fit::<4>
	ldp q1, q0, [sp, #16]
	mov x0, x20
	subs x19, x19, #16
	mov v0.s[3], v1.s[3]
	str q0, [x0], #16
	b.ne .LBB23_5
	b .LBB23_13
.LBB23_6:
	fcmp s8, #0.0
	lsl x8, x1, #2
	and x8, x8, #0x7ffffffffffffff0
	b.ne .LBB23_10
	cbz x8, .LBB23_13
	fmov v0.4s, #1.00000000
	mov w9, #981467136
	dup v1.4s, w9
.LBB23_9:
	ldr q2, [x0]
	subs x8, x8, #16
	dup v3.4s, v2.s[3]
	fdiv v4.4s, v0.4s, v3.4s
	fcmgt v3.4s, v3.4s, v1.4s
	and v3.16b, v3.16b, v4.16b
	fmul v3.4s, v2.4s, v3.4s
	fcmgt v4.4s, v3.4s, #0.0
	fcmeq v5.4s, v3.4s, v3.4s
	bic v3.16b, v3.16b, v5.16b
	and v4.16b, v4.16b, v0.16b
	orr v3.16b, v4.16b, v3.16b
	mov v3.s[3], v2.s[3]
	str q3, [x0], #16
	b.ne .LBB23_9
	b .LBB23_13
.LBB23_10:
	cbz x8, .LBB23_13
	fcmp s8, s0
	mov w9, #8388607
	movi d28, #0000000000000000
	fmov s24, #1.00000000
	movi v3.4s, #127, msl #16
	fmov v5.4s, #-1.00000000
	movi v18.4s, #67, lsl #24
	fcsel s27, s0, s8, gt
	fmov v0.4s, #1.00000000
	fmov w10, s27
	sub w10, w10, #1
	cmp w10, w9
	mov w10, #64269
	mov w9, #981467136
	movk w10, #74, lsl #16
	dup v2.4s, w9
	mov w9, #64269
	dup v4.4s, w10
	mov w10, #1267
	movk w9, #16330, lsl #16
	movk w10, #16181, lsl #16
	dup v6.4s, w9
	mov w9, #27455
	dup v7.4s, w10
	mov w10, #46233
	movk w9, #16094, lsl #16
	movk w10, #16147, lsl #16
	dup v16.4s, w9
	mov w9, #14483
	dup v17.4s, w10
	mov w10, #43579
	movk w9, #16246, lsl #16
	movk w10, #16440, lsl #16
	dup v1.4s, w9
	mov w9, #-1023672320
	dup v20.4s, w10
	mov w10, #1123942400
	dup v21.4s, w9
	mov w9, #63802
	dup v22.4s, w10
	mov w10, #50297
	movk w9, #14625, lsl #16
	movk w10, #15022, lsl #16
	fcsel s29, s24, s28, lo
	dup v23.4s, w9
	mov w9, #38156
	dup v24.4s, w10
	mov w10, #22716
	movk w9, #15389, lsl #16
	fcsel s27, s28, s27, lo
	movk w10, #15715, lsl #16
	dup v25.4s, w9
	mov w9, #65005
	dup v26.4s, w10
	mov w10, #29208
	movk w9, #15989, lsl #16
	movk w10, #16177, lsl #16
	dup v28.4s, v29.s[0]
	dup v29.4s, w9
	dup v30.4s, w10
	adrp x9, .LCPI23_0
	ldr q31, [x9, :lo12:.LCPI23_0]
.LBB23_12:
	ldr q8, [x0]
	movi v19.4s, #128, lsl #16
	mov v14.16b, v17.16b
	mvni v15.4s, #126
	subs x8, x8, #16
	dup v9.4s, v8.s[3]
	fdiv v10.4s, v0.4s, v9.4s
	fcmgt v9.4s, v9.4s, v2.4s
	and v9.16b, v9.16b, v10.16b
	fmul v9.4s, v8.4s, v9.4s
	fcmge v10.4s, v9.4s, v19.4s
	mov v19.16b, v1.16b
	and v10.16b, v10.16b, v9.16b
	fminnm v10.4s, v10.4s, v0.4s
	cmeq v11.4s, v10.4s, #0
	add v10.4s, v10.4s, v4.4s
	bit v10.16b, v6.16b, v11.16b
	and v12.16b, v10.16b, v3.16b
	ssra v15.4s, v10.4s, #23
	mov v10.16b, v20.16b
	add v12.4s, v12.4s, v7.4s
	fadd v13.4s, v12.4s, v5.4s
	fadd v12.4s, v12.4s, v0.4s
	fdiv v12.4s, v13.4s, v12.4s
	fmul v13.4s, v12.4s, v12.4s
	fmla v14.4s, v16.4s, v13.4s
	fmla v19.4s, v13.4s, v14.4s
	mov v14.16b, v25.16b
	fmla v10.4s, v13.4s, v19.4s
	scvtf v19.4s, v15.4s
	mov v13.16b, v24.16b
	fmla v19.4s, v12.4s, v10.4s
	fmul v19.4s, v19.4s, v27.s[0]
	fmaxnm v10.4s, v19.4s, v21.4s
	fcmge v15.4s, v19.4s, v18.4s
	fcmgt v19.4s, v21.4s, v19.4s
	fminnm v10.4s, v10.4s, v18.4s
	frintn v12.4s, v10.4s
	fminnm v12.4s, v12.4s, v22.4s
	fsub v10.4s, v10.4s, v12.4s
	fcvtns v12.4s, v12.4s
	fmla v13.4s, v23.4s, v10.4s
	shl v12.4s, v12.4s, #23
	fmla v14.4s, v10.4s, v13.4s
	mov v13.16b, v26.16b
	fmla v13.4s, v10.4s, v14.4s
	mov v14.16b, v29.16b
	fmla v14.4s, v10.4s, v13.4s
	mov v13.16b, v30.16b
	fmla v13.4s, v10.4s, v14.4s
	fmov v14.4s, #1.00000000
	fmla v14.4s, v10.4s, v13.4s
	add v10.4s, v12.4s, v0.4s
	mvn v12.16b, v15.16b
	bic v19.16b, v12.16b, v19.16b
	fmul v10.4s, v14.4s, v10.4s
	and v19.16b, v19.16b, v10.16b
	and v10.16b, v15.16b, v31.16b
	orr v19.16b, v19.16b, v10.16b
	fcmeq v10.4s, v9.4s, v9.4s
	bit v19.16b, v28.16b, v11.16b
	bif v19.16b, v9.16b, v10.16b
	mov v19.s[3], v8.s[3]
	str q19, [x0], #16
	b.ne .LBB23_12
.LBB23_13:
	.cfi_def_cfa wsp, 160
	ldp x20, x19, [sp, #144]
	ldp x29, x30, [sp, #128]
	ldp d9, d8, [sp, #112]
	ldp d11, d10, [sp, #96]
	ldp d13, d12, [sp, #80]
	ldp d15, d14, [sp, #64]
	add sp, sp, #160
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
.LBB23_14:
	.cfi_restore_state
	mov x19, x0
	mov x20, x1
	fmov s8, s0
	bl archmage::tokens::generated::arm::neon_detect
	fmov s0, s8
	mov x1, x20
	mov w8, w0
	mov x0, x19
	cbnz w8, .LBB23_2
.LBB23_15:
	.cfi_def_cfa wsp, 160
	ldp x20, x19, [sp, #144]
	ldp x29, x30, [sp, #128]
	ldp d9, d8, [sp, #112]
	ldp d11, d10, [sp, #96]
	ldp d13, d12, [sp, #80]
	ldp d15, d14, [sp, #64]
	add sp, sp, #160
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
	b linear_srgb::simd::unpremultiply_linear_to_gamma_rgba_slice_tier_scalar
