.section .text.stub_unpremultiply_linear_to_gamma_rgba_slice,"ax",@progbits
	.globl	stub_unpremultiply_linear_to_gamma_rgba_slice
	.p2align	2
.type	stub_unpremultiply_linear_to_gamma_rgba_slice,@function
stub_unpremultiply_linear_to_gamma_rgba_slice:
	.cfi_startproc
	stp d15, d14, [sp, #-160]!
	.cfi_def_cfa_offset 160
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	stp x29, x30, [sp, #64]
	stp x28, x27, [sp, #80]
	stp x26, x25, [sp, #96]
	stp x24, x23, [sp, #112]
	stp x22, x21, [sp, #128]
	stp x20, x19, [sp, #144]
	add x29, sp, #64
	.cfi_def_cfa w29, 96
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -56
	.cfi_offset w26, -64
	.cfi_offset w27, -72
	.cfi_offset w28, -80
	.cfi_offset w30, -88
	.cfi_offset w29, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	.cfi_offset b10, -120
	.cfi_offset b11, -128
	.cfi_offset b12, -136
	.cfi_offset b13, -144
	.cfi_offset b14, -152
	.cfi_offset b15, -160
	.cfi_remember_state
	sub sp, sp, #592
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	mov x28, x0
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB20_27
	cmp w8, #2
	b.ne .LBB20_26
.LBB20_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x19, x1, #0xc
	lsr x20, x1, #4
	add x21, x28, x8, lsl #2
	lsl x22, x19, #2
	stp x22, x21, [sp, #16]
	b.eq .LBB20_4
	mov w8, #64
	mov w1, wzr
	fmov s8, s0
	sub x2, x8, x22
	add x8, sp, #348
	add x0, x8, x22
	bl memset
	add x0, sp, #348
	mov x1, x21
	mov x2, x22
	bl memcpy
	add x8, sp, #93
	fmov s0, s8
	add x9, sp, #77
	ldur q1, [x8, #255]
	add x8, sp, #109
	ldur q2, [x8, #255]
	add x8, sp, #29
	stur q1, [x8, #255]
	add x8, sp, #125
	ldur q1, [x8, #255]
	add x8, sp, #141
	ldur q3, [x8, #255]
	add x8, sp, #45
	stur q2, [x8, #255]
	add x8, sp, #61
	stur q1, [x8, #255]
	mov w8, #1
	stur q3, [x9, #255]
	b .LBB20_5
.LBB20_4:
	mov w8, wzr
.LBB20_5:
	fmov s8, #1.00000000
	movi d1, #0000000000000000
	str w8, [sp, #280]
	add x8, sp, #280
	cmp x19, #0
	fmov v25.4s, #1.00000000
	add x8, x8, #4
	mov w11, #931135488
	mov w10, #981467136
	fdiv s30, s8, s0
	fmov s0, #4.00000000
	csel x22, xzr, x8, eq
	str x8, [sp, #8]
	mov w15, #39322
	mov w12, #64269
	mov w13, #1267
	fmov s9, w10
	movk w15, #16409, lsl #16
	movk w12, #74, lsl #16
	movk w13, #16181, lsl #16
	mov w9, #8388607
	fmov s10, w15
	dup v2.4s, w12
	add x24, x28, x20, lsl #6
	mov w20, #29208
	mov w27, #21845
	sub x23, x29, #240
	movk w20, #16177, lsl #16
	sub x21, x29, #81
	sub x25, x29, #240
	sub x26, x29, #244
	movk w27, #16085, lsl #16
	str s10, [sp, #108]
	fcmeq v1.4s, v30.4s, v1.4s
	fcmp s30, #0.0
	fmov w8, s30
	cset w14, lt
	fcmp s30, s0
	ubfx w10, w8, #23, #8
	and v0.16b, v1.16b, v25.16b
	dup v1.4s, w11
	and w8, w8, #0x7fffffff
	sub w8, w8, #1
	csinc w14, w14, wzr, ls
	cmp w10, #254
	stp q1, q30, [sp, #112]
	dup v1.4s, w13
	dup v0.4s, v0.s[0]
	ccmp w8, w9, #0, ls
	csinc w8, w14, wzr, hs
	stp q1, q2, [sp, #64]
	str q0, [sp, #32]
	str w8, [sp, #60]
	b .LBB20_8
.LBB20_6:
	sub x0, x29, #176
	sub x1, x29, #112
	stp x21, x25, [x29, #-112]
	stur x26, [x29, #-96]
	bl linear_srgb::tf::math::gamma_x16::<archmage::tokens::generated::arm::NeonToken>::{closure#0}
	ldp q0, q1, [x23, #64]
	fmov v25.4s, #1.00000000
	ldp q2, q3, [x23, #96]
	ldr q30, [sp, #128]
.LBB20_7:
	ldp q21, q20, [sp, #240]
	ldp q23, q22, [sp, #208]
	ldp q17, q16, [sp, #144]
	fcmge v4.4s, v20.4s, v25.4s
	fcmge v5.4s, v21.4s, v25.4s
	fcmge v6.4s, v22.4s, v25.4s
	fcmge v7.4s, v23.4s, v25.4s
	ldp q18, q19, [sp, #176]
	bic v16.16b, v16.16b, v4.16b
	bic v17.16b, v17.16b, v5.16b
	and v4.16b, v4.16b, v25.16b
	bic v18.16b, v18.16b, v6.16b
	bic v19.16b, v19.16b, v7.16b
	and v5.16b, v5.16b, v25.16b
	and v6.16b, v6.16b, v25.16b
	and v7.16b, v7.16b, v25.16b
	and v0.16b, v16.16b, v0.16b
	and v1.16b, v17.16b, v1.16b
	fcmeq v16.4s, v21.4s, v21.4s
	and v2.16b, v18.16b, v2.16b
	and v3.16b, v19.16b, v3.16b
	fcmeq v17.4s, v20.4s, v20.4s
	orr v0.16b, v0.16b, v4.16b
	fcmeq v4.4s, v22.4s, v22.4s
	orr v1.16b, v1.16b, v5.16b
	fcmeq v5.4s, v23.4s, v23.4s
	orr v2.16b, v2.16b, v6.16b
	orr v3.16b, v3.16b, v7.16b
	bif v0.16b, v20.16b, v17.16b
	bif v1.16b, v21.16b, v16.16b
	bif v2.16b, v22.16b, v4.16b
	bif v3.16b, v23.16b, v5.16b
	stp q0, q1, [x19]
	str s11, [x19, #12]
	stp q2, q3, [x19, #32]
	str s12, [x19, #28]
	str s13, [x19, #44]
	str s14, [x19, #60]
.LBB20_8:
	cmp x28, x24
	ccmp x28, #0, #4, ne
	b.ne .LBB20_11
	cbz x22, .LBB20_23
	mov x28, xzr
	mov x19, x22
	mov x22, xzr
	b .LBB20_12
.LBB20_11:
	add x8, x28, #64
	mov x19, x28
	mov x28, x8
.LBB20_12:
	ldr s11, [x19, #12]
	ldr s12, [x19, #28]
	movi d16, #0000000000000000
	ldr s13, [x19, #44]
	ldr s14, [x19, #60]
	fmov v4.4s, #1.00000000
	fdiv s0, s8, s11
	fcmp s11, s9
	fmov v5.4s, #1.00000000
	fmov v6.4s, #1.00000000
	fmov v7.4s, #1.00000000
	stur s30, [x29, #-244]
	fdiv s1, s8, s12
	fcsel s0, s0, s16, gt
	fcmp s12, s9
	mov v4.s[0], v0.s[0]
	fdiv s2, s8, s13
	mov v4.s[1], v0.s[0]
	fcsel s1, s1, s16, gt
	fcmp s13, s9
	mov v4.s[2], v0.s[0]
	mov v5.s[0], v1.s[0]
	fdiv s3, s8, s14
	mov v5.s[1], v1.s[0]
	fcsel s2, s2, s16, gt
	fcmp s14, s9
	mov v5.s[2], v1.s[0]
	ldp q0, q1, [x19]
	mov v6.s[0], v2.s[0]
	fmul v4.4s, v0.4s, v4.4s
	fmul v5.4s, v1.4s, v5.4s
	mov v6.s[1], v2.s[0]
	fcmgt v18.4s, v4.4s, #0.0
	fcmgt v19.4s, v5.4s, #0.0
	fcmgt v1.4s, v25.4s, v5.4s
	stp q5, q4, [sp, #240]
	fcsel s3, s3, s16, gt
	fcmp s30, s10
	mov v6.s[2], v2.s[0]
	ldp q2, q0, [x19, #32]
	mov v7.s[0], v3.s[0]
	and v1.16b, v1.16b, v19.16b
	stp q19, q18, [sp, #144]
	fmul v16.4s, v2.4s, v6.4s
	mov v7.s[1], v3.s[0]
	fcmgt v2.4s, v25.4s, v16.4s
	mov v7.s[2], v3.s[0]
	fmul v17.4s, v0.4s, v7.4s
	fcmgt v0.4s, v25.4s, v4.4s
	fcmgt v7.4s, v16.4s, #0.0
	fcmgt v6.4s, v17.4s, #0.0
	fcmgt v3.4s, v25.4s, v17.4s
	and v0.16b, v0.16b, v18.16b
	and v2.16b, v2.16b, v7.16b
	stp q17, q16, [sp, #208]
	and v3.16b, v3.16b, v6.16b
	stp q7, q6, [sp, #176]
	mov v7.16b, v0.16b
	mov v6.16b, v1.16b
	mov v1.16b, v2.16b
	mov v0.16b, v3.16b
	bsl v7.16b, v4.16b, v25.16b
	bsl v6.16b, v5.16b, v25.16b
	bsl v1.16b, v16.16b, v25.16b
	bsl v0.16b, v17.16b, v25.16b
	stp q7, q6, [x23]
	stp q1, q0, [x23, #32]
	b.ne .LBB20_15
	fcmge v2.4s, v25.4s, v7.4s
	fcmge v3.4s, v25.4s, v6.4s
	ldr q19, [sp, #112]
	fcmge v4.4s, v25.4s, v1.4s
	fcmge v5.4s, v25.4s, v0.4s
	fcmge v16.4s, v7.4s, v19.4s
	fcmge v17.4s, v6.4s, v19.4s
	fcmge v18.4s, v1.4s, v19.4s
	fcmge v19.4s, v0.4s, v19.4s
	mvn v20.16b, v2.16b
	mvn v21.16b, v3.16b
	cmeq v2.4s, v7.4s, #0
	mvn v22.16b, v4.16b
	mvn v23.16b, v5.16b
	cmeq v3.4s, v6.4s, #0
	cmeq v4.4s, v1.4s, #0
	cmeq v5.4s, v0.4s, #0
	orn v16.16b, v20.16b, v16.16b
	orn v17.16b, v21.16b, v17.16b
	orn v18.16b, v22.16b, v18.16b
	orn v19.16b, v23.16b, v19.16b
	bic v16.16b, v16.16b, v2.16b
	bic v17.16b, v17.16b, v3.16b
	bic v18.16b, v18.16b, v4.16b
	bic v19.16b, v19.16b, v5.16b
	orr v16.16b, v17.16b, v16.16b
	orr v17.16b, v18.16b, v19.16b
	orr v16.16b, v16.16b, v17.16b
	xtn v16.4h, v16.4s
	shl v16.4h, v16.4h, #15
	cmlt v16.4h, v16.4h, #0
	fmov x8, d16
	cbnz x8, .LBB20_6
	ldp q24, q23, [sp, #64]
	movi v21.4s, #127, msl #16
	fmov v22.4s, #-1.00000000
	mov w8, #46233
	mvni v30.4s, #126
	movk w8, #16147, lsl #16
	mvni v31.4s, #126
	add v7.4s, v7.4s, v23.4s
	add v18.4s, v6.4s, v23.4s
	and v16.16b, v7.16b, v21.16b
	and v19.16b, v18.16b, v21.16b
	ssra v30.4s, v18.4s, #23
	add v16.4s, v16.4s, v24.4s
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	fdiv v6.4s, v17.4s, v16.4s
	add v16.4s, v19.4s, v24.4s
	add v19.4s, v1.4s, v23.4s
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	and v20.16b, v19.16b, v21.16b
	fdiv v1.4s, v17.4s, v16.4s
	add v16.4s, v20.4s, v24.4s
	add v20.4s, v0.4s, v23.4s
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	and v21.16b, v20.16b, v21.16b
	ssra v31.4s, v20.4s, #23
	scvtf v20.4s, v30.4s
	fdiv v0.4s, v17.4s, v16.4s
	add v16.4s, v21.4s, v24.4s
	dup v21.4s, w8
	mov w8, #27455
	mvni v24.4s, #126
	movk w8, #16094, lsl #16
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	dup v23.4s, w8
	mov w8, #14483
	mov v27.16b, v21.16b
	fmul v22.4s, v1.4s, v1.4s
	movk w8, #16246, lsl #16
	mov v28.16b, v21.16b
	ssra v24.4s, v7.4s, #23
	dup v29.4s, w8
	mov v7.16b, v21.16b
	mov w8, #43579
	movk w8, #16440, lsl #16
	fmla v28.4s, v23.4s, v22.4s
	dup v10.4s, w8
	mov w8, #39322
	mov v15.16b, v29.16b
	mov v18.16b, v29.16b
	movk w8, #16409, lsl #16
	fdiv v16.4s, v17.4s, v16.4s
	fmul v17.4s, v6.4s, v6.4s
	fmul v25.4s, v0.4s, v0.4s
	fmla v27.4s, v23.4s, v17.4s
	fmla v7.4s, v23.4s, v25.4s
	fmla v15.4s, v17.4s, v27.4s
	mov v27.16b, v29.16b
	fmla v27.4s, v22.4s, v28.4s
	fmla v18.4s, v25.4s, v7.4s
	scvtf v7.4s, v24.4s
	fmul v26.4s, v16.4s, v16.4s
	fmla v21.4s, v23.4s, v26.4s
	mvni v23.4s, #126
	ssra v23.4s, v19.4s, #23
	mov v19.16b, v10.16b
	fmla v29.4s, v26.4s, v21.4s
	fmla v19.4s, v17.4s, v15.4s
	mov v17.16b, v10.16b
	scvtf v21.4s, v23.4s
	scvtf v23.4s, v31.4s
	fmla v17.4s, v22.4s, v27.4s
	mov v22.16b, v10.16b
	fmla v10.4s, v26.4s, v29.4s
	fmla v7.4s, v6.4s, v19.4s
	fmla v22.4s, v25.4s, v18.4s
	fmla v20.4s, v1.4s, v17.4s
	dup v1.4s, w8
	fmla v23.4s, v16.4s, v10.4s
	b .LBB20_18
.LBB20_15:
	fmov s2, w27
	fcmp s30, s2
	b.ne .LBB20_20
	fcmge v2.4s, v25.4s, v7.4s
	fcmge v3.4s, v25.4s, v6.4s
	ldr q19, [sp, #112]
	fcmge v4.4s, v25.4s, v1.4s
	fcmge v5.4s, v25.4s, v0.4s
	fcmge v16.4s, v7.4s, v19.4s
	fcmge v17.4s, v6.4s, v19.4s
	fcmge v18.4s, v1.4s, v19.4s
	fcmge v19.4s, v0.4s, v19.4s
	mvn v20.16b, v2.16b
	mvn v21.16b, v3.16b
	cmeq v2.4s, v7.4s, #0
	mvn v22.16b, v4.16b
	mvn v23.16b, v5.16b
	cmeq v3.4s, v6.4s, #0
	cmeq v4.4s, v1.4s, #0
	cmeq v5.4s, v0.4s, #0
	orn v16.16b, v20.16b, v16.16b
	orn v17.16b, v21.16b, v17.16b
	orn v18.16b, v22.16b, v18.16b
	orn v19.16b, v23.16b, v19.16b
	bic v16.16b, v16.16b, v2.16b
	bic v17.16b, v17.16b, v3.16b
	bic v18.16b, v18.16b, v4.16b
	bic v19.16b, v19.16b, v5.16b
	orr v16.16b, v17.16b, v16.16b
	orr v17.16b, v18.16b, v19.16b
	orr v16.16b, v16.16b, v17.16b
	xtn v16.4h, v16.4s
	shl v16.4h, v16.4h, #15
	cmlt v16.4h, v16.4h, #0
	fmov x8, d16
	cbnz x8, .LBB20_6
	ldp q24, q23, [sp, #64]
	movi v21.4s, #127, msl #16
	fmov v22.4s, #-1.00000000
	mov w8, #46233
	mvni v30.4s, #126
	movk w8, #16147, lsl #16
	mvni v31.4s, #126
	add v7.4s, v7.4s, v23.4s
	add v18.4s, v6.4s, v23.4s
	and v16.16b, v7.16b, v21.16b
	and v19.16b, v18.16b, v21.16b
	ssra v30.4s, v18.4s, #23
	add v16.4s, v16.4s, v24.4s
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	fdiv v6.4s, v17.4s, v16.4s
	add v16.4s, v19.4s, v24.4s
	add v19.4s, v1.4s, v23.4s
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	and v20.16b, v19.16b, v21.16b
	fdiv v1.4s, v17.4s, v16.4s
	add v16.4s, v20.4s, v24.4s
	add v20.4s, v0.4s, v23.4s
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	and v21.16b, v20.16b, v21.16b
	ssra v31.4s, v20.4s, #23
	scvtf v20.4s, v30.4s
	fdiv v0.4s, v17.4s, v16.4s
	add v16.4s, v21.4s, v24.4s
	dup v21.4s, w8
	mov w8, #27455
	mvni v24.4s, #126
	movk w8, #16094, lsl #16
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	dup v23.4s, w8
	mov w8, #14483
	mov v27.16b, v21.16b
	fmul v22.4s, v1.4s, v1.4s
	movk w8, #16246, lsl #16
	mov v28.16b, v21.16b
	ssra v24.4s, v7.4s, #23
	dup v29.4s, w8
	mov v7.16b, v21.16b
	mov w8, #43579
	movk w8, #16440, lsl #16
	fmla v28.4s, v23.4s, v22.4s
	dup v15.4s, w8
	mov v10.16b, v29.16b
	mov v18.16b, v29.16b
	fdiv v16.4s, v17.4s, v16.4s
	fmul v17.4s, v6.4s, v6.4s
	fmul v25.4s, v0.4s, v0.4s
	fmla v27.4s, v23.4s, v17.4s
	fmla v7.4s, v23.4s, v25.4s
	fmla v10.4s, v17.4s, v27.4s
	mov v27.16b, v29.16b
	fmla v27.4s, v22.4s, v28.4s
	fmla v18.4s, v25.4s, v7.4s
	scvtf v7.4s, v24.4s
	fmul v26.4s, v16.4s, v16.4s
	fmla v21.4s, v23.4s, v26.4s
	mvni v23.4s, #126
	ssra v23.4s, v19.4s, #23
	mov v19.16b, v15.16b
	fmla v29.4s, v26.4s, v21.4s
	fmla v19.4s, v17.4s, v10.4s
	mov v17.16b, v15.16b
	scvtf v21.4s, v23.4s
	scvtf v23.4s, v31.4s
	fmla v17.4s, v22.4s, v27.4s
	mov v22.16b, v15.16b
	fmla v15.4s, v26.4s, v29.4s
	fmla v7.4s, v6.4s, v19.4s
	fmla v22.4s, v25.4s, v18.4s
	fmla v20.4s, v1.4s, v17.4s
	fmla v23.4s, v16.4s, v15.4s
	dup v1.4s, w27
.LBB20_18:
	fmla v21.4s, v0.4s, v22.4s
	fmul v0.4s, v7.4s, v1.4s
	fmul v6.4s, v20.4s, v1.4s
	mov w8, #50297
	fmul v7.4s, v21.4s, v1.4s
	fmul v20.4s, v23.4s, v1.4s
	movk w8, #15022, lsl #16
	dup v21.4s, w8
	mov w8, #63802
	ldr q30, [sp, #128]
	movk w8, #14625, lsl #16
	frintn v16.4s, v0.4s
	frintn v17.4s, v6.4s
	dup v22.4s, w8
	frintn v18.4s, v7.4s
	frintn v19.4s, v20.4s
	mov w8, #38156
	movk w8, #15389, lsl #16
	mov v23.16b, v21.16b
	mov v25.16b, v21.16b
	mov v24.16b, v21.16b
	fsub v0.4s, v0.4s, v16.4s
	fsub v1.4s, v6.4s, v17.4s
	fcvtns v16.4s, v16.4s
	fsub v6.4s, v7.4s, v18.4s
	fsub v7.4s, v20.4s, v19.4s
	dup v20.4s, w8
	mov w8, #22716
	fcvtns v17.4s, v17.4s
	fcvtns v18.4s, v18.4s
	movk w8, #15715, lsl #16
	fcvtns v19.4s, v19.4s
	fmla v23.4s, v22.4s, v0.4s
	mov v26.16b, v20.16b
	fmla v24.4s, v22.4s, v1.4s
	fmla v25.4s, v22.4s, v6.4s
	fmla v21.4s, v22.4s, v7.4s
	dup v22.4s, w8
	mov v28.16b, v20.16b
	mov v27.16b, v20.16b
	mov w8, #65005
	movk w8, #15989, lsl #16
	shl v16.4s, v16.4s, #23
	shl v17.4s, v17.4s, #23
	fmla v26.4s, v0.4s, v23.4s
	mov v23.16b, v22.16b
	shl v18.4s, v18.4s, #23
	fmla v28.4s, v6.4s, v25.4s
	fmla v27.4s, v1.4s, v24.4s
	mov v25.16b, v22.16b
	fmla v20.4s, v7.4s, v21.4s
	dup v21.4s, w8
	mov v24.16b, v22.16b
	shl v19.4s, v19.4s, #23
	fmla v23.4s, v0.4s, v26.4s
	fmov v26.4s, #1.00000000
	fmla v25.4s, v6.4s, v28.4s
	mov v28.16b, v21.16b
	fmla v24.4s, v1.4s, v27.4s
	fmla v22.4s, v7.4s, v20.4s
	mov v29.16b, v21.16b
	dup v20.4s, w20
	fmov v27.4s, #1.00000000
	fmla v28.4s, v0.4s, v23.4s
	mov v23.16b, v21.16b
	fmla v29.4s, v1.4s, v24.4s
	fmov v24.4s, #1.00000000
	fmla v21.4s, v7.4s, v22.4s
	mov v22.16b, v20.16b
	fmla v23.4s, v6.4s, v25.4s
	mov v25.16b, v20.16b
	fmla v22.4s, v0.4s, v28.4s
	mov v28.16b, v20.16b
	fmla v25.4s, v1.4s, v29.4s
	fmov v29.4s, #1.00000000
	fmla v20.4s, v7.4s, v21.4s
	fmla v28.4s, v6.4s, v23.4s
	fmla v26.4s, v0.4s, v22.4s
	fmla v27.4s, v1.4s, v25.4s
	fmov v25.4s, #1.00000000
	fmla v29.4s, v7.4s, v20.4s
	fmla v24.4s, v6.4s, v28.4s
	add v0.4s, v16.4s, v25.4s
	add v1.4s, v17.4s, v25.4s
	add v6.4s, v18.4s, v25.4s
	add v7.4s, v19.4s, v25.4s
	fmul v0.4s, v26.4s, v0.4s
	fmul v1.4s, v27.4s, v1.4s
	fmul v6.4s, v24.4s, v6.4s
	fmul v7.4s, v29.4s, v7.4s
	bic v0.16b, v0.16b, v2.16b
	bic v1.16b, v1.16b, v3.16b
	bic v2.16b, v6.16b, v4.16b
	bic v3.16b, v7.16b, v5.16b
.LBB20_19:
	ldr s10, [sp, #108]
	str q0, [x23, #64]
	b .LBB20_7
.LBB20_20:
	ldr w8, [sp, #60]
	tbnz w8, #0, .LBB20_6
	fcmge v2.4s, v25.4s, v7.4s
	fcmge v3.4s, v25.4s, v6.4s
	ldr q19, [sp, #112]
	fcmge v4.4s, v25.4s, v1.4s
	fcmge v5.4s, v25.4s, v0.4s
	fcmge v16.4s, v7.4s, v19.4s
	fcmge v17.4s, v6.4s, v19.4s
	fcmge v18.4s, v1.4s, v19.4s
	fcmge v19.4s, v0.4s, v19.4s
	mvn v20.16b, v2.16b
	mvn v21.16b, v3.16b
	cmeq v2.4s, v7.4s, #0
	mvn v22.16b, v4.16b
	mvn v23.16b, v5.16b
	cmeq v3.4s, v6.4s, #0
	cmeq v4.4s, v1.4s, #0
	cmeq v5.4s, v0.4s, #0
	orn v16.16b, v20.16b, v16.16b
	orn v17.16b, v21.16b, v17.16b
	orn v18.16b, v22.16b, v18.16b
	orn v19.16b, v23.16b, v19.16b
	bic v16.16b, v16.16b, v2.16b
	bic v17.16b, v17.16b, v3.16b
	bic v18.16b, v18.16b, v4.16b
	bic v19.16b, v19.16b, v5.16b
	orr v16.16b, v17.16b, v16.16b
	orr v17.16b, v18.16b, v19.16b
	orr v16.16b, v16.16b, v17.16b
	xtn v16.4h, v16.4s
	shl v16.4h, v16.4h, #15
	cmlt v16.4h, v16.4h, #0
	fmov x8, d16
	cbnz x8, .LBB20_6
	ldp q24, q23, [sp, #64]
	movi v21.4s, #127, msl #16
	fmov v22.4s, #-1.00000000
	mov w8, #46233
	mvni v30.4s, #126
	movk w8, #16147, lsl #16
	mvni v31.4s, #126
	add v7.4s, v7.4s, v23.4s
	add v18.4s, v6.4s, v23.4s
	and v16.16b, v7.16b, v21.16b
	and v19.16b, v18.16b, v21.16b
	ssra v30.4s, v18.4s, #23
	add v16.4s, v16.4s, v24.4s
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	fdiv v6.4s, v17.4s, v16.4s
	add v16.4s, v19.4s, v24.4s
	add v19.4s, v1.4s, v23.4s
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	and v20.16b, v19.16b, v21.16b
	fdiv v1.4s, v17.4s, v16.4s
	add v16.4s, v20.4s, v24.4s
	add v20.4s, v0.4s, v23.4s
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	and v21.16b, v20.16b, v21.16b
	ssra v31.4s, v20.4s, #23
	scvtf v20.4s, v30.4s
	ldr q30, [sp, #128]
	fdiv v0.4s, v17.4s, v16.4s
	add v16.4s, v21.4s, v24.4s
	dup v21.4s, w8
	mov w8, #27455
	mvni v24.4s, #126
	movk w8, #16094, lsl #16
	fadd v17.4s, v16.4s, v22.4s
	fadd v16.4s, v16.4s, v25.4s
	dup v23.4s, w8
	mov w8, #14483
	mov v27.16b, v21.16b
	fmul v22.4s, v1.4s, v1.4s
	movk w8, #16246, lsl #16
	mov v28.16b, v21.16b
	ssra v24.4s, v7.4s, #23
	dup v29.4s, w8
	mov v7.16b, v21.16b
	mov w8, #43579
	movk w8, #16440, lsl #16
	fmla v28.4s, v23.4s, v22.4s
	dup v15.4s, w8
	mov w8, #50297
	mov v10.16b, v29.16b
	mov v18.16b, v29.16b
	movk w8, #15022, lsl #16
	fdiv v16.4s, v17.4s, v16.4s
	fmul v17.4s, v6.4s, v6.4s
	fmul v25.4s, v0.4s, v0.4s
	fmla v27.4s, v23.4s, v17.4s
	fmla v7.4s, v23.4s, v25.4s
	fmla v10.4s, v17.4s, v27.4s
	mov v27.16b, v29.16b
	fmla v27.4s, v22.4s, v28.4s
	fmla v18.4s, v25.4s, v7.4s
	scvtf v7.4s, v24.4s
	fmul v26.4s, v16.4s, v16.4s
	fmla v21.4s, v23.4s, v26.4s
	mvni v23.4s, #126
	ssra v23.4s, v19.4s, #23
	mov v19.16b, v15.16b
	fmla v29.4s, v26.4s, v21.4s
	fmla v19.4s, v17.4s, v10.4s
	mov v17.16b, v15.16b
	scvtf v21.4s, v23.4s
	scvtf v23.4s, v31.4s
	fmla v17.4s, v22.4s, v27.4s
	mov v22.16b, v15.16b
	fmla v15.4s, v26.4s, v29.4s
	fmla v7.4s, v6.4s, v19.4s
	fmla v22.4s, v25.4s, v18.4s
	fmla v20.4s, v1.4s, v17.4s
	fmla v23.4s, v16.4s, v15.4s
	fmla v21.4s, v0.4s, v22.4s
	fmul v0.4s, v7.4s, v30.s[0]
	fmul v1.4s, v20.4s, v30.s[0]
	fmul v7.4s, v23.4s, v30.s[0]
	dup v20.4s, w8
	mov w8, #63802
	movk w8, #14625, lsl #16
	fmul v6.4s, v21.4s, v30.s[0]
	frintn v16.4s, v0.4s
	dup v21.4s, w8
	frintn v17.4s, v1.4s
	frintn v19.4s, v7.4s
	mov w8, #38156
	movk w8, #15389, lsl #16
	mov v23.16b, v20.16b
	mov v25.16b, v20.16b
	dup v22.4s, w8
	mov v24.16b, v20.16b
	mov w8, #22716
	frintn v18.4s, v6.4s
	fsub v0.4s, v0.4s, v16.4s
	movk w8, #15715, lsl #16
	fsub v1.4s, v1.4s, v17.4s
	fsub v7.4s, v7.4s, v19.4s
	fcvtns v16.4s, v16.4s
	mov v26.16b, v22.16b
	mov v28.16b, v22.16b
	mov v27.16b, v22.16b
	fcvtns v17.4s, v17.4s
	fcvtns v19.4s, v19.4s
	fsub v6.4s, v6.4s, v18.4s
	fmla v23.4s, v21.4s, v0.4s
	fcvtns v18.4s, v18.4s
	fmla v24.4s, v21.4s, v1.4s
	fmla v20.4s, v21.4s, v7.4s
	shl v16.4s, v16.4s, #23
	shl v17.4s, v17.4s, #23
	shl v19.4s, v19.4s, #23
	fmla v25.4s, v21.4s, v6.4s
	dup v21.4s, w8
	fmla v26.4s, v0.4s, v23.4s
	mov w8, #65005
	fmla v27.4s, v1.4s, v24.4s
	fmla v22.4s, v7.4s, v20.4s
	movk w8, #15989, lsl #16
	shl v18.4s, v18.4s, #23
	mov v23.16b, v21.16b
	dup v20.4s, w8
	mov v24.16b, v21.16b
	fmla v28.4s, v6.4s, v25.4s
	mov v25.16b, v21.16b
	fmla v21.4s, v7.4s, v22.4s
	dup v22.4s, w20
	fmla v23.4s, v0.4s, v26.4s
	fmla v24.4s, v1.4s, v27.4s
	mov v29.16b, v20.16b
	fmov v27.4s, #1.00000000
	fmov v26.4s, #1.00000000
	fmla v25.4s, v6.4s, v28.4s
	mov v28.16b, v20.16b
	fmla v29.4s, v1.4s, v24.4s
	fmov v24.4s, #1.00000000
	fmla v28.4s, v0.4s, v23.4s
	mov v23.16b, v20.16b
	fmla v20.4s, v7.4s, v21.4s
	mov v21.16b, v22.16b
	fmla v23.4s, v6.4s, v25.4s
	mov v25.16b, v22.16b
	fmla v21.4s, v0.4s, v28.4s
	mov v28.16b, v22.16b
	fmla v22.4s, v7.4s, v20.4s
	fmla v25.4s, v1.4s, v29.4s
	fmov v29.4s, #1.00000000
	fmla v28.4s, v6.4s, v23.4s
	fmla v26.4s, v0.4s, v21.4s
	fmla v27.4s, v1.4s, v25.4s
	fmov v25.4s, #1.00000000
	fmla v29.4s, v7.4s, v22.4s
	fmla v24.4s, v6.4s, v28.4s
	add v0.4s, v16.4s, v25.4s
	add v1.4s, v17.4s, v25.4s
	add v6.4s, v18.4s, v25.4s
	add v7.4s, v19.4s, v25.4s
	ldr q16, [sp, #32]
	fmul v0.4s, v26.4s, v0.4s
	fmul v1.4s, v27.4s, v1.4s
	fmul v6.4s, v24.4s, v6.4s
	fmul v7.4s, v29.4s, v7.4s
	bit v0.16b, v16.16b, v2.16b
	bit v1.16b, v16.16b, v3.16b
	mov v2.16b, v4.16b
	mov v3.16b, v5.16b
	bsl v2.16b, v16.16b, v6.16b
	bsl v3.16b, v16.16b, v7.16b
	b .LBB20_19
.LBB20_23:
	ldr w8, [sp, #280]
	cbz w8, .LBB20_25
	ldp x2, x0, [sp, #16]
	ldr x1, [sp, #8]
	bl memcpy
.LBB20_25:
	add sp, sp, #592
	.cfi_def_cfa wsp, 160
	ldp x20, x19, [sp, #144]
	ldp x22, x21, [sp, #128]
	ldp x24, x23, [sp, #112]
	ldp x26, x25, [sp, #96]
	ldp x28, x27, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #160
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w26
	.cfi_restore w27
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
.LBB20_26:
	.cfi_restore_state
	fmov s8, s0
	mov x19, x1
	bl archmage::tokens::generated::arm::neon_detect
	fmov s0, s8
	mov x1, x19
	cbnz w0, .LBB20_2
.LBB20_27:
	mov x0, x28
	add sp, sp, #592
	.cfi_def_cfa wsp, 160
	ldp x20, x19, [sp, #144]
	ldp x22, x21, [sp, #128]
	ldp x24, x23, [sp, #112]
	ldp x26, x25, [sp, #96]
	ldp x28, x27, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #160
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w26
	.cfi_restore w27
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
	b linear_srgb::simd::unpremultiply_linear_to_gamma_rgba_slice_tier_scalar
