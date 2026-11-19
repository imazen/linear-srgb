.section .text.stub_srgb_to_linear_extended_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_extended_slice
	.p2align	2
.type	stub_srgb_to_linear_extended_slice,@function
stub_srgb_to_linear_extended_slice:
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
	sub sp, sp, #656
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	mov x25, x0
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB11_27
	cmp w8, #2
	b.ne .LBB11_26
.LBB11_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x19, x1, #0xf
	lsr x20, x1, #4
	add x21, x25, x8, lsl #2
	lsl x22, x19, #2
	stp x22, x21, [sp, #16]
	b.eq .LBB11_4
	mov w8, #64
	mov w1, wzr
	sub x2, x8, x22
	add x8, sp, #400
	add x0, x8, x22
	bl memset
	add x0, sp, #400
	mov x1, x21
	mov x2, x22
	bl memcpy
	ldp q0, q1, [sp, #400]
	add x8, sp, #81
	add x9, sp, #129
	stur q0, [x8, #255]
	ldp q0, q2, [sp, #432]
	add x8, sp, #97
	stur q1, [x8, #255]
	add x8, sp, #113
	stur q0, [x8, #255]
	mov w8, #1
	stur q2, [x9, #255]
	b .LBB11_5
.LBB11_4:
	mov w8, wzr
.LBB11_5:
	str w8, [sp, #332]
	add x8, x25, x20, lsl #6
	cmp x19, #0
	mov w27, #21227
	mov w20, #2711
	fmov s8, #8.00000000
	str x8, [sp, #280]
	add x8, sp, #332
	movk w27, #15713, lsl #16
	add x9, x8, #4
	mov w8, #33681
	movk w20, #16263, lsl #16
	str x9, [sp, #8]
	csel x26, xzr, x9, eq
	mov w9, #21091
	movk w8, #15774, lsl #16
	movk w9, #18053, lsl #16
	fmov s9, w27
	dup v1.4s, w8
	dup v0.4s, w9
	mov w8, #45243
	movk w8, #18434, lsl #16
	mov w9, #22505
	fmov s10, w20
	movk w9, #18375, lsl #16
	sub x10, x29, #192
	mov w22, #39322
	stp q0, q1, [sp, #240]
	dup v1.4s, w8
	mov w8, #54856
	movk w8, #18518, lsl #16
	sub x23, x29, #256
	add x28, x10, #32
	dup v0.4s, w8
	mov w8, #26646
	movk w22, #16409, lsl #16
	movk w8, #18037, lsl #16
	sub x21, x29, #256
	stp q0, q1, [sp, #208]
	dup v1.4s, w9
	dup v0.4s, w8
	mov w8, #49826
	mov w9, #11201
	movk w8, #17507, lsl #16
	movk w9, #16784, lsl #16
	stp q0, q1, [sp, #176]
	dup v1.4s, w8
	dup v0.4s, w9
	mov w8, #23593
	mov w9, #60686
	movk w8, #49790, lsl #16
	movk w9, #18336, lsl #16
	stp q0, q1, [sp, #144]
	dup v1.4s, w8
	mov w8, #51710
	movk w8, #17803, lsl #16
	dup v0.4s, w8
	mov w8, #61050
	movk w8, #18528, lsl #16
	stp q0, q1, [sp, #112]
	dup v1.4s, w9
	dup v0.4s, w8
	mov w8, #20961
	mov w9, #46089
	movk w8, #18451, lsl #16
	movk w9, #18088, lsl #16
	stp q0, q1, [sp, #80]
	dup v1.4s, w8
	dup v0.4s, w9
	mov w8, #61974
	movk w8, #15648, lsl #16
	stp q0, q1, [sp, #48]
	dup v0.4s, w8
	str q0, [sp, #32]
	b .LBB11_8
.LBB11_6:
	ldp q2, q5, [x23]
	ldp q6, q7, [x23, #32]
.LBB11_7:
	stp q2, q5, [x19]
	stp q6, q7, [x19, #32]
.LBB11_8:
	ldr x8, [sp, #280]
	cmp x25, x8
	ccmp x25, #0, #4, ne
	b.ne .LBB11_11
	cbz x26, .LBB11_23
	mov x25, xzr
	mov x19, x26
	mov x26, xzr
	b .LBB11_12
.LBB11_11:
	add x8, x25, #64
	mov x19, x25
	mov x25, x8
.LBB11_12:
	ldp q0, q1, [x19]
	ldr q31, [sp, #144]
	ldp q26, q25, [sp, #224]
	ldp q12, q11, [sp, #112]
	fabs v2.4s, v0.4s
	ldp q28, q27, [sp, #192]
	mov v3.16b, v26.16b
	ldp q14, q13, [sp, #80]
	mov v7.16b, v12.16b
	fabs v5.4s, v1.4s
	str q0, [sp, #304]
	mov v6.16b, v27.16b
	ldp q30, q29, [sp, #160]
	fmla v3.4s, v25.4s, v2.4s
	fadd v4.4s, v2.4s, v11.4s
	mov v17.16b, v14.16b
	mov v19.16b, v12.16b
	ldp q0, q15, [sp, #48]
	mov v16.16b, v29.16b
	mov v18.16b, v27.16b
	mov v21.16b, v26.16b
	mov v23.16b, v14.16b
	mov v24.16b, v12.16b
	fmla v6.4s, v2.4s, v3.4s
	fmla v7.4s, v2.4s, v4.4s
	mov v3.16b, v28.16b
	mov v4.16b, v13.16b
	mov v20.16b, v15.16b
	fmla v3.4s, v2.4s, v6.4s
	fmla v4.4s, v2.4s, v7.4s
	mov v6.16b, v26.16b
	fadd v7.4s, v5.4s, v11.4s
	fmla v6.4s, v25.4s, v5.4s
	fmla v16.4s, v2.4s, v3.4s
	fmla v17.4s, v2.4s, v4.4s
	fmla v19.4s, v5.4s, v7.4s
	mov v7.16b, v30.16b
	ldp q3, q4, [x19, #32]
	fmla v18.4s, v5.4s, v6.4s
	fmla v7.4s, v2.4s, v16.4s
	fmla v20.4s, v2.4s, v17.4s
	mov v16.16b, v28.16b
	mov v17.16b, v13.16b
	fabs v6.4s, v3.4s
	fmla v16.4s, v5.4s, v18.4s
	mov v18.16b, v31.16b
	fmla v17.4s, v5.4s, v19.4s
	mov v19.16b, v0.16b
	fmla v21.4s, v25.4s, v6.4s
	fadd v22.4s, v6.4s, v11.4s
	fmla v18.4s, v2.4s, v7.4s
	fmla v19.4s, v2.4s, v20.4s
	mov v20.16b, v29.16b
	fmla v23.4s, v5.4s, v17.4s
	mov v17.16b, v27.16b
	fmla v24.4s, v6.4s, v22.4s
	mov v22.16b, v26.16b
	fmla v20.4s, v5.4s, v16.4s
	fabs v16.4s, v4.4s
	fdiv v7.4s, v18.4s, v19.4s
	mov v18.16b, v30.16b
	fmla v17.4s, v6.4s, v21.4s
	mov v19.16b, v15.16b
	mov v21.16b, v13.16b
	fmla v18.4s, v5.4s, v20.4s
	mov v20.16b, v28.16b
	fmla v22.4s, v25.4s, v16.4s
	fmla v19.4s, v5.4s, v23.4s
	mov v23.16b, v0.16b
	fmla v21.4s, v6.4s, v24.4s
	fadd v24.4s, v16.4s, v11.4s
	fmla v20.4s, v6.4s, v17.4s
	mov v17.16b, v31.16b
	fmla v23.4s, v5.4s, v19.4s
	mov v19.16b, v14.16b
	fmla v17.4s, v5.4s, v18.4s
	mov v18.16b, v29.16b
	fmla v19.4s, v6.4s, v21.4s
	mov v21.16b, v12.16b
	fmla v18.4s, v6.4s, v20.4s
	mov v20.16b, v27.16b
	fdiv v17.4s, v17.4s, v23.4s
	mov v23.16b, v15.16b
	fmla v21.4s, v16.4s, v24.4s
	ldr q24, [sp, #256]
	fmla v20.4s, v16.4s, v22.4s
	mov v22.16b, v30.16b
	fmla v23.4s, v6.4s, v19.4s
	mov v19.16b, v13.16b
	fmla v22.4s, v6.4s, v18.4s
	mov v18.16b, v28.16b
	fmla v19.4s, v16.4s, v21.4s
	mov v21.16b, v0.16b
	fmla v18.4s, v16.4s, v20.4s
	mov v20.16b, v31.16b
	fmla v21.4s, v6.4s, v23.4s
	mov v23.16b, v14.16b
	fmla v20.4s, v6.4s, v22.4s
	mov v22.16b, v29.16b
	fmla v23.4s, v16.4s, v19.4s
	mov v19.16b, v30.16b
	fmla v22.4s, v16.4s, v18.4s
	fdiv v18.4s, v20.4s, v21.4s
	mov v20.16b, v15.16b
	mov v21.16b, v31.16b
	fmla v19.4s, v16.4s, v22.4s
	fmla v20.4s, v16.4s, v23.4s
	fmul v22.4s, v2.4s, v24.4s
	fmul v23.4s, v5.4s, v24.4s
	fmla v21.4s, v16.4s, v19.4s
	fmla v0.4s, v16.4s, v20.4s
	fmaxnm v20.4s, v2.4s, v5.4s
	fdiv v19.4s, v21.4s, v0.4s
	fmaxnm v21.4s, v6.4s, v16.4s
	ldr q0, [sp, #32]
	fcmgt v2.4s, v0.4s, v2.4s
	fcmgt v5.4s, v0.4s, v5.4s
	fmaxnm v20.4s, v20.4s, v21.4s
	fmul v21.4s, v6.4s, v24.4s
	fmul v24.4s, v16.4s, v24.4s
	fcmgt v6.4s, v0.4s, v6.4s
	fcmgt v16.4s, v0.4s, v16.4s
	movi v0.4s, #65, lsl #24
	bsl v2.16b, v22.16b, v7.16b
	bsl v5.16b, v23.16b, v17.16b
	ldr q17, [sp, #304]
	fcmgt v20.4s, v20.4s, v0.4s
	mov v7.16b, v16.16b
	bsl v6.16b, v21.16b, v18.16b
	mvni v0.4s, #128, lsl #24
	bsl v7.16b, v24.16b, v19.16b
	addp d16, v20.2d
	bif v2.16b, v17.16b, v0.16b
	bif v5.16b, v1.16b, v0.16b
	bif v6.16b, v3.16b, v0.16b
	bif v7.16b, v4.16b, v0.16b
	fmov x8, d16
	cbz x8, .LBB11_7
	mov x24, xzr
	sub x8, x29, #192
	stp q2, q5, [x23]
	stp q6, q7, [x23, #32]
	stp xzr, xzr, [x8, #96]
	stp q17, q1, [x23, #96]
	stp q3, q4, [x23, #128]
	b .LBB11_15
.LBB11_14:
	add x24, x24, #16
	cmp x24, #64
	b.eq .LBB11_6
.LBB11_15:
	ldr q0, [x28, x24]
	fabs v1.4s, v0.4s
	fcmp s1, s8
	stp q1, q0, [sp, #288]
	b.gt .LBB11_19
	mov s0, v1.s[1]
	fcmp s0, s8
	b.gt .LBB11_20
.LBB11_17:
	mov s0, v1.s[2]
	fcmp s0, s8
	b.gt .LBB11_21
.LBB11_18:
	mov s0, v1.s[3]
	fcmp s0, s8
	b.le .LBB11_14
	b .LBB11_22
.LBB11_19:
	fadd s0, s1, s9
	fmov s1, w20
	fdiv s0, s0, s1
	fmov s1, w22
	bl powf
	mvni v1.4s, #128, lsl #24
	fmov s2, #1.00000000
	ldr q3, [sp, #304]
	bsl v1.16b, v2.16b, v3.16b
	fmul s0, s1, s0
	ldr q1, [sp, #288]
	str s0, [x21, x24]
	mov s0, v1.s[1]
	fcmp s0, s8
	b.le .LBB11_17
.LBB11_20:
	fadd s0, s0, s9
	fmov s1, w22
	fdiv s0, s0, s10
	bl powf
	ldr q1, [sp, #304]
	mvni v2.4s, #128, lsl #24
	fmov s3, #1.00000000
	add x8, x21, x24
	mov s1, v1.s[1]
	bit v1.16b, v3.16b, v2.16b
	fmul s0, s1, s0
	ldr q1, [sp, #288]
	str s0, [x8, #4]
	mov s0, v1.s[2]
	fcmp s0, s8
	b.le .LBB11_18
.LBB11_21:
	fmov s1, w27
	fadd s0, s0, s1
	fmov s1, w20
	fdiv s0, s0, s1
	fmov s1, w22
	bl powf
	ldr q1, [sp, #304]
	mvni v2.4s, #128, lsl #24
	fmov s3, #1.00000000
	add x8, x21, x24
	mov s1, v1.s[2]
	bit v1.16b, v3.16b, v2.16b
	fmul s0, s1, s0
	ldr q1, [sp, #288]
	str s0, [x8, #8]
	mov s0, v1.s[3]
	fcmp s0, s8
	b.le .LBB11_14
.LBB11_22:
	fmov s1, w27
	fadd s0, s0, s1
	fmov s1, w20
	fdiv s0, s0, s1
	fmov s1, w22
	bl powf
	ldr q1, [sp, #304]
	mvni v2.4s, #128, lsl #24
	fmov s3, #1.00000000
	add x8, x21, x24
	mov s1, v1.s[3]
	bit v1.16b, v3.16b, v2.16b
	fmul s0, s1, s0
	str s0, [x8, #12]
	b .LBB11_14
.LBB11_23:
	ldr w8, [sp, #332]
	cbz w8, .LBB11_25
	ldp x2, x0, [sp, #16]
	ldr x1, [sp, #8]
	bl memcpy
.LBB11_25:
	add sp, sp, #656
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
.LBB11_26:
	.cfi_restore_state
	mov x19, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x19
	cbnz w0, .LBB11_2
.LBB11_27:
	mov x0, x25
	add sp, sp, #656
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
	b linear_srgb::simd::srgb_to_linear_extended_slice_tier_scalar_t
