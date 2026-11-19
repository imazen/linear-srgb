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
	sub sp, sp, #624
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	mov x19, x0
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB16_25
	cmp w8, #2
	b.ne .LBB16_24
.LBB16_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x20, x1, #0xf
	lsr x22, x1, #4
	add x23, x19, x8, lsl #2
	lsl x21, x20, #2
	str x23, [sp, #8]
	b.eq .LBB16_4
	mov w8, #64
	mov w1, wzr
	sub x2, x8, x21
	add x8, sp, #368
	add x0, x8, x21
	bl memset
	add x0, sp, #368
	mov x1, x23
	mov x2, x21
	bl memcpy
	ldp q0, q1, [sp, #368]
	add x8, sp, #49
	add x9, sp, #97
	stur q0, [x8, #255]
	ldp q0, q2, [sp, #400]
	add x8, sp, #65
	stur q1, [x8, #255]
	add x8, sp, #81
	stur q0, [x8, #255]
	mov w8, #1
	stur q2, [x9, #255]
	b .LBB16_5
.LBB16_4:
	mov w8, wzr
.LBB16_5:
	str w8, [sp, #300]
	add x8, sp, #300
	cmp x20, #0
	add x9, x8, #4
	mov w8, #33681
	fmov s9, #8.00000000
	str x9, [sp]
	csel x26, xzr, x9, eq
	mov w9, #21091
	movk w8, #15774, lsl #16
	movk w9, #18053, lsl #16
	add x24, x19, x22, lsl #6
	dup v1.4s, w8
	dup v0.4s, w9
	mov w8, #45243
	movk w8, #18434, lsl #16
	mov w9, #22505
	sub x25, x29, #192
	movk w9, #18375, lsl #16
	sub x23, x29, #256
	add x27, x25, #32
	stp q0, q1, [sp, #224]
	dup v1.4s, w8
	mov w8, #54856
	movk w8, #18518, lsl #16
	sub x28, x29, #256
	dup v0.4s, w8
	mov w8, #26646
	movk w8, #18037, lsl #16
	stp q0, q1, [sp, #192]
	dup v1.4s, w9
	dup v0.4s, w8
	mov w8, #49826
	mov w9, #11201
	movk w8, #17507, lsl #16
	movk w9, #16784, lsl #16
	stp q0, q1, [sp, #160]
	dup v1.4s, w8
	dup v0.4s, w9
	mov w8, #23593
	mov w9, #60686
	movk w8, #49790, lsl #16
	movk w9, #18336, lsl #16
	stp q0, q1, [sp, #128]
	dup v1.4s, w8
	mov w8, #51710
	movk w8, #17803, lsl #16
	dup v0.4s, w8
	mov w8, #61050
	movk w8, #18528, lsl #16
	stp q0, q1, [sp, #96]
	dup v1.4s, w9
	dup v0.4s, w8
	mov w8, #20961
	mov w9, #46089
	movk w8, #18451, lsl #16
	movk w9, #18088, lsl #16
	stp q0, q1, [sp, #64]
	dup v1.4s, w8
	dup v0.4s, w9
	mov w8, #61974
	mov w9, #2711
	movk w8, #15648, lsl #16
	movk w9, #16263, lsl #16
	stp q0, q1, [sp, #32]
	dup v0.4s, w8
	mov w8, #21227
	movk w8, #15713, lsl #16
	fmov s11, w9
	fmov s10, w8
	mov w8, #39322
	movk w8, #16409, lsl #16
	str q0, [sp, #16]
	fmov s8, w8
	b .LBB16_7
.LBB16_6:
	ldp q0, q1, [x23]
	ldp q2, q3, [x23, #32]
	stp q0, q1, [x23, #64]
	stp q2, q3, [x23, #96]
	stp q0, q1, [x19]
	stp q2, q3, [x19, #32]
	mov x19, x20
.LBB16_7:
	cmp x19, x24
	ccmp x19, #0, #4, ne
	b.ne .LBB16_10
	cbz x26, .LBB16_21
	mov x20, xzr
	mov x19, x26
	mov x26, xzr
	b .LBB16_11
.LBB16_10:
	add x20, x19, #64
.LBB16_11:
	ldp q1, q0, [x19]
	mov x22, xzr
	ldp q23, q24, [sp, #208]
	ldr q29, [sp, #128]
	ldp q31, q30, [sp, #96]
	fabs v2.4s, v1.4s
	ldp q26, q25, [sp, #176]
	mov v3.16b, v23.16b
	ldp q13, q12, [sp, #64]
	mov v6.16b, v31.16b
	ldp q28, q27, [sp, #144]
	mov v5.16b, v25.16b
	mov v19.16b, v31.16b
	mov v21.16b, v23.16b
	fmla v3.4s, v24.4s, v2.4s
	fadd v4.4s, v2.4s, v30.4s
	mov v7.16b, v12.16b
	mov v17.16b, v27.16b
	mov v18.16b, v13.16b
	ldp q15, q14, [sp, #32]
	fmla v5.4s, v2.4s, v3.4s
	fmla v6.4s, v2.4s, v4.4s
	fabs v3.4s, v0.4s
	mov v4.16b, v26.16b
	mov v20.16b, v14.16b
	fmla v4.4s, v2.4s, v5.4s
	fmla v7.4s, v2.4s, v6.4s
	mov v6.16b, v23.16b
	fadd v16.4s, v3.4s, v30.4s
	fmla v6.4s, v24.4s, v3.4s
	fmla v17.4s, v2.4s, v4.4s
	fmla v18.4s, v2.4s, v7.4s
	mov v7.16b, v25.16b
	fmla v19.4s, v3.4s, v16.4s
	mov v16.16b, v28.16b
	ldp q5, q4, [x19, #32]
	stp q1, q0, [x23, #96]
	fmla v7.4s, v3.4s, v6.4s
	stp xzr, xzr, [x25, #96]
	fmla v16.4s, v2.4s, v17.4s
	fmla v20.4s, v2.4s, v18.4s
	mov v17.16b, v26.16b
	mov v18.16b, v12.16b
	fabs v6.4s, v5.4s
	stp q5, q4, [x23, #128]
	fmla v17.4s, v3.4s, v7.4s
	mov v7.16b, v29.16b
	fmla v18.4s, v3.4s, v19.4s
	mov v19.16b, v15.16b
	fmla v21.4s, v24.4s, v6.4s
	fadd v22.4s, v6.4s, v30.4s
	fmla v7.4s, v2.4s, v16.4s
	mov v16.16b, v27.16b
	fmla v19.4s, v2.4s, v20.4s
	mov v20.16b, v13.16b
	fmla v16.4s, v3.4s, v17.4s
	mov v17.16b, v25.16b
	fmla v20.4s, v3.4s, v18.4s
	mov v18.16b, v31.16b
	fdiv v7.4s, v7.4s, v19.4s
	mov v19.16b, v28.16b
	fmla v17.4s, v6.4s, v21.4s
	mov v21.16b, v14.16b
	fmla v18.4s, v6.4s, v22.4s
	fabs v22.4s, v4.4s
	fmla v19.4s, v3.4s, v16.4s
	mov v16.16b, v26.16b
	fmla v21.4s, v3.4s, v20.4s
	mov v20.16b, v12.16b
	fmla v16.4s, v6.4s, v17.4s
	mov v17.16b, v29.16b
	fmla v23.4s, v24.4s, v22.4s
	fmla v20.4s, v6.4s, v18.4s
	mov v18.16b, v15.16b
	fadd v24.4s, v22.4s, v30.4s
	fmla v17.4s, v3.4s, v19.4s
	mov v19.16b, v27.16b
	fmla v18.4s, v3.4s, v21.4s
	mov v21.16b, v13.16b
	fmla v19.4s, v6.4s, v16.4s
	mov v16.16b, v25.16b
	fmla v21.4s, v6.4s, v20.4s
	mov v20.16b, v31.16b
	fdiv v17.4s, v17.4s, v18.4s
	mov v18.16b, v28.16b
	fmla v16.4s, v22.4s, v23.4s
	mov v23.16b, v14.16b
	fmla v20.4s, v22.4s, v24.4s
	ldr q24, [sp, #16]
	fmla v18.4s, v6.4s, v19.4s
	mov v19.16b, v26.16b
	fmla v23.4s, v6.4s, v21.4s
	mov v21.16b, v12.16b
	fmla v19.4s, v22.4s, v16.4s
	mov v16.16b, v29.16b
	fmla v21.4s, v22.4s, v20.4s
	mov v20.16b, v15.16b
	fmla v16.4s, v6.4s, v18.4s
	mov v18.16b, v27.16b
	fmla v20.4s, v6.4s, v23.4s
	mov v23.16b, v13.16b
	fmla v18.4s, v22.4s, v19.4s
	mov v19.16b, v28.16b
	fmla v23.4s, v22.4s, v21.4s
	mov v21.16b, v15.16b
	fdiv v16.4s, v16.4s, v20.4s
	mov v20.16b, v14.16b
	fmla v19.4s, v22.4s, v18.4s
	mov v18.16b, v29.16b
	fmla v20.4s, v22.4s, v23.4s
	ldr q23, [sp, #240]
	fmla v18.4s, v22.4s, v19.4s
	fmul v19.4s, v2.4s, v23.4s
	fcmgt v2.4s, v24.4s, v2.4s
	fmla v21.4s, v22.4s, v20.4s
	fmul v20.4s, v3.4s, v23.4s
	fcmgt v3.4s, v24.4s, v3.4s
	bsl v2.16b, v19.16b, v7.16b
	fdiv v18.4s, v18.4s, v21.4s
	fmul v21.4s, v6.4s, v23.4s
	fmul v23.4s, v22.4s, v23.4s
	fcmgt v22.4s, v24.4s, v22.4s
	fcmgt v6.4s, v24.4s, v6.4s
	bsl v3.16b, v20.16b, v17.16b
	mov v7.16b, v22.16b
	bsl v6.16b, v21.16b, v16.16b
	mvni v16.4s, #128, lsl #24
	bif v2.16b, v1.16b, v16.16b
	bit v0.16b, v3.16b, v16.16b
	mov v1.16b, v16.16b
	mov v3.16b, v16.16b
	bsl v1.16b, v6.16b, v5.16b
	stp q2, q0, [x23]
	bsl v7.16b, v23.16b, v18.16b
	bsl v3.16b, v7.16b, v4.16b
	stp q1, q3, [x23, #32]
	b .LBB16_13
.LBB16_12:
	add x22, x22, #16
	cmp x22, #64
	b.eq .LBB16_6
.LBB16_13:
	ldr q0, [x27, x22]
	fabs v1.4s, v0.4s
	fcmp s1, s9
	stp q1, q0, [sp, #256]
	b.gt .LBB16_17
	mov s0, v1.s[1]
	fcmp s0, s9
	b.gt .LBB16_18
.LBB16_15:
	mov s0, v1.s[2]
	fcmp s0, s9
	b.gt .LBB16_19
.LBB16_16:
	mov s0, v1.s[3]
	fcmp s0, s9
	b.le .LBB16_12
	b .LBB16_20
.LBB16_17:
	fadd s0, s1, s10
	fmov s1, s8
	fdiv s0, s0, s11
	bl powf
	mvni v1.4s, #128, lsl #24
	fmov s2, #1.00000000
	ldr q3, [sp, #272]
	bsl v1.16b, v2.16b, v3.16b
	fmul s0, s1, s0
	ldr q1, [sp, #256]
	str s0, [x28, x22]
	mov s0, v1.s[1]
	fcmp s0, s9
	b.le .LBB16_15
.LBB16_18:
	fadd s0, s0, s10
	fmov s1, s8
	fdiv s0, s0, s11
	bl powf
	ldr q1, [sp, #272]
	mvni v2.4s, #128, lsl #24
	fmov s3, #1.00000000
	add x8, x28, x22
	mov s1, v1.s[1]
	bit v1.16b, v3.16b, v2.16b
	fmul s0, s1, s0
	ldr q1, [sp, #256]
	str s0, [x8, #4]
	mov s0, v1.s[2]
	fcmp s0, s9
	b.le .LBB16_16
.LBB16_19:
	fadd s0, s0, s10
	fmov s1, s8
	fdiv s0, s0, s11
	bl powf
	ldr q1, [sp, #272]
	mvni v2.4s, #128, lsl #24
	fmov s3, #1.00000000
	add x8, x28, x22
	mov s1, v1.s[2]
	bit v1.16b, v3.16b, v2.16b
	fmul s0, s1, s0
	ldr q1, [sp, #256]
	str s0, [x8, #8]
	mov s0, v1.s[3]
	fcmp s0, s9
	b.le .LBB16_12
.LBB16_20:
	fadd s0, s0, s10
	fmov s1, s8
	fdiv s0, s0, s11
	bl powf
	ldr q1, [sp, #272]
	mvni v2.4s, #128, lsl #24
	fmov s3, #1.00000000
	add x8, x28, x22
	mov s1, v1.s[3]
	bit v1.16b, v3.16b, v2.16b
	fmul s0, s1, s0
	str s0, [x8, #12]
	b .LBB16_12
.LBB16_21:
	ldr w8, [sp, #300]
	cbz w8, .LBB16_23
	ldp x1, x0, [sp]
	mov x2, x21
	bl memcpy
.LBB16_23:
	add sp, sp, #624
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
.LBB16_24:
	.cfi_restore_state
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	cbnz w0, .LBB16_2
.LBB16_25:
	mov x0, x19
	add sp, sp, #624
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
