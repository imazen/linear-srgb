.section .text.stub_linear_to_srgb_extended_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_extended_slice
	.p2align	2
.type	stub_linear_to_srgb_extended_slice,@function
stub_linear_to_srgb_extended_slice:
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
	sub sp, sp, #672
	adrp x8, :got:archmage::tokens::generated::arm::NEON_CACHE
	mov x19, x0
	ldr x8, [x8, :got_lo12:archmage::tokens::generated::arm::NEON_CACHE]
	ldrb w8, [x8]
	cmp w8, #1
	b.eq .LBB13_25
	cmp w8, #2
	b.ne .LBB13_24
.LBB13_2:
	and x8, x1, #0x1ffffffffffffff0
	ands x20, x1, #0xf
	lsr x21, x1, #4
	add x22, x19, x8, lsl #2
	lsl x23, x20, #2
	stp x23, x22, [sp, #8]
	b.eq .LBB13_4
	mov w8, #64
	mov w1, wzr
	sub x2, x8, x23
	add x8, sp, #416
	add x0, x8, x23
	bl memset
	add x0, sp, #416
	mov x1, x22
	mov x2, x23
	bl memcpy
	ldp q0, q1, [sp, #416]
	add x8, sp, #97
	add x9, sp, #145
	stur q0, [x8, #255]
	ldp q0, q2, [sp, #448]
	add x8, sp, #113
	stur q1, [x8, #255]
	add x8, sp, #129
	stur q0, [x8, #255]
	mov w8, #1
	stur q2, [x9, #255]
	b .LBB13_5
.LBB13_4:
	mov w8, wzr
.LBB13_5:
	str w8, [sp, #348]
	add x8, x19, x21, lsl #6
	cmp x20, #0
	sub x9, x29, #192
	mov w20, #21845
	mov w28, #1115684864
	str x8, [sp, #296]
	add x8, sp, #348
	add x26, x9, #32
	add x8, x8, #4
	mov w9, #60277
	movk w20, #16085, lsl #16
	str x8, [sp]
	csel x8, xzr, x8, eq
	movk w9, #17737, lsl #16
	str x8, [sp, #264]
	mov w8, #47186
	fmov s9, w28
	movk w8, #16718, lsl #16
	fmov s8, w20
	mov w22, #21227
	dup v0.4s, w8
	mov w8, #59837
	mov w21, #2711
	movk w8, #17200, lsl #16
	sub x23, x29, #256
	movk w22, #48481, lsl #16
	dup v1.4s, w8
	mov w8, #21186
	movk w21, #16263, lsl #16
	str q0, [sp, #272]
	dup v0.4s, w9
	movk w8, #17949, lsl #16
	mov w9, #26919
	sub x24, x29, #256
	str w22, [sp, #28]
	movk w9, #17536, lsl #16
	stp q0, q1, [sp, #224]
	dup v1.4s, w8
	mov w8, #44370
	movk w8, #17885, lsl #16
	dup v0.4s, w8
	mov w8, #30049
	movk w8, #48797, lsl #16
	stp q0, q1, [sp, #192]
	dup v1.4s, w9
	dup v0.4s, w8
	mov w8, #17028
	mov w9, #63323
	movk w8, #49027, lsl #16
	movk w9, #17310, lsl #16
	stp q0, q1, [sp, #160]
	dup v1.4s, w8
	dup v0.4s, w9
	mov w8, #60486
	mov w9, #18007
	movk w8, #17793, lsl #16
	movk w9, #17852, lsl #16
	stp q0, q1, [sp, #128]
	dup v1.4s, w8
	mov w8, #4173
	movk w8, #17952, lsl #16
	dup v0.4s, w8
	mov w8, #46319
	movk w8, #17487, lsl #16
	stp q0, q1, [sp, #96]
	dup v1.4s, w9
	dup v0.4s, w8
	mov w8, #12900
	mov w9, #20545
	movk w8, #16798, lsl #16
	movk w9, #15175, lsl #16
	stp q0, q1, [sp, #64]
	dup v1.4s, w8
	dup v0.4s, w9
	stp q0, q1, [sp, #32]
	b .LBB13_7
.LBB13_6:
	ldp q0, q1, [x23]
	ldp q2, q3, [x23, #32]
	stp q0, q1, [x23, #64]
	stp q2, q3, [x23, #96]
	stp q0, q1, [x19]
	stp q2, q3, [x19, #32]
	mov x19, x25
.LBB13_7:
	ldr x8, [sp, #296]
	cmp x19, x8
	ccmp x19, #0, #4, ne
	b.ne .LBB13_10
	ldr x19, [sp, #264]
	cbz x19, .LBB13_21
	mov x25, xzr
	str xzr, [sp, #264]
	b .LBB13_11
.LBB13_10:
	add x25, x19, #64
.LBB13_11:
	ldp q0, q1, [x19]
	mov x27, xzr
	ldp q28, q27, [sp, #224]
	sub x8, x29, #192
	ldp q14, q13, [sp, #112]
	ldr q12, [sp, #144]
	fabs v2.4s, v0.4s
	fabs v3.4s, v1.4s
	stp q1, q0, [sp, #304]
	mov v17.16b, v28.16b
	ldp q30, q29, [sp, #192]
	mov v20.16b, v14.16b
	ldp q1, q15, [sp, #80]
	ldp q4, q5, [x19, #32]
	mov v26.16b, v28.16b
	fsqrt v16.4s, v2.4s
	mov v19.16b, v29.16b
	mov v21.16b, v30.16b
	mov v22.16b, v15.16b
	ldp q11, q31, [sp, #160]
	fabs v7.4s, v4.4s
	mov v24.16b, v1.16b
	stp q4, q5, [x23, #128]
	ldp q0, q10, [sp, #48]
	stp xzr, xzr, [x8, #96]
	mov v23.16b, v31.16b
	mov v25.16b, v10.16b
	fsqrt v6.4s, v3.4s
	fadd v18.4s, v16.4s, v13.4s
	fmla v17.4s, v27.4s, v16.4s
	fmla v20.4s, v16.4s, v18.4s
	fmla v19.4s, v16.4s, v17.4s
	fabs v17.4s, v5.4s
	fsqrt v18.4s, v7.4s
	fmla v22.4s, v16.4s, v20.4s
	fmla v21.4s, v16.4s, v19.4s
	mov v19.16b, v28.16b
	fadd v20.4s, v6.4s, v13.4s
	fmla v19.4s, v27.4s, v6.4s
	fmla v24.4s, v16.4s, v22.4s
	mov v22.16b, v14.16b
	fmla v23.4s, v16.4s, v21.4s
	mov v21.16b, v29.16b
	fmla v22.4s, v6.4s, v20.4s
	mov v20.16b, v11.16b
	fmla v25.4s, v16.4s, v24.4s
	mov v24.16b, v15.16b
	fmla v21.4s, v6.4s, v19.4s
	fsqrt v19.4s, v17.4s
	fmla v20.4s, v16.4s, v23.4s
	mov v23.16b, v30.16b
	fmla v26.4s, v27.4s, v18.4s
	fmla v24.4s, v6.4s, v22.4s
	mov v22.16b, v0.16b
	fmla v23.4s, v6.4s, v21.4s
	mov v21.16b, v12.16b
	fmla v22.4s, v16.4s, v25.4s
	mov v25.16b, v1.16b
	fmla v21.4s, v16.4s, v20.4s
	fadd v20.4s, v18.4s, v13.4s
	mov v16.16b, v31.16b
	fmla v25.4s, v6.4s, v24.4s
	mov v24.16b, v14.16b
	fmla v16.4s, v6.4s, v23.4s
	mov v23.16b, v29.16b
	fmla v24.4s, v18.4s, v20.4s
	fdiv v20.4s, v21.4s, v22.4s
	mov v22.16b, v10.16b
	mov v21.16b, v11.16b
	fmla v23.4s, v18.4s, v26.4s
	mov v26.16b, v28.16b
	fmla v22.4s, v6.4s, v25.4s
	mov v25.16b, v15.16b
	fmla v21.4s, v6.4s, v16.4s
	mov v16.16b, v30.16b
	fmla v26.4s, v27.4s, v19.4s
	fmla v25.4s, v18.4s, v24.4s
	mov v24.16b, v0.16b
	fmla v16.4s, v18.4s, v23.4s
	mov v23.16b, v12.16b
	fmla v24.4s, v6.4s, v22.4s
	mov v22.16b, v1.16b
	fmla v23.4s, v6.4s, v21.4s
	fadd v21.4s, v19.4s, v13.4s
	mov v6.16b, v31.16b
	fmla v22.4s, v18.4s, v25.4s
	mov v25.16b, v14.16b
	fmla v6.4s, v18.4s, v16.4s
	mov v16.16b, v29.16b
	fmla v25.4s, v19.4s, v21.4s
	fdiv v21.4s, v23.4s, v24.4s
	mov v23.16b, v11.16b
	mov v24.16b, v10.16b
	fmla v16.4s, v19.4s, v26.4s
	fmla v23.4s, v18.4s, v6.4s
	mov v6.16b, v30.16b
	fmla v24.4s, v18.4s, v22.4s
	mov v22.16b, v15.16b
	fmla v6.4s, v19.4s, v16.4s
	mov v16.16b, v12.16b
	fmla v22.4s, v19.4s, v25.4s
	mov v25.16b, v0.16b
	fmla v16.4s, v18.4s, v23.4s
	ldr q23, [sp, #272]
	fmla v25.4s, v18.4s, v24.4s
	mov v18.16b, v31.16b
	fmla v1.4s, v19.4s, v22.4s
	fmul v22.4s, v7.4s, v23.4s
	fmla v18.4s, v19.4s, v6.4s
	fdiv v6.4s, v16.4s, v25.4s
	mov v16.16b, v11.16b
	fmla v10.4s, v19.4s, v1.4s
	fmla v16.4s, v19.4s, v18.4s
	mov v18.16b, v12.16b
	fmla v0.4s, v19.4s, v10.4s
	ldr s10, [sp, #28]
	fmla v18.4s, v19.4s, v16.4s
	fmul v19.4s, v3.4s, v23.4s
	fdiv v16.4s, v18.4s, v0.4s
	ldr q0, [sp, #32]
	fmul v18.4s, v2.4s, v23.4s
	fmul v23.4s, v17.4s, v23.4s
	fcmgt v7.4s, v0.4s, v7.4s
	fcmgt v17.4s, v0.4s, v17.4s
	fcmgt v2.4s, v0.4s, v2.4s
	fcmgt v3.4s, v0.4s, v3.4s
	mvni v0.4s, #128, lsl #24
	bit v6.16b, v22.16b, v7.16b
	mov v7.16b, v17.16b
	bsl v2.16b, v18.16b, v20.16b
	bsl v3.16b, v19.16b, v21.16b
	bif v6.16b, v4.16b, v0.16b
	bsl v7.16b, v23.16b, v16.16b
	ldp q1, q16, [sp, #304]
	bif v2.16b, v16.16b, v0.16b
	bif v3.16b, v1.16b, v0.16b
	stp q16, q1, [x23, #96]
	bif v7.16b, v5.16b, v0.16b
	stp q2, q3, [x23]
	stp q6, q7, [x23, #32]
	b .LBB13_13
.LBB13_12:
	add x27, x27, #16
	cmp x27, #64
	b.eq .LBB13_6
.LBB13_13:
	ldr q0, [x26, x27]
	fabs v2.4s, v0.4s
	fcmp s2, s9
	stp q2, q0, [sp, #304]
	b.gt .LBB13_17
	mov s0, v2.s[1]
	fcmp s0, s9
	b.gt .LBB13_18
.LBB13_15:
	mov s0, v2.s[2]
	fmov s1, w28
	fcmp s0, s1
	b.gt .LBB13_19
.LBB13_16:
	mov s0, v2.s[3]
	fmov s1, w28
	fcmp s0, s1
	b.le .LBB13_12
	b .LBB13_20
.LBB13_17:
	fmov s0, s2
	fmov s1, s8
	bl powf
	fmov s1, w22
	fmov s2, w21
	ldr q3, [sp, #320]
	fmadd s0, s0, s2, s1
	mvni v1.4s, #128, lsl #24
	fmov s2, #1.00000000
	bsl v1.16b, v2.16b, v3.16b
	ldr q2, [sp, #304]
	fmul s0, s1, s0
	str s0, [x24, x27]
	mov s0, v2.s[1]
	fcmp s0, s9
	b.le .LBB13_15
.LBB13_18:
	fmov s1, s8
	bl powf
	ldr q1, [sp, #320]
	fmov s2, w21
	fmov s3, #1.00000000
	add x8, x24, x27
	mov s1, v1.s[1]
	fmadd s0, s0, s2, s10
	mvni v2.4s, #128, lsl #24
	bit v1.16b, v3.16b, v2.16b
	ldr q2, [sp, #304]
	fmul s0, s1, s0
	str s0, [x8, #4]
	mov s0, v2.s[2]
	fmov s1, w28
	fcmp s0, s1
	b.le .LBB13_16
.LBB13_19:
	fmov s1, w20
	bl powf
	ldr q1, [sp, #320]
	fmov s2, w22
	fmov s3, w21
	add x8, x24, x27
	mov s1, v1.s[2]
	fmadd s0, s0, s3, s2
	mvni v2.4s, #128, lsl #24
	fmov s3, #1.00000000
	bit v1.16b, v3.16b, v2.16b
	ldr q2, [sp, #304]
	fmul s0, s1, s0
	str s0, [x8, #8]
	mov s0, v2.s[3]
	fmov s1, w28
	fcmp s0, s1
	b.le .LBB13_12
.LBB13_20:
	fmov s1, w20
	bl powf
	ldr q1, [sp, #320]
	fmov s2, w22
	fmov s3, w21
	add x8, x24, x27
	mov s1, v1.s[3]
	fmadd s0, s0, s3, s2
	mvni v2.4s, #128, lsl #24
	fmov s3, #1.00000000
	bit v1.16b, v3.16b, v2.16b
	fmul s0, s1, s0
	str s0, [x8, #12]
	b .LBB13_12
.LBB13_21:
	ldr w8, [sp, #348]
	cbz w8, .LBB13_23
	ldp x2, x0, [sp, #8]
	ldr x1, [sp]
	bl memcpy
.LBB13_23:
	add sp, sp, #672
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
.LBB13_24:
	.cfi_restore_state
	mov x20, x1
	bl archmage::tokens::generated::arm::neon_detect
	mov x1, x20
	cbnz w0, .LBB13_2
.LBB13_25:
	mov x0, x19
	add sp, sp, #672
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
	b linear_srgb::simd::linear_to_srgb_extended_slice_tier_scalar_t
