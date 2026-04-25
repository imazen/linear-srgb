.section .text.stub_linear_to_srgb_extended_slice,"ax",@progbits
	.globl	stub_linear_to_srgb_extended_slice
	.p2align	2
.type	stub_linear_to_srgb_extended_slice,@function
stub_linear_to_srgb_extended_slice:
	.cfi_startproc
	sub sp, sp, #160
	.cfi_def_cfa_offset 160
	stp d15, d14, [sp, #48]
	stp d13, d12, [sp, #64]
	stp d11, d10, [sp, #80]
	stp d9, d8, [sp, #96]
	stp x29, x30, [sp, #112]
	str x21, [sp, #128]
	stp x20, x19, [sp, #144]
	add x29, sp, #112
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	.cfi_offset b8, -56
	.cfi_offset b9, -64
	.cfi_offset b10, -72
	.cfi_offset b11, -80
	.cfi_offset b12, -88
	.cfi_offset b13, -96
	.cfi_offset b14, -104
	.cfi_offset b15, -112
	lsl x8, x1, #2
	ands x9, x8, #0x7fffffffffffffc0
	b.eq .LBB8_3
	mov w10, #47186
	mov w11, #59837
	mov w12, #60277
	movk w10, #16718, lsl #16
	movk w11, #17200, lsl #16
	movk w12, #17737, lsl #16
	dup v0.4s, w10
	dup v1.4s, w11
	dup v2.4s, w12
	mov w10, #21186
	mov w11, #44370
	mov w12, #26919
	movk w10, #17949, lsl #16
	movk w11, #17885, lsl #16
	movk w12, #17536, lsl #16
	dup v3.4s, w10
	dup v4.4s, w11
	dup v5.4s, w12
	mov w10, #30049
	mov w11, #17028
	mov w12, #63323
	movk w10, #48797, lsl #16
	movk w11, #49027, lsl #16
	movk w12, #17310, lsl #16
	dup v6.4s, w10
	dup v7.4s, w11
	dup v16.4s, w12
	mov w10, #60486
	mov w11, #4173
	mov w12, #18007
	movk w10, #17793, lsl #16
	movk w11, #17952, lsl #16
	movk w12, #17852, lsl #16
	str q0, [sp]
	dup v23.4s, w10
	dup v18.4s, w11
	dup v0.4s, w12
	mov w10, #46319
	mov w11, #12900
	mov w12, #20545
	movk w10, #17487, lsl #16
	movk w11, #16798, lsl #16
	movk w12, #15175, lsl #16
	dup v24.4s, w10
	dup v21.4s, w11
	dup v22.4s, w12
	add x9, x0, x9
	mov x10, x0
.LBB8_2:
	ldp q19, q17, [x10]
	mov v8.16b, v2.16b
	mov v10.16b, v3.16b
	mov v11.16b, v23.16b
	mov v12.16b, v4.16b
	ldp q27, q26, [x10, #32]
	mov v13.16b, v18.16b
	fabs v25.4s, v19.4s
	fabs v28.4s, v17.4s
	mov v14.16b, v0.16b
	stp q17, q19, [sp, #16]
	mov v17.16b, v23.16b
	mov v20.16b, v24.16b
	fabs v29.4s, v27.4s
	mov v19.16b, v0.16b
	fcmlt v27.4s, v27.4s, #0.0
	fsqrt v30.4s, v25.4s
	fsqrt v31.4s, v28.4s
	fmla v8.4s, v1.4s, v30.4s
	fadd v9.4s, v30.4s, v16.4s
	fmla v10.4s, v30.4s, v8.4s
	fmla v11.4s, v30.4s, v9.4s
	fabs v9.4s, v26.4s
	fcmlt v26.4s, v26.4s, #0.0
	fsqrt v8.4s, v29.4s
	fmla v12.4s, v30.4s, v10.4s
	mov v10.16b, v2.16b
	fmla v13.4s, v30.4s, v11.4s
	mov v11.16b, v5.16b
	fadd v15.4s, v31.4s, v16.4s
	fmla v10.4s, v1.4s, v31.4s
	fmla v11.4s, v30.4s, v12.4s
	mov v12.16b, v3.16b
	fmla v14.4s, v30.4s, v13.4s
	mov v13.16b, v6.16b
	fmla v17.4s, v31.4s, v15.4s
	mov v15.16b, v21.16b
	fmla v12.4s, v31.4s, v10.4s
	fsqrt v10.4s, v9.4s
	fmla v13.4s, v30.4s, v11.4s
	mov v11.16b, v4.16b
	fmla v20.4s, v30.4s, v14.4s
	mov v14.16b, v18.16b
	fmla v11.4s, v31.4s, v12.4s
	mov v12.16b, v7.16b
	fmla v14.4s, v31.4s, v17.4s
	mov v17.16b, v2.16b
	fmla v15.4s, v30.4s, v20.4s
	mov v20.16b, v5.16b
	fmla v12.4s, v30.4s, v13.4s
	fadd v13.4s, v8.4s, v16.4s
	fmla v17.4s, v1.4s, v8.4s
	fmla v20.4s, v31.4s, v11.4s
	mov v11.16b, v3.16b
	fmla v19.4s, v31.4s, v14.4s
	mov v14.16b, v6.16b
	fdiv v30.4s, v12.4s, v15.4s
	mov v12.16b, v23.16b
	fmla v11.4s, v8.4s, v17.4s
	mov v17.16b, v24.16b
	fmla v14.4s, v31.4s, v20.4s
	mov v20.16b, v18.16b
	fmla v12.4s, v8.4s, v13.4s
	mov v13.16b, v7.16b
	fmla v17.4s, v31.4s, v19.4s
	mov v19.16b, v4.16b
	fmla v13.4s, v31.4s, v14.4s
	mov v14.16b, v0.16b
	fmla v19.4s, v8.4s, v11.4s
	fmla v20.4s, v8.4s, v12.4s
	mov v11.16b, v21.16b
	mov v12.16b, v2.16b
	fmla v11.4s, v31.4s, v17.4s
	fadd v17.4s, v10.4s, v16.4s
	mov v31.16b, v5.16b
	fmla v14.4s, v8.4s, v20.4s
	mov v20.16b, v23.16b
	fmla v12.4s, v1.4s, v10.4s
	fmla v31.4s, v8.4s, v19.4s
	mov v19.16b, v3.16b
	fmla v20.4s, v10.4s, v17.4s
	mov v17.16b, v6.16b
	fdiv v11.4s, v13.4s, v11.4s
	mov v13.16b, v18.16b
	fmla v19.4s, v10.4s, v12.4s
	mov v12.16b, v24.16b
	fmla v17.4s, v8.4s, v31.4s
	mov v31.16b, v4.16b
	fmla v13.4s, v10.4s, v20.4s
	mov v20.16b, v21.16b
	fmla v12.4s, v8.4s, v14.4s
	fmla v31.4s, v10.4s, v19.4s
	mov v19.16b, v7.16b
	fmla v19.4s, v8.4s, v17.4s
	fmla v20.4s, v8.4s, v12.4s
	mov v17.16b, v5.16b
	mov v8.16b, v0.16b
	fmla v17.4s, v10.4s, v31.4s
	mov v31.16b, v24.16b
	fmla v8.4s, v10.4s, v13.4s
	fdiv v19.4s, v19.4s, v20.4s
	mov v20.16b, v6.16b
	fmla v20.4s, v10.4s, v17.4s
	mov v17.16b, v7.16b
	fmla v31.4s, v10.4s, v8.4s
	mov v8.16b, v21.16b
	fmla v17.4s, v10.4s, v20.4s
	fmla v8.4s, v10.4s, v31.4s
	ldr q10, [sp]
	fmul v20.4s, v25.4s, v10.4s
	fcmgt v25.4s, v22.4s, v25.4s
	fmul v31.4s, v28.4s, v10.4s
	fcmgt v28.4s, v22.4s, v28.4s
	fdiv v17.4s, v17.4s, v8.4s
	fmul v8.4s, v29.4s, v10.4s
	fcmgt v29.4s, v22.4s, v29.4s
	fmul v10.4s, v9.4s, v10.4s
	fcmgt v9.4s, v22.4s, v9.4s
	bif v20.16b, v30.16b, v25.16b
	mov v25.16b, v28.16b
	ldp q30, q28, [sp, #16]
	bit v19.16b, v8.16b, v29.16b
	bsl v25.16b, v31.16b, v11.16b
	fcmlt v28.4s, v28.4s, #0.0
	fneg v29.4s, v20.4s
	fcmlt v30.4s, v30.4s, #0.0
	fneg v8.4s, v19.4s
	fneg v31.4s, v25.4s
	bit v20.16b, v29.16b, v28.16b
	bit v19.16b, v8.16b, v27.16b
	bit v17.16b, v10.16b, v9.16b
	bit v25.16b, v31.16b, v30.16b
	fneg v9.4s, v17.4s
	stp q20, q25, [x10]
	bit v17.16b, v9.16b, v26.16b
	stp q19, q17, [x10, #32]
	add x10, x10, #64
	cmp x10, x9
	b.ne .LBB8_2
.LBB8_3:
	ands x19, x8, #0x3c
	b.eq .LBB8_8
	and x8, x1, #0x1ffffffffffffff0
	mov w9, #20545
	add x20, x0, x8, lsl #2
	mov w8, #47186
	movk w9, #15175, lsl #16
	movk w8, #16718, lsl #16
	fmov s9, w9
	mov w9, #2711
	fmov s10, w8
	mov w8, #2143289344
	movk w9, #16263, lsl #16
	fmov s11, w8
	mov w8, #21845
	fmov s13, w9
	movk w8, #16085, lsl #16
	mov x21, x20
	fmov s8, w8
	mov w8, #21227
	movk w8, #48481, lsl #16
	fmov s12, w8
	b .LBB8_6
.LBB8_5:
	fmul s0, s1, s10
	subs x19, x19, #4
	str s0, [x20]
	mov x20, x21
	b.eq .LBB8_8
.LBB8_6:
	ldr s1, [x21], #4
	fabs s0, s1
	fcmp s0, s9
	b.mi .LBB8_5
	fmov s2, #1.00000000
	mvni v3.4s, #128, lsl #24
	fcmp s1, s1
	bif v2.16b, v1.16b, v3.16b
	fmov s1, s8
	fcsel s14, s11, s2, vs
	bl powf
	fmadd s0, s0, s13, s12
	fmul s0, s14, s0
	subs x19, x19, #4
	str s0, [x20]
	mov x20, x21
	b.ne .LBB8_6
.LBB8_8:
	.cfi_def_cfa wsp, 160
	ldp x20, x19, [sp, #144]
	ldr x21, [sp, #128]
	ldp x29, x30, [sp, #112]
	ldp d9, d8, [sp, #96]
	ldp d11, d10, [sp, #80]
	ldp d13, d12, [sp, #64]
	ldp d15, d14, [sp, #48]
	add sp, sp, #160
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
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
