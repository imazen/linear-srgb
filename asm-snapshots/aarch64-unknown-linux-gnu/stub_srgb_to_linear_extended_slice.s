.section .text.stub_srgb_to_linear_extended_slice,"ax",@progbits
	.globl	stub_srgb_to_linear_extended_slice
	.p2align	2
.type	stub_srgb_to_linear_extended_slice,@function
stub_srgb_to_linear_extended_slice:
	.cfi_startproc
	stp d15, d14, [sp, #-112]!
	.cfi_def_cfa_offset 112
	stp d13, d12, [sp, #16]
	stp d11, d10, [sp, #32]
	stp d9, d8, [sp, #48]
	stp x29, x30, [sp, #64]
	str x21, [sp, #80]
	stp x20, x19, [sp, #96]
	add x29, sp, #64
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
	b.eq .LBB11_3
	mov w10, #33681
	mov w11, #21091
	mov w12, #45243
	movk w10, #15774, lsl #16
	movk w11, #18053, lsl #16
	movk w12, #18434, lsl #16
	dup v0.4s, w10
	dup v1.4s, w11
	dup v2.4s, w12
	mov w10, #54856
	mov w11, #22505
	mov w12, #26646
	movk w10, #18518, lsl #16
	movk w11, #18375, lsl #16
	movk w12, #18037, lsl #16
	dup v3.4s, w10
	dup v4.4s, w11
	dup v5.4s, w12
	mov w10, #49826
	mov w11, #11201
	mov w12, #23593
	movk w10, #17507, lsl #16
	movk w11, #16784, lsl #16
	movk w12, #49790, lsl #16
	dup v6.4s, w10
	dup v7.4s, w11
	dup v16.4s, w12
	mov w10, #51710
	mov w11, #60686
	mov w12, #61050
	movk w10, #17803, lsl #16
	movk w11, #18336, lsl #16
	movk w12, #18528, lsl #16
	dup v17.4s, w10
	dup v18.4s, w11
	dup v19.4s, w12
	mov w10, #20961
	mov w11, #46089
	mov w12, #61974
	movk w10, #18451, lsl #16
	movk w11, #18088, lsl #16
	movk w12, #15648, lsl #16
	dup v20.4s, w10
	dup v21.4s, w11
	dup v22.4s, w12
	add x9, x0, x9
	mov x10, x0
.LBB11_2:
	ldp q24, q23, [x10]
	mov v26.16b, v2.16b
	mov v28.16b, v3.16b
	mov v29.16b, v17.16b
	mov v30.16b, v18.16b
	mov v8.16b, v5.16b
	mov v9.16b, v19.16b
	mov v10.16b, v17.16b
	fabs v25.4s, v24.4s
	mov v11.16b, v20.16b
	mov v12.16b, v2.16b
	mov v14.16b, v19.16b
	fcmlt v24.4s, v24.4s, #0.0
	fmla v26.4s, v1.4s, v25.4s
	fadd v27.4s, v25.4s, v16.4s
	fmla v28.4s, v25.4s, v26.4s
	fmla v29.4s, v25.4s, v27.4s
	fabs v26.4s, v23.4s
	mov v27.16b, v4.16b
	fcmlt v23.4s, v23.4s, #0.0
	fmla v27.4s, v25.4s, v28.4s
	fmla v30.4s, v25.4s, v29.4s
	mov v29.16b, v2.16b
	fadd v31.4s, v26.4s, v16.4s
	fmla v29.4s, v1.4s, v26.4s
	fmla v8.4s, v25.4s, v27.4s
	fmla v9.4s, v25.4s, v30.4s
	mov v30.16b, v3.16b
	fmla v10.4s, v26.4s, v31.4s
	mov v31.16b, v6.16b
	ldp q28, q27, [x10, #32]
	fmla v30.4s, v26.4s, v29.4s
	fmla v31.4s, v25.4s, v8.4s
	fmla v11.4s, v25.4s, v9.4s
	mov v8.16b, v4.16b
	mov v9.16b, v18.16b
	fabs v29.4s, v28.4s
	fcmlt v28.4s, v28.4s, #0.0
	fmla v8.4s, v26.4s, v30.4s
	mov v30.16b, v7.16b
	fmla v9.4s, v26.4s, v10.4s
	mov v10.16b, v21.16b
	fmla v12.4s, v1.4s, v29.4s
	fadd v13.4s, v29.4s, v16.4s
	fmla v30.4s, v25.4s, v31.4s
	fabs v31.4s, v27.4s
	fcmlt v27.4s, v27.4s, #0.0
	fmla v10.4s, v25.4s, v11.4s
	mov v11.16b, v5.16b
	fmla v14.4s, v26.4s, v9.4s
	mov v9.16b, v17.16b
	fmla v11.4s, v26.4s, v8.4s
	mov v8.16b, v3.16b
	fadd v15.4s, v31.4s, v16.4s
	fdiv v30.4s, v30.4s, v10.4s
	mov v10.16b, v6.16b
	fmla v9.4s, v29.4s, v13.4s
	mov v13.16b, v18.16b
	fmla v8.4s, v29.4s, v12.4s
	mov v12.16b, v20.16b
	fmla v10.4s, v26.4s, v11.4s
	mov v11.16b, v4.16b
	fmla v13.4s, v29.4s, v9.4s
	mov v9.16b, v21.16b
	fmla v12.4s, v26.4s, v14.4s
	mov v14.16b, v2.16b
	fmla v11.4s, v29.4s, v8.4s
	mov v8.16b, v7.16b
	fmla v14.4s, v1.4s, v31.4s
	fmla v8.4s, v26.4s, v10.4s
	mov v10.16b, v5.16b
	fmla v9.4s, v26.4s, v12.4s
	mov v12.16b, v19.16b
	fmla v10.4s, v29.4s, v11.4s
	mov v11.16b, v3.16b
	fmla v12.4s, v29.4s, v13.4s
	mov v13.16b, v17.16b
	fdiv v8.4s, v8.4s, v9.4s
	mov v9.16b, v6.16b
	fmla v11.4s, v31.4s, v14.4s
	mov v14.16b, v20.16b
	fmla v13.4s, v31.4s, v15.4s
	fmla v9.4s, v29.4s, v10.4s
	mov v10.16b, v4.16b
	fmla v14.4s, v29.4s, v12.4s
	mov v12.16b, v18.16b
	fmla v10.4s, v31.4s, v11.4s
	mov v11.16b, v7.16b
	fmla v12.4s, v31.4s, v13.4s
	mov v13.16b, v21.16b
	fmla v11.4s, v29.4s, v9.4s
	mov v9.16b, v5.16b
	fmla v13.4s, v29.4s, v14.4s
	mov v14.16b, v19.16b
	fmla v9.4s, v31.4s, v10.4s
	fmla v14.4s, v31.4s, v12.4s
	mov v12.16b, v20.16b
	fdiv v10.4s, v11.4s, v13.4s
	mov v11.16b, v6.16b
	mov v13.16b, v21.16b
	fmla v11.4s, v31.4s, v9.4s
	fmla v12.4s, v31.4s, v14.4s
	mov v9.16b, v7.16b
	fmul v14.4s, v31.4s, v0.4s
	fmla v9.4s, v31.4s, v11.4s
	fmla v13.4s, v31.4s, v12.4s
	fmul v11.4s, v25.4s, v0.4s
	fcmgt v25.4s, v22.4s, v25.4s
	fmul v12.4s, v26.4s, v0.4s
	fcmgt v26.4s, v22.4s, v26.4s
	fcmgt v31.4s, v22.4s, v31.4s
	fdiv v9.4s, v9.4s, v13.4s
	fmul v13.4s, v29.4s, v0.4s
	fcmgt v29.4s, v22.4s, v29.4s
	bsl v25.16b, v11.16b, v30.16b
	bsl v26.16b, v12.16b, v8.16b
	mov v30.16b, v31.16b
	bsl v29.16b, v13.16b, v10.16b
	fneg v31.4s, v25.4s
	fneg v8.4s, v26.4s
	bsl v24.16b, v31.16b, v25.16b
	bsl v23.16b, v8.16b, v26.16b
	mov v25.16b, v28.16b
	mov v26.16b, v27.16b
	bsl v30.16b, v14.16b, v9.16b
	fneg v9.4s, v29.4s
	stp q24, q23, [x10]
	fneg v10.4s, v30.4s
	bsl v25.16b, v9.16b, v29.16b
	bsl v26.16b, v10.16b, v30.16b
	stp q25, q26, [x10, #32]
	add x10, x10, #64
	cmp x10, x9
	b.ne .LBB11_2
.LBB11_3:
	ands x19, x8, #0x3c
	b.eq .LBB11_8
	and x8, x1, #0x1ffffffffffffff0
	mov w9, #61974
	add x20, x0, x8, lsl #2
	mov w8, #33681
	movk w9, #15648, lsl #16
	movk w8, #15774, lsl #16
	fmov s9, w9
	mov w9, #39322
	fmov s10, w8
	mov w8, #2143289344
	movk w9, #16409, lsl #16
	fmov s11, w8
	mov w8, #21227
	fmov s8, w9
	movk w8, #15713, lsl #16
	mov x21, x20
	fmov s12, w8
	mov w8, #2711
	movk w8, #16263, lsl #16
	fmov s13, w8
	b .LBB11_6
.LBB11_5:
	fmul s0, s1, s10
	subs x19, x19, #4
	str s0, [x20]
	mov x20, x21
	b.eq .LBB11_8
.LBB11_6:
	ldr s1, [x21], #4
	fabs s0, s1
	fcmp s0, s9
	b.mi .LBB11_5
	fadd s0, s0, s12
	fmov s2, #1.00000000
	mvni v3.4s, #128, lsl #24
	fcmp s1, s1
	bif v2.16b, v1.16b, v3.16b
	fmov s1, s8
	fdiv s0, s0, s13
	fcsel s14, s11, s2, vs
	bl powf
	fmul s0, s14, s0
	subs x19, x19, #4
	str s0, [x20]
	mov x20, x21
	b.ne .LBB11_6
.LBB11_8:
	.cfi_def_cfa wsp, 112
	ldp x20, x19, [sp, #96]
	ldr x21, [sp, #80]
	ldp x29, x30, [sp, #64]
	ldp d9, d8, [sp, #48]
	ldp d11, d10, [sp, #32]
	ldp d13, d12, [sp, #16]
	ldp d15, d14, [sp], #112
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
