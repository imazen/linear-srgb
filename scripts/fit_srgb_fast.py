"""
Fit C0-continuous sRGB rational polynomials (fast version).
"""
import numpy as np
from scipy.optimize import least_squares
import sys

sys.stdout.reconfigure(line_buffering=True)

# C0-continuous (moxcms) constants
THRESHOLD_LINEAR = 0.003041282560127521
THRESHOLD_GAMMA = 12.92 * THRESHOLD_LINEAR
A = 0.0550107189475866
A_PLUS_1 = 1.0 + A

def srgb_to_linear_exact(v):
    if v <= THRESHOLD_GAMMA: return v / 12.92
    return ((v + A) / A_PLUS_1) ** 2.4

def linear_to_srgb_exact(v):
    if v <= THRESHOLD_LINEAR: return v * 12.92
    return A_PLUS_1 * v ** (1.0/2.4) - A

srgb_to_linear_v = np.vectorize(srgb_to_linear_exact)
linear_to_srgb_v = np.vectorize(linear_to_srgb_exact)

def eval_rp_arr(x, p, q):
    yp = np.full_like(x, p[4], dtype=np.float64)
    for i in range(3, -1, -1): yp = yp * x + p[i]
    yq = np.full_like(x, q[4], dtype=np.float64)
    for i in range(3, -1, -1): yq = yq * x + q[i]
    return yp / yq

def eval_rp_sc(x, p, q):
    yp = p[4]
    for i in range(3, -1, -1): yp = yp * x + p[i]
    yq = q[4]
    for i in range(3, -1, -1): yq = yq * x + q[i]
    return yp / yq

def eval_rp_f32(x, p, q):
    x = np.float32(x)
    yp = np.float32(p[4])
    for i in range(3, -1, -1): yp = np.float32(np.float32(yp * x) + np.float32(p[i]))
    yq = np.float32(q[4])
    for i in range(3, -1, -1): yq = np.float32(np.float32(yq * x) + np.float32(q[i]))
    return np.float32(yp / yq)

# Existing libjxl coefficients
OLD_S2L_P = np.array([2.200_248_3e-4, 1.043_637_6e-2, 1.624_820_4e-1, 7.961_565e-1, 8.210_153e-1])
OLD_S2L_Q = np.array([2.631_847e-1, 1.076_976_5, 4.987_528_3e-1, -5.512_498_3e-2, 6.521_209e-3])
OLD_L2S_P = np.array([-5.135_152_6e-4, 5.287_254_7e-3, 3.903_843e-1, 1.474_205_3, 7.352_63e-1])
OLD_L2S_Q = np.array([1.004_519_6e-2, 3.036_675_5e-1, 1.340_817, 9.258_482e-1, 2.424_867_8e-2])

u16_step = 1.0 / 65535.0

# ================================================================
# sRGB -> linear
# ================================================================
print("Fitting sRGB -> linear..."); sys.stdout.flush()

n = 20000
t = np.linspace(0, 1, n)
t_cheb = 0.5 * (1 - np.cos(np.pi * t))
v_s2l = THRESHOLD_GAMMA + (1.0 - THRESHOLD_GAMMA) * t_cheb
y_s2l = srgb_to_linear_v(v_s2l)

boundary_x = THRESHOLD_GAMMA
boundary_y = THRESHOLD_LINEAR

best = None
best_cost = float('inf')

for restart in range(5):
    q_last = OLD_S2L_Q[4]
    p0 = OLD_S2L_P / q_last
    q0 = OLD_S2L_Q / q_last
    if restart > 0:
        rng = np.random.default_rng(restart * 37)
        p0 *= 1 + 0.1 * rng.standard_normal(5)
        q0 *= 1 + 0.1 * rng.standard_normal(5)
    params0 = np.concatenate([p0, q0[:4]])

    def res(params):
        p = params[:5]; q = np.concatenate([params[5:9], [1.0]])
        main = (eval_rp_arr(v_s2l, p, q) - y_s2l) / np.maximum(np.abs(y_s2l), 1e-20)
        bnd = (eval_rp_sc(boundary_x, p, q) - boundary_y) / boundary_y * 1e6
        return np.append(main, bnd)

    try:
        r = least_squares(res, params0, method='lm', max_nfev=50000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
        if r.cost < best_cost: best_cost = r.cost; best = r
    except: pass

p_s2l = best.x[:5]; q_s2l = np.concatenate([best.x[5:9], [1.0]])

# Check
pred = eval_rp_arr(v_s2l, p_s2l, q_s2l)
err = np.abs(pred - y_s2l)
bnd_pred = eval_rp_sc(boundary_x, p_s2l, q_s2l)

p_f32 = np.array(p_s2l, dtype=np.float32); q_f32 = np.array(q_s2l, dtype=np.float32)
thresh_f32 = np.float32(THRESHOLD_GAMMA)
lin_seg = np.float32(thresh_f32 * np.float32(1.0/12.92))
poly_at = eval_rp_f32(thresh_f32, p_f32, q_f32)
ulp = abs(int(lin_seg.view(np.uint32)) - int(poly_at.view(np.uint32)))

print(f"  Max abs error: {err.max():.6e}")
print(f"  Boundary err (f64): {abs(bnd_pred - boundary_y):.6e}")
print(f"  f32 boundary ULP: {ulp} (was 110)")
print(f"  Coeffs P: {p_s2l}")
print(f"  Coeffs Q: {q_s2l}")

# ================================================================
# linear -> sRGB
# ================================================================
print("\nFitting linear -> sRGB..."); sys.stdout.flush()

v_l2s = THRESHOLD_LINEAR + (1.0 - THRESHOLD_LINEAR) * t_cheb
s_l2s = np.sqrt(v_l2s)
y_l2s = linear_to_srgb_v(v_l2s)

boundary_s = np.sqrt(THRESHOLD_LINEAR)
boundary_y2 = THRESHOLD_GAMMA

best2 = None; best_cost2 = float('inf')

for restart in range(5):
    q_last = OLD_L2S_Q[4]
    p0 = OLD_L2S_P / q_last; q0 = OLD_L2S_Q / q_last
    if restart > 0:
        rng = np.random.default_rng(restart * 53)
        p0 *= 1 + 0.1 * rng.standard_normal(5)
        q0 *= 1 + 0.1 * rng.standard_normal(5)
    params0 = np.concatenate([p0, q0[:4]])

    def res2(params):
        p = params[:5]; q = np.concatenate([params[5:9], [1.0]])
        main = (eval_rp_arr(s_l2s, p, q) - y_l2s) / np.maximum(np.abs(y_l2s), 1e-20)
        bnd = (eval_rp_sc(boundary_s, p, q) - boundary_y2) / boundary_y2 * 1e6
        return np.append(main, bnd)

    try:
        r = least_squares(res2, params0, method='lm', max_nfev=50000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
        if r.cost < best_cost2: best_cost2 = r.cost; best2 = r
    except: pass

p_l2s = best2.x[:5]; q_l2s = np.concatenate([best2.x[5:9], [1.0]])

pred2 = eval_rp_arr(s_l2s, p_l2s, q_l2s)
err2 = np.abs(pred2 - y_l2s)
bnd_pred2 = eval_rp_sc(boundary_s, p_l2s, q_l2s)

p2_f32 = np.array(p_l2s, dtype=np.float32); q2_f32 = np.array(q_l2s, dtype=np.float32)
thresh2_f32 = np.float32(THRESHOLD_LINEAR)
gam_seg = np.float32(thresh2_f32 * np.float32(12.92))
sqrt_t = np.float32(np.sqrt(float(thresh2_f32)))
poly_at2 = eval_rp_f32(sqrt_t, p2_f32, q2_f32)
ulp2 = abs(int(gam_seg.view(np.uint32)) - int(poly_at2.view(np.uint32)))

print(f"  Max abs error: {err2.max():.6e}")
print(f"  Boundary err (f64): {abs(bnd_pred2 - boundary_y2):.6e}")
print(f"  f32 boundary ULP: {ulp2} (was 19)")
print(f"  Coeffs P: {p_l2s}")
print(f"  Coeffs Q: {q_l2s}")

# ================================================================
# Combined roundtrip
# ================================================================
print("\nFull roundtrip..."); sys.stdout.flush()

v_full = np.linspace(0.0, 1.0, 100000)
mask1 = v_full <= THRESHOLD_GAMMA
fwd = np.where(mask1, v_full / 12.92, eval_rp_arr(v_full, p_s2l, q_s2l))
mask2 = fwd <= THRESHOLD_LINEAR
sqrt_fwd = np.sqrt(np.maximum(fwd, 0))
inv = np.where(mask2, fwd * 12.92, eval_rp_arr(sqrt_fwd, p_l2s, q_l2s))
rt = np.abs(inv - v_full)
print(f"  Max roundtrip: {rt.max():.6e} ({rt.max()/u16_step:.1f} U16)")
print(f"  > 1 U16: {(rt > u16_step).sum()}/{len(v_full)}")

# ================================================================
# Print Rust format
# ================================================================
print(f"\n{'='*60}")
print("Rust coefficients:")
print(f"{'='*60}")
print(f"pub(crate) const SRGB_THRESHOLD: f32 = {THRESHOLD_GAMMA};")
print(f"pub(crate) const LINEAR_THRESHOLD: f32 = {THRESHOLD_LINEAR};")
for name, arr in [("S2L_P", p_s2l), ("S2L_Q", q_s2l), ("L2S_P", p_l2s), ("L2S_Q", q_l2s)]:
    parts = ", ".join(f"{c:.8e}" for c in arr)
    print(f"pub(crate) const {name}: [f32; 5] = [{parts}];")
