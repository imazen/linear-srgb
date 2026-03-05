"""
Fit two-range degree-4 rational polynomials for PQ EOTF (signal -> linear).

The existing single polynomial covers [0.02, 1.0] with input transform x = v + v*v.
It has good absolute accuracy but poor relative accuracy at low signal values,
causing ~185 U16 steps of roundtrip error at signal=0.02.

Strategy: split into two ranges like the inverse direction already does.
Minimize RELATIVE error (which corresponds to signal-space roundtrip error).
"""
import numpy as np
from scipy.optimize import least_squares

# PQ constants
M1 = 0.1593017578125
M2 = 78.84375
C1 = 0.8359375
C2 = 18.8515625
C3 = 18.6875

def pq_exact(v):
    """PQ EOTF exact (f64): signal -> linear"""
    if v <= 0:
        return 0.0
    vp = v ** (1.0 / M2)
    num = max(vp - C1, 0.0)
    den = C2 - C3 * vp
    if den <= 0:
        return 1.0
    return (num / den) ** (1.0 / M1)

pq_exact_v = np.vectorize(pq_exact)

def pq_inv_exact(y):
    """PQ inverse EOTF exact (f64): linear -> signal"""
    if y <= 0:
        return 0.0
    ym1 = y ** M1
    num = C1 + C2 * ym1
    den = 1.0 + C3 * ym1
    return (num / den) ** M2

pq_inv_exact_v = np.vectorize(pq_inv_exact)

def eval_rp_array(x_arr, p, q):
    """Evaluate rational polynomial P(x)/Q(x) on an array using Horner's method."""
    yp = np.full_like(x_arr, p[4], dtype=np.float64)
    for i in range(3, -1, -1):
        yp = yp * x_arr + p[i]
    yq = np.full_like(x_arr, q[4], dtype=np.float64)
    for i in range(3, -1, -1):
        yq = yq * x_arr + q[i]
    return yp / yq

def eval_rp_scalar(x, p, q):
    """Evaluate rational polynomial P(x)/Q(x) on a scalar using Horner's method."""
    yp = p[4]
    for i in range(3, -1, -1):
        yp = yp * x + p[i]
    yq = q[4]
    for i in range(3, -1, -1):
        yq = yq * x + q[i]
    return yp / yq

# Existing coefficients for reference
EXISTING_P = np.array([2.6297566e-4, -6.235531e-3, 7.386023e-1, 2.6455317, 5.500349e-1])
EXISTING_Q = np.array([4.213501e2, -4.2873682e2, 1.7436467e2, -3.3907887e1, 2.6771877])

def fit_range(v_lo, v_hi, n_samples=100000, use_relative=True, n_restarts=8):
    """Fit a degree-4/4 rational polynomial for PQ EOTF over [v_lo, v_hi]."""
    # Use Chebyshev-spaced nodes for better conditioning
    t = np.linspace(0, 1, n_samples)
    t_cheb = 0.5 * (1 - np.cos(np.pi * t))
    v_data = v_lo + (v_hi - v_lo) * t_cheb

    x_data = v_data + v_data**2
    y_data = pq_exact_v(v_data)

    best_result = None
    best_cost = float('inf')

    for restart in range(n_restarts):
        if restart == 0:
            # Start from existing coefficients (normalized to monic Q)
            q_last = EXISTING_Q[4]
            p0 = EXISTING_P / q_last
            q0 = EXISTING_Q / q_last
            params0 = np.concatenate([p0, q0[:4]])
        else:
            # Random perturbation of the starting point
            rng = np.random.default_rng(restart * 42 + 7)
            q_last = EXISTING_Q[4]
            p0 = EXISTING_P / q_last
            q0 = EXISTING_Q / q_last
            scale = 0.3 if restart < 4 else 1.0
            params0 = np.concatenate([
                p0 * (1 + scale * rng.standard_normal(5)),
                q0[:4] * (1 + scale * rng.standard_normal(4))
            ])

        def residuals(params):
            p = params[:5]
            q = np.concatenate([params[5:9], [1.0]])
            y_pred = eval_rp_array(x_data, p, q)
            if use_relative:
                return (y_pred - y_data) / np.maximum(np.abs(y_data), 1e-30)
            else:
                return y_pred - y_data

        try:
            result = least_squares(residuals, params0, method='lm', max_nfev=100000,
                                   ftol=1e-15, xtol=1e-15, gtol=1e-15)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                # print(f"  Restart {restart}: cost = {result.cost:.6e}")
        except Exception as e:
            pass  # silently skip failed restarts

    if best_result is None:
        raise RuntimeError(f"All restarts failed for range [{v_lo}, {v_hi}]")

    p_fit = best_result.x[:5]
    q_fit = np.concatenate([best_result.x[5:9], [1.0]])

    return p_fit, q_fit, v_data, x_data, y_data

def evaluate_accuracy(p, q, v_data, x_data, y_data, label):
    """Print accuracy metrics for a fitted polynomial."""
    y_pred = eval_rp_array(x_data, p, q)
    abs_err = np.abs(y_pred - y_data)
    rel_err = abs_err / np.maximum(np.abs(y_data), 1e-30)

    # Also check roundtrip error (signal -> linear -> signal)
    signal_back = pq_inv_exact_v(y_pred)
    rt_err = np.abs(signal_back - v_data)
    u16_step = 1.0 / 65535.0

    print(f"\n{label}:")
    print(f"  Signal range: [{v_data[0]:.4f}, {v_data[-1]:.4f}]")
    print(f"  Linear range: [{y_data.min():.6e}, {y_data.max():.6e}]")
    print(f"  Max abs error (linear): {abs_err.max():.6e} at signal {v_data[abs_err.argmax()]:.4f}")
    print(f"  Max rel error (linear): {rel_err.max():.6e} at signal {v_data[rel_err.argmax()]:.4f}")
    print(f"  Max roundtrip error (signal): {rt_err.max():.6e} ({rt_err.max()/u16_step:.1f} U16 steps)")
    print(f"  Roundtrip > 1 U16 step: {(rt_err > u16_step).sum()}/{len(v_data)}")

    # Show errors at key points
    for v_test in [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]:
        if v_test < v_data[0] - 0.001 or v_test > v_data[-1] + 0.001:
            continue
        x_test = v_test + v_test**2
        y_true = pq_exact(v_test)
        y_approx = eval_rp_scalar(x_test, p, q)
        abs_e = abs(y_approx - y_true)
        rel_e = abs_e / max(abs(y_true), 1e-30)
        sig_back = pq_inv_exact(y_approx)
        rt_e = abs(sig_back - v_test)
        print(f"    v={v_test:.2f}: abs={abs_e:.3e}, rel={rel_e:.3e}, rt={rt_e:.3e} ({rt_e/u16_step:.1f} U16)")

def format_coeffs(name, coeffs):
    """Format coefficients as Rust array."""
    parts = [f"{c:.8e}" for c in coeffs]
    formatted = ", ".join(parts)
    return f"pub(crate) const {name}: [f32; 5] = [{formatted}];"

# ============================================================
# Baseline: existing single polynomial
# ============================================================
u16_step = 1.0 / 65535.0
print("=" * 70)
print("Baseline: existing single polynomial")
print("=" * 70)
v_all = np.linspace(0.02, 1.0, 200000)
x_all = v_all + v_all**2
y_all = pq_exact_v(v_all)
y_pred_existing = eval_rp_array(x_all, EXISTING_P, EXISTING_Q)
rt_existing = np.abs(pq_inv_exact_v(y_pred_existing) - v_all)
print(f"  Roundtrip: max {rt_existing.max():.6e} ({rt_existing.max()/u16_step:.1f} U16 steps)")
print(f"  Roundtrip > 1 U16 step: {(rt_existing > u16_step).sum()}/{len(v_all)}")

# ============================================================
# Fit two ranges
# ============================================================
print(f"\n{'='*70}")
print("Fitting PQ EOTF two-range rational polynomials")
print("=" * 70)

# Try different split points
best_threshold = None
best_combined_max = float('inf')
best_coeffs = None

for threshold in [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25]:
    print(f"\n{'='*70}")
    print(f"Split threshold: signal = {threshold}")
    print(f"{'='*70}")

    print("\nFitting small range...")
    p_small, q_small, v_s, x_s, y_s = fit_range(0.02, threshold, use_relative=True)
    evaluate_accuracy(p_small, q_small, v_s, x_s, y_s, f"Small range [0.02, {threshold}]")

    print("\nFitting large range...")
    p_large, q_large, v_l, x_l, y_l = fit_range(threshold, 1.0, use_relative=True)
    evaluate_accuracy(p_large, q_large, v_l, x_l, y_l, f"Large range [{threshold}, 1.0]")

    # Check full-range combined
    v_full = np.linspace(0.02, 1.0, 200000)
    x_full = v_full + v_full**2
    y_full = pq_exact_v(v_full)

    mask_small = v_full < threshold
    y_pred_full = np.zeros_like(v_full)
    y_pred_full[mask_small] = eval_rp_array(x_full[mask_small], p_small, q_small)
    y_pred_full[~mask_small] = eval_rp_array(x_full[~mask_small], p_large, q_large)

    rt_full = np.abs(pq_inv_exact_v(y_pred_full) - v_full)
    combined_max = rt_full.max()
    combined_over = (rt_full > u16_step).sum()
    print(f"\n  COMBINED roundtrip: max {combined_max:.6e} ({combined_max/u16_step:.1f} U16 steps)")
    print(f"  COMBINED > 1 U16 step: {combined_over}/{len(v_full)}")

    # Check the boundary continuity
    x_at_thresh = threshold + threshold**2
    y_small_at = eval_rp_scalar(x_at_thresh, p_small, q_small)
    y_large_at = eval_rp_scalar(x_at_thresh, p_large, q_large)
    y_exact_at = pq_exact(threshold)
    print(f"  Boundary: small={y_small_at:.8e}, large={y_large_at:.8e}, exact={y_exact_at:.8e}")
    print(f"  Boundary discontinuity: {abs(y_small_at - y_large_at):.6e}")

    if combined_max < best_combined_max:
        best_combined_max = combined_max
        best_threshold = threshold
        best_coeffs = (p_small.copy(), q_small.copy(), p_large.copy(), q_large.copy())

    print(f"\n  Coefficients (Rust format):")
    print(f"  {format_coeffs('PQ_EOTF_P_SMALL', p_small)}")
    print(f"  {format_coeffs('PQ_EOTF_Q_SMALL', q_small)}")
    print(f"  {format_coeffs('PQ_EOTF_P_LARGE', p_large)}")
    print(f"  {format_coeffs('PQ_EOTF_Q_LARGE', q_large)}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"BEST: threshold = {best_threshold}, max roundtrip = {best_combined_max:.6e} ({best_combined_max/u16_step:.1f} U16 steps)")
print(f"{'='*70}")
p_s, q_s, p_l, q_l = best_coeffs
print(f"\n  {format_coeffs('PQ_EOTF_P_SMALL', p_s)}")
print(f"  {format_coeffs('PQ_EOTF_Q_SMALL', q_s)}")
print(f"  {format_coeffs('PQ_EOTF_P_LARGE', p_l)}")
print(f"  {format_coeffs('PQ_EOTF_Q_LARGE', q_l)}")
print(f"  Threshold: {best_threshold}")
