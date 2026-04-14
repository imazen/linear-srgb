"""
Extended-range sRGB rational polynomial fitter.
Uses existing [0,1] coefficients as warm start, multiple perturbation
strategies, and Chebyshev-spaced sample points for stability.
"""
import numpy as np
from scipy.optimize import least_squares
import sys, time
sys.stdout.reconfigure(line_buffering=True)

THRESHOLD_LINEAR = 0.003041282560127521
THRESHOLD_GAMMA = 12.92 * THRESHOLD_LINEAR
A = 0.0550107189475866
A1 = 1.0 + A
u16_half = 0.5 / 65535.0
u8_half = 0.5 / 255.0

def s2l_exact(v):
    return np.where(v <= THRESHOLD_GAMMA, v / 12.92, ((v + A) / A1) ** 2.4)
def l2s_exact(v):
    return np.where(v <= THRESHOLD_LINEAR, v * 12.92, A1 * np.abs(v) ** (1.0/2.4) - A)

def horner_v(x, coeffs):
    p, q = coeffs
    yp = np.full_like(x, p[-1])
    for i in range(len(p)-2, -1, -1): yp = yp * x + p[i]
    yq = np.full_like(x, q[-1])
    for i in range(len(q)-2, -1, -1): yq = yq * x + q[i]
    return yp / yq

def horner_s(x, coeffs):
    p, q = coeffs
    yp = float(p[-1])
    for i in range(len(p)-2, -1, -1): yp = yp * x + float(p[i])
    yq = float(q[-1])
    for i in range(len(q)-2, -1, -1): yq = yq * x + float(q[i])
    return yp / yq

def ulp_dist_v(a, b):
    a32 = np.float32(a); b32 = np.float32(b)
    return np.abs(a32.view(np.int32).astype(np.int64) - b32.view(np.int32).astype(np.int64))

# Current 4/4 coefficients
CUR = {
    's2l': (np.array([1.724_942_4e-2, 8.335_514_7e-1, 1.326_215_8e1, 7.033_073_4e1, 8.387_046e1]),
            np.array([2.066_183e1, 9.917_607e1, 5.466_011e1, -7.183_806, 1.0])),
    'l2s': (np.array([-1.513_885e-2, 1.167_372_8e-1, 1.257_921_2e1, 5.259_309_8e1, 2.852_907_6e1]),
            np.array([2.943_901_4e-1, 9.779_103, 4.726_487_7e1, 3.546_463_8e1, 1.0])),
}

def extend_coeffs(p4, q4, nc):
    """Extend 5-element coefficients to nc by padding with small values."""
    p = np.zeros(nc); p[:5] = p4
    q = np.zeros(nc); q[:5] = q4; q[-1] = 1.0
    return p, q

def fit(nc, x_nodes, y_target, bnd_x, bnd_y, seed_p, seed_q, n_restarts=20):
    """Fit a degree (nc-1)/(nc-1) rational polynomial."""
    def make_res(x, y):
        def res(params):
            p = params[:nc]; q = np.concatenate([params[nc:2*nc-1], [1.0]])
            pred = horner_v(x, (p, q))
            main = (pred - y) / np.maximum(np.abs(y), 1e-20)
            bnd = (horner_s(bnd_x, (p, q)) - bnd_y) / bnd_y * 1e6
            return np.append(main, bnd)
        return res

    res = make_res(x_nodes, y_target)
    best = None; best_cost = float('inf')

    # Seed strategies:
    # 0: direct extension of [0,1] coefficients
    # 1-4: scaled versions (the wider domain needs different magnitudes)
    # 5+: random perturbations of the best found so far
    for i in range(n_restarts):
        rng = np.random.default_rng(i * 31 + 17)
        if i == 0:
            p0, q0 = extend_coeffs(seed_p, seed_q, nc)
        elif i < 5:
            p0, q0 = extend_coeffs(seed_p, seed_q, nc)
            scale = [0.5, 2.0, 0.1, 5.0][i-1]
            p0 *= scale; q0[:-1] *= scale
        elif best is not None:
            p0 = best.x[:nc] * (1 + 0.1 * rng.standard_normal(nc))
            q0_full = np.concatenate([best.x[nc:2*nc-1], [1.0]])
            q0_full[:-1] *= (1 + 0.1 * rng.standard_normal(nc-1))
            p0 = p0; q0 = q0_full
        else:
            p0 = rng.standard_normal(nc); p0[0] = bnd_y
            q0 = np.zeros(nc); q0[0] = 1.0; q0[-1] = 1.0

        params0 = np.concatenate([p0, q0[:nc-1]])
        try:
            r = least_squares(res, params0, method='lm', max_nfev=100000,
                             ftol=1e-15, xtol=1e-15, gtol=1e-15)
            if r.cost < best_cost:
                best_cost = r.cost; best = r
        except: pass

    p = best.x[:nc]; q = np.concatenate([best.x[nc:2*nc-1], [1.0]])
    return p, q, best_cost

def evaluate(label, p, q, nc, upper, exact_fn, xform, domain_lo):
    v01 = np.linspace(max(domain_lo, 1e-6), 1.0, 100000)
    max_ulp = ulp_dist_v(horner_v(xform(v01), (p, q)), exact_fn(v01)).max()

    x_full = np.linspace(max(domain_lo, 1e-6), upper, 500000)
    err = np.abs(horner_v(xform(x_full), (p, q)) - exact_fn(x_full))
    u16_e = np.where(err >= u16_half)[0]
    u8_e = np.where(err >= u8_half)[0]
    u16_bnd = x_full[u16_e[0]] if len(u16_e) else upper
    u8_bnd = x_full[u8_e[0]] if len(u8_e) else upper

    print(f"  {label}: ULP[0,1]={max_ulp:>4}  u16≤{u16_bnd:.2f}  u8≤{u8_bnd:.2f}", flush=True)
    print(f"  P: [{', '.join(f'{c:.10e}' for c in p)}]")
    print(f"  Q: [{', '.join(f'{c:.10e}' for c in q)}]")
    return max_ulp, u16_bnd, u8_bnd

n = 30000
t = np.linspace(0, 1, n)
t_cheb = 0.5 * (1 - np.cos(np.pi * t))

configs = [
    # (direction, degree, upper_domain)
    ('s2l', 5, 4),
    ('s2l', 5, 8),
    ('s2l', 6, 4),
    ('s2l', 6, 8),
    ('l2s', 5, 64),
    ('l2s', 6, 64),
]

for direction, deg, upper in configs:
    nc = deg + 1
    t0 = time.time()

    if direction == 's2l':
        lo = THRESHOLD_GAMMA
        v = lo + (upper - lo) * t_cheb
        y = s2l_exact(v)
        bnd_x, bnd_y = THRESHOLD_GAMMA, THRESHOLD_LINEAR
        seed_p, seed_q = CUR['s2l']
        exact_fn, xform, domain_lo = s2l_exact, lambda x: x, THRESHOLD_GAMMA
    else:
        lo = THRESHOLD_LINEAR
        v_lin = lo + (upper - lo) * t_cheb
        v = np.sqrt(v_lin)  # polynomial input is sqrt(linear)
        y = l2s_exact(v_lin)
        bnd_x = np.sqrt(THRESHOLD_LINEAR)
        bnd_y = THRESHOLD_GAMMA
        seed_p, seed_q = CUR['l2s']
        exact_fn, xform, domain_lo = l2s_exact, np.sqrt, THRESHOLD_LINEAR

    p, q, cost = fit(nc, v, y, bnd_x, bnd_y, seed_p, seed_q, n_restarts=20)
    elapsed = time.time() - t0

    label = f"{direction.upper()} {deg}/{deg} [0,{upper}]"
    print(f"\n{label} ({elapsed:.1f}s, cost={cost:.2e})", flush=True)
    evaluate(label, p, q, nc, upper, exact_fn, xform, domain_lo)
