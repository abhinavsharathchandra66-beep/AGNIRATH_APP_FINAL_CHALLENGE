
import numpy as np
from scipy.optimize import minimize
from physics import calculate_net_power, update_SOC, battery_capacity

# ── CONSTANTS ─────────────────────────────────────────────────
LEGAL_MAX_VELOCITY = 27.8       # m/s  (100 km/h)
LEGAL_MIN_VELOCITY = 1.0        # m/s
MAX_ACCELERATION   = 2.0        # m/s²
MIN_SOC            = 0.20       # absolute battery floor
MIN_SOC_ROUTE      = 0.40       # SOC floor at Zeerust arrival
AVAILABLE_TIME     = (17 - 8) * 3600   # 9 h driving window (s)
ZEERUST_STOP       = 30 * 60    # mandatory rest (s)
LOOP_STOP          = 5  * 60    # inter-loop stop (s)
LOOP_DISTANCE      = 35_000     # m per loop
TOTAL_DISTANCE     = 303_000    # m base route


# ── ROUTE SIMULATION ──────────────────────────────────────────
def simulate_route(velocities, slopes, irradiance_values):
    """
    Forward-simulate the base route.

    Returns
    -------
    distance      : float  (m)
    total_time    : float  (s)
    SOC_profile   : list[float]
    acc_profile   : list[float]
    """
    n                = len(velocities)
    dist_per_segment = TOTAL_DISTANCE / n
    SOC              = 1.0
    total_time       = 0.0
    SOC_profile      = []
    acc_profile      = []
    prev_v           = velocities[0]

    for i in range(n):
        v          = velocities[i]
        segment_dt = dist_per_segment / v

        acceleration = 0.0 if i == 0 else (v - prev_v) / segment_dt

        net_power  = calculate_net_power(v, slopes[i],
                                         irradiance_values[i], acceleration)
        SOC        = update_SOC(SOC, net_power, segment_dt)
        SOC        = float(np.clip(SOC, 0.0, 1.0))

        total_time    += segment_dt
        SOC_profile.append(SOC)
        acc_profile.append(acceleration)
        prev_v = v

    return TOTAL_DISTANCE, total_time, SOC_profile, acc_profile


# ── SINGLE LOOP SIMULATION ────────────────────────────────────
def simulate_one_loop(velocity, irradiance_avg):
    """
    Simulate one 35 km loop at constant velocity.

    Returns
    -------
    loop_time   : float  (s)
    soc_change  : float  (fraction)
    """
    if velocity <= 0:
        return 0.0, 0.0

    segment_len = 1000          # m
    n_segments  = LOOP_DISTANCE // segment_len
    soc_change  = 0.0
    time        = 0.0

    for _ in range(int(n_segments)):
        dt         = segment_len / velocity
        net_power  = calculate_net_power(velocity, 0.0, irradiance_avg, 0.0)
        soc_change += net_power * dt / 3600 / battery_capacity
        time       += dt

    return time, soc_change


# ── LOOP OPTIMIZER ────────────────────────────────────────────
def run_loop_optimizer(remaining_time, remaining_SOC, irradiance_avg):
    """
    Brute-force search for the velocity that completes the most loops
    while keeping SOC ≥ MIN_SOC.

    Returns
    -------
    best_loops    : int
    best_velocity : float  (m/s)
    """
    best_loops    = 0
    best_velocity = 0.0

    for v in np.arange(LEGAL_MIN_VELOCITY, LEGAL_MAX_VELOCITY, 0.5):
        loop_time, loop_soc_change = simulate_one_loop(v, irradiance_avg)
        total_loop_time = loop_time + LOOP_STOP
        if total_loop_time <= 0:
            continue

        n_loops   = int(remaining_time // total_loop_time)
        total_soc = remaining_SOC + n_loops * loop_soc_change

        if total_soc >= MIN_SOC and n_loops > best_loops:
            best_loops    = n_loops
            best_velocity = v

    return best_loops, best_velocity


# ── OBJECTIVE & CONSTRAINTS ───────────────────────────────────
def _objective(velocities, slopes, irradiance_values):
    _, time, _, _ = simulate_route(velocities, slopes, irradiance_values)
    smoothness    = np.sum(np.diff(velocities) ** 2) * 0.05
    return time / 3600 + smoothness


def _soc_constraint(velocities, slopes, irradiance_values):
    _, _, SOC_profile, _ = simulate_route(velocities, slopes, irradiance_values)
    return (SOC_profile[-1] - MIN_SOC_ROUTE) * 100.0      # ≥ 0


def _time_constraint(velocities, slopes, irradiance_values):
    _, time, _, _ = simulate_route(velocities, slopes, irradiance_values)
    return (AVAILABLE_TIME - ZEERUST_STOP - time) / 3600   # ≥ 0


def _accel_constraint(velocities, slopes, irradiance_values):
    _, _, _, acc_profile = simulate_route(velocities, slopes, irradiance_values)
    return MAX_ACCELERATION - np.abs(acc_profile)           # each element ≥ 0


# ── MAIN OPTIMIZER ────────────────────────────────────────────
def run_optimizer(slopes, irradiance_values, n_opt=150):
    """
    SLSQP optimisation of the velocity profile for the base route.

    Downsamples to n_opt control points so the gradient computation
    stays tractable (3030 segments → ~150 variables → fast convergence).
    The result is interpolated back to the original resolution.

    Returns scipy OptimizeResult; result.x contains velocities at the
    ORIGINAL resolution (same length as slopes).
    """
    n_full = len(slopes)

    # ── Downsample slopes & irradiance to n_opt points ────────
    idx_opt       = np.linspace(0, n_full - 1, n_opt).astype(int)
    slopes_opt    = slopes[idx_opt]
    irr_opt       = irradiance_values[idx_opt]

    constraints = [
        {'type': 'ineq',
         'fun': lambda v: _soc_constraint(v, slopes_opt, irr_opt)},
        {'type': 'ineq',
         'fun': lambda v: _time_constraint(v, slopes_opt, irr_opt)},
        {'type': 'ineq',
         'fun': lambda v: _accel_constraint(v, slopes_opt, irr_opt)},
    ]

    initial = np.full(n_opt, 15.0)
    initial = np.clip(initial, LEGAL_MIN_VELOCITY, LEGAL_MAX_VELOCITY)
    initial[:10] = np.linspace(2.0, 15.0, 10)

    bounds = [(LEGAL_MIN_VELOCITY, LEGAL_MAX_VELOCITY)] * n_opt

    print(f"Optimising over {n_opt} control points (downsampled from {n_full}) …")
    result = minimize(
        _objective,
        x0          = initial,
        args        = (slopes_opt, irr_opt),
        method      = 'SLSQP',
        bounds      = bounds,
        constraints = constraints,
        options     = {'maxiter': 500, 'ftol': 1e-5, 'disp': True},
    )

    # ── Interpolate back to full resolution ───────────────────
    x_full       = np.linspace(0, 1, n_full)
    x_opt        = np.linspace(0, 1, n_opt)
    full_velocities        = np.interp(x_full, x_opt, result.x)
    full_velocities        = np.clip(full_velocities,
                                     LEGAL_MIN_VELOCITY, LEGAL_MAX_VELOCITY)

    # Wrap in a result-like object so callers use result.x as before
    result.x = full_velocities
    return result