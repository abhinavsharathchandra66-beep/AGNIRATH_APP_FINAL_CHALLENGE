import numpy as np

# ── VEHICLE CONSTANTS ─────────────────────────────────────────
PANEL_EFFICIENCY = 0.24
PANEL_AREA       = 6.0      # m²
FRONTAL_AREA     = 1.0      # m²
MASS             = 200      # kg
G                = 9.81     # m/s²
Cd               = 0.11     # drag coefficient
RHO              = 1.225    # air density kg/m³
Cr               = 0.004    # rolling resistance
MOTOR_EFFICIENCY = 0.95
REGEN_EFFICIENCY = 0.70
MAX_REGEN_POWER  = 500      # W  (motor/inverter cap)
battery_capacity = 3100     # Wh


def calculate_drag_power(velocity):
    """Aerodynamic drag power (W). Works on scalars or arrays."""
    return 0.5 * Cd * FRONTAL_AREA * RHO * velocity ** 3


def calculate_rolling_power(velocity, slope):
    """Rolling resistance power (W)."""
    angle = np.arctan(slope / 100)
    return Cr * MASS * G * np.cos(angle) * velocity


def calculate_gravity_power(velocity, slope):
    """Gravitational power (W). Positive = climbing, negative = descending."""
    angle = np.arctan(slope / 100)
    return MASS * G * np.sin(angle) * velocity


def calculate_regen_power(velocity, acceleration):
    """
    Regenerative braking power recovered (W).
    Only active when decelerating (acceleration < 0).
    Capped at MAX_REGEN_POWER.
    Supports both scalars and NumPy arrays.
    """
    braking_power   = MASS * velocity * np.abs(acceleration)
    regen_potential = braking_power * REGEN_EFFICIENCY
    actual_regen    = np.where(acceleration < 0, regen_potential, 0.0)
    return np.minimum(actual_regen, MAX_REGEN_POWER)


def calculate_net_power(velocity, slope, irradiance, acceleration):
    """
    Net power into the battery (W).
    Positive  → battery charging.
    Negative  → battery draining.
    """
    solar_power   = irradiance * PANEL_EFFICIENCY * PANEL_AREA
    drag_power    = calculate_drag_power(velocity)
    rolling_power = calculate_rolling_power(velocity, slope)
    gravity_power = calculate_gravity_power(velocity, slope)
    regen_power   = calculate_regen_power(velocity, acceleration)

    p_load = drag_power + rolling_power + gravity_power

    # Motor efficiency penalty only when driving forward against load
    battery_drawn = np.where(p_load > 0, p_load / MOTOR_EFFICIENCY, 0.0)

    return solar_power + regen_power - battery_drawn


def update_SOC(SOC, net_power, dt):
    """Update state-of-charge given net power (W) over time dt (s)."""
    energy_change = net_power * dt / 3600          # Wh
    return SOC + energy_change / battery_capacity