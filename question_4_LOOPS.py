import numpy as np
from scipy.optimize import root_scalar

# ==========================================
# STEP 1: DEFINE CONSTANTS
# ==========================================
T_max = 1.5 * 3600         # 1.5 hours in seconds
L = 30 * 1000              # 30 km loop in meters
E_avail = (1.5 - 0.2) * 3.6e6 # 1.3 kWh available in Joules
P_solar = 250              # Watts
k = 0.09                   # Aero drag coefficient
P_loss = 100               # Base mechanical/electrical losses
eta = 0.90                 # Motor efficiency

# ==========================================
# STEP 2: DEFINE FORMULAS (Functions)
# ==========================================
def calc_vmin(N):
    """Calculates the minimum speed required to beat the clock."""
    distance = N * L
    return distance / T_max 

def energy_margin(v, N):
    """Calculates Energy Available minus Energy Consumed at speed v."""
    distance = N * L
    time_on_track = distance / v
    P_mech = k * (v**3) + P_loss
    P_draw = (P_mech / eta) - P_solar
    E_consumed = P_draw * time_on_track
    return E_avail - E_consumed

def calc_vmax(N):
    """Uses a numerical root solver to find the exact speed where Battery hits Emin."""
    # We ask scipy to find the velocity (v) where energy_margin exactly equals 0
    try:
        sol = root_scalar(energy_margin, args=(N,), bracket=[1, 50]) # bracket in m/s
        return sol.root
    except ValueError:
        return 0 # If it fails to find a root, vmax is effectively 0

# ==========================================
# STEP 3: LOOP CONDITION
# ==========================================
max_loops = 0
recommended_vmin = 0
recommended_vmax = 0

# Test N from 1 loop up to 10 loops
for N in range(1, 11):
    vmin = calc_vmin(N)
    vmax = calc_vmax(N)
    
    # The ultimate feasibility check:
    if vmin <= vmax:
        max_loops = N
        recommended_vmin = vmin
        recommended_vmax = vmax
    else:
        # As soon as vmin > vmax, the battery will die before we finish in time.
        break 

# ==========================================
# STEP 4: PRINT RETURN
# ==========================================
print(f"--- STRATEGY REPORT ---")
print(f"Maximum Feasible Loops: {max_loops}")
print(f"Speed Range (m/s) : {recommended_vmin:.2f} to {recommended_vmax:.2f}")
print(f"Speed Range (km/h): {recommended_vmin * 3.6:.1f} to {recommended_vmax * 3.6:.1f}")
print(f"Target Velocity   : {recommended_vmin * 3.6 + 1:.1f} km/h (Safest energy margin)")
        
