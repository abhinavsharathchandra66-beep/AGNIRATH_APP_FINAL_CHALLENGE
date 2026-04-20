import os
import numpy as np
import matplotlib.pyplot as plt

from data_pipeline import load_route, calculate_irradiance
from optimizer import (
    run_optimizer, simulate_route, simulate_one_loop,
    run_loop_optimizer,
    AVAILABLE_TIME, ZEERUST_STOP, LOOP_DISTANCE, TOTAL_DISTANCE,
)
from physics import (
    calculate_net_power,
    calculate_drag_power, calculate_rolling_power,
    calculate_gravity_power, calculate_regen_power,
)

# ── HOUSEKEEPING ──────────────────────────────────────────────
if os.path.exists('optimal_velocities.npy'):
    os.remove('optimal_velocities.npy')
    print("Deleted cached velocities.")

# ── LOAD DATA ─────────────────────────────────────────────────
df                = load_route('route_with_slope__.csv')
slopes            = df['Slope_percent'].values
irradiance_values = calculate_irradiance(len(slopes))

print(f"Route segments : {len(slopes)}")
print(f"Max irradiance : {max(irradiance_values):.1f} W/m²")
print(f"Min irradiance : {min(irradiance_values):.1f} W/m²")

# ── PHASE 1 : BASE ROUTE ──────────────────────────────────────
result             = run_optimizer(slopes, irradiance_values)
optimal_velocities = result.x

if not result.success:
    print(f"\n⚠️  Optimiser warning: {result.message}")

distance, time, SOC_profile, acc_profile = simulate_route(
    optimal_velocities, slopes, irradiance_values
)

# Energy audit
total_solar  = sum(irr * 0.24 * 4.0 for irr in irradiance_values)
total_regen  = sum(calculate_regen_power(v, a)
                   for v, a in zip(optimal_velocities, acc_profile))
total_losses = sum(calculate_drag_power(v)
                   + calculate_rolling_power(v, s)
                   + calculate_gravity_power(v, s)
                   for v, s in zip(optimal_velocities, slopes))

print(f"\n── Energy audit ──")
print(f"  Solar input  : {total_solar:,.0f} W")
print(f"  Regen input  : {total_regen:,.0f} W")
print(f"  Mech losses  : {total_losses:,.0f} W")

print(f"\n✅ Reached Zeerust")
print(f"   Distance : {distance/1000:.2f} km")
print(f"   Time     : {time/3600:.2f} h")
print(f"   Final SOC: {SOC_profile[-1]*100:.1f}%")

# ── ZEERUST STOP ──────────────────────────────────────────────
print(f"\n⏸  30-min mandatory stop at Zeerust …")
remaining_time   = AVAILABLE_TIME - time - ZEERUST_STOP
remaining_SOC    = SOC_profile[-1]

loop_start_index = int(len(irradiance_values) * (time / AVAILABLE_TIME))
irradiance_avg   = float(np.mean(irradiance_values[loop_start_index:]))
print(f"   Loop irradiance avg: {irradiance_avg:.1f} W/m²")

# ── PHASE 2 : LOOPS ───────────────────────────────────────────
best_loops, best_velocity = run_loop_optimizer(
    remaining_time, remaining_SOC, irradiance_avg
)

loop_time, loop_soc_change = simulate_one_loop(best_velocity, irradiance_avg)
final_soc = remaining_SOC + best_loops * loop_soc_change

print(f"\n🔁 Loop strategy")
print(f"   Velocity       : {best_velocity*3.6:.1f} km/h")
print(f"   Loops completed: {best_loops}")
print(f"   Loop distance  : {best_loops * LOOP_DISTANCE/1000:.1f} km")
print(f"   SOC after loops: {final_soc*100:.1f}%")
print(f"\n🏁 Total distance: {(distance + best_loops*LOOP_DISTANCE)/1000:.2f} km")

if final_soc < 0.20:
    print("⚠️  WARNING: SOC drops below 20% during loops!")
else:
    print("✅ SOC stays above 20% throughout loops.")

# ── PLOTS ─────────────────────────────────────────────────────
dist_base = np.linspace(0, 303, len(optimal_velocities))

# Build full velocity/distance trace including loops
velocities_kmh    = optimal_velocities * 3.6
vel_zeerust_stop  = np.zeros(10)
dist_zeerust_stop = np.full(10, 303.0)

loop_vel_arrays  = []
loop_dist_arrays = []
cur_dist         = 303.0

for i in range(best_loops):
    loop_vel_arrays.append(np.full(35, best_velocity * 3.6))
    loop_dist_arrays.append(np.linspace(cur_dist, cur_dist + 35, 35))
    cur_dist += 35
    if i < best_loops - 1:          # inter-loop stop
        loop_vel_arrays.append(np.zeros(5))
        loop_dist_arrays.append(np.full(5, cur_dist))

if loop_vel_arrays:
    velocity_full = np.concatenate([velocities_kmh, vel_zeerust_stop,
                                    *loop_vel_arrays])
    distance_full = np.concatenate([dist_base, dist_zeerust_stop,
                                    *loop_dist_arrays])
else:
    velocity_full = np.concatenate([velocities_kmh, vel_zeerust_stop])
    distance_full = np.concatenate([dist_base, dist_zeerust_stop])

# Net power along base route
net_powers = [
    calculate_net_power(v, s, irr, a)
    for v, s, irr, a in zip(optimal_velocities, slopes,
                             irradiance_values, acc_profile)
]

soc_percent = [s * 100 for s in SOC_profile]
pos_acc     = [a if a > 0 else 0 for a in acc_profile]
neg_acc     = [a if a < 0 else 0 for a in acc_profile]
# Instead of concatenating zeros into the line, plot stops as scatter points
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot base route
axes[0,0].plot(dist_base, velocities_kmh, color='royalblue', label='Driving')

# Plot loops without connecting the stop gaps
cur_dist = 303.0
for i in range(best_loops):
    loop_d = np.linspace(cur_dist, cur_dist + 35, 35)
    axes[0,0].plot(loop_d, np.full(35, best_velocity * 3.6), color='royalblue')
    cur_dist += 35
    # Mark stop as a dot instead of dropping to zero
    axes[0,0].axvline(x=cur_dist, color='gray', linestyle=':', alpha=0.5)
    
# 1. Velocity profile
axes[0, 0].plot(distance_full, velocity_full, color='royalblue')
axes[0, 0].set_title('Velocity Profile')
axes[0, 0].set_xlabel('Distance (km)')
axes[0, 0].set_ylabel('Velocity (km/h)')
axes[0, 0].grid(True)

# 2. SOC profile
axes[0, 1].plot(dist_base, soc_percent, color='seagreen')
axes[0, 1].axhline(y=20, color='red', linestyle='--', label='Min 20%')
axes[0, 1].axhline(y=40, color='orange', linestyle='--', label='Min at Zeerust 40%')
axes[0, 1].set_title('State of Charge')
axes[0, 1].set_xlabel('Distance (km)')
axes[0, 1].set_ylabel('SOC (%)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Acceleration profile
axes[1, 0].fill_between(dist_base, pos_acc, alpha=0.5,
                         color='green', label='Accelerating')
axes[1, 0].fill_between(dist_base, neg_acc, alpha=0.5,
                         color='red', label='Braking / Regen')
axes[1, 0].axhline(y=0, color='black', linestyle='--')
axes[1, 0].set_title('Acceleration Profile')
axes[1, 0].set_xlabel('Distance (km)')
axes[1, 0].set_ylabel('Acceleration (m/s²)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. Net power profile
axes[1, 1].plot(dist_base, net_powers, color='darkorange')
axes[1, 1].axhline(y=0, color='red', linestyle='--', label='Zero line')
axes[1, 1].fill_between(dist_base, net_powers, 0,
                         where=[p > 0 for p in net_powers],
                         color='green', alpha=0.3, label='Charging')
axes[1, 1].fill_between(dist_base, net_powers, 0,
                         where=[p < 0 for p in net_powers],
                         color='red', alpha=0.3, label='Draining')
axes[1, 1].set_title('Net Power Profile')
axes[1, 1].set_xlabel('Distance (km)')
axes[1, 1].set_ylabel('Power (W)')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('strategy_profile.png', dpi=150)
plt.show()
print("\nPlot saved → strategy_profile.png")