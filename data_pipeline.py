import pandas as pd
import numpy as np

# ── IRRADIANCE MODEL ──────────────────────────────────────────
PEAK_IRRADIANCE    = 1073   # W/m²
SOLAR_NOON         = 12 * 3600          # seconds from midnight
SOLAR_STD_DEV      = 11600              # seconds (~3.2 h half-width)
DAYLIGHT_HALF_SPAN = 4 * 3600          # 4 h either side of noon

def load_route(file_path):
    df = pd.read_csv("route_with_slope__.csv")
    
    # 1. Cap the slope to a realistic maximum (e.g., 24%)
    limit = 24.0
    df['Slope_percent'] = df['Slope_percent'].clip(lower=-limit, upper=limit)
    
    # 2. Smooth the altitude to match the capped slopes
    # This prevents the 'spikes' in your graph from breaking the physics
    for i in range(1, len(df)):
        # New Altitude = Previous Altitude + Slope (since distance is 100m)
        df.loc[i, 'Altitude_m'] = df.loc[i-1, 'Altitude_m'] + df.loc[i-1, 'Slope_percent']
        
    return df


def calculate_irradiance(num_points,
                         start_time=None,
                         end_time=None):
    """
    Return a Gaussian irradiance profile (W/m²) sampled at num_points.

    start_time / end_time in seconds-from-midnight.
    Defaults to 8 AM → 5 PM (the available driving window).
    """
    if start_time is None:
        start_time = 8 * 3600
    if end_time is None:
        end_time = 17 * 3600

    t = np.linspace(start_time, end_time, num_points)
    return PEAK_IRRADIANCE * np.exp(
        -((t - SOLAR_NOON) ** 2) / (2 * SOLAR_STD_DEV ** 2)
    )



