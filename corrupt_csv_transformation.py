import pandas as pd
import numpy as np

def process_telemetry():
    try:
        df = pd.read_csv('telemetry_data.csv')
    except FileNotFoundError:
        print("Error: 'telemetry_data.csv' not found.")
        return

    print("--- PART A: INITIAL AUDIT ---")
    print(f"Dataset Shape: {df.shape}")
    print("\nMissing Values:\n", df.isnull().sum())

    # --- b) Data Cleaning ---
    # Convert invalid data to NaN so we can interpolate them
    df.loc[df['velocity_kmh'] < 0, 'velocity_kmh'] = np.nan
    df.loc[df['battery_voltage'] > 160, 'battery_voltage'] = np.nan
    df.loc[df['solar_irradiance_wm2'] > 1200, 'solar_irradiance_wm2'] = np.nan
    
    # FIX: Only interpolate numeric columns to avoid the Dtype error
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    
    # Final backfill/forward fill for the edges
    df = df.bfill().ffill()
    print("\n--- PART B: CLEANING COMPLETE ---")

    # --- c) Power Input ---
    # Assumptions: Area = 4.0 m^2, Eff = 22%
    df['power_input_W'] = df['solar_irradiance_wm2'] * 0.22 * 4.0
    
    df.to_csv('cleaned_telemetry_data.csv', index=False)
    print("Success: 'cleaned_telemetry_data.csv' created.")
    print(df.head())

if __name__ == "__main__":
    process_telemetry()