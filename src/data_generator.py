import pandas as pd
import numpy as np

def generate_indian_5g_data(num_samples=10000):
    np.random.seed(42)

    timestamps = pd.date_range(start="2023-01-01", periods=num_samples, freq="s")
    mobility = np.random.choice([0, 20, 60, 100], p=[0.3, 0.3, 0.3, 0.1], size=num_samples)
    rssi = np.random.normal(loc=-85, scale=6, size=num_samples)
    sinr = np.random.normal(loc=15, scale=4, size=num_samples)
    rsrp = np.random.normal(loc=-95, scale=5, size=num_samples)
    rsrq = np.random.normal(loc=-10, scale=2, size=num_samples)

    # Location simulation
    latitudes = np.random.uniform(19.0, 28.5, num_samples)   # India bounds
    longitudes = np.random.uniform(72.8, 88.0, num_samples)

    bandwidth = (
        50 + sinr * 2.5 - mobility * 0.5 + np.random.normal(0, 5, num_samples)
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "latitude": latitudes,
        "longitude": longitudes,
        "mobility": mobility,
        "RSSI": rssi,
        "SINR": sinr,
        "RSRP": rsrp,
        "RSRQ": rsrq,
        "bandwidth": np.clip(bandwidth, 0, None)
    })

    return df

if __name__ == "__main__":
    df = generate_indian_5g_data()
    df.to_csv("data/simulated_indian_5g.csv", index=False)
    print("✅ Simulated Indian dataset saved.")
