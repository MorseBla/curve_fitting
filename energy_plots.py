import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
csv_path = "results.csv"
df = pd.read_csv(csv_path)

# Compute energy (power * time)
# avg_time_ms -> seconds for consistency
df["energy_J"] = df["avg_power_W"] * (df["avg_time_ms"] / 1000.0)

# Get unique benchmarks
benchmarks = df["benchmark"].unique()

for bench in benchmarks:
    bench_df = df[df["benchmark"] == bench].sort_values("frequency_MHz")

    plt.figure()
    plt.plot(
        bench_df["frequency_MHz"],
        bench_df["energy_J"],
        marker="o"
    )
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power × Time (J)")
    plt.title(f"Energy vs Frequency — {bench}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

