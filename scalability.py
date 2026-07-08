import csv
import pandas as pd
import matplotlib.pyplot as plt

from tsp_bm import main

# -------------------------------------------------------
# Read first n cities from master CSV
# -------------------------------------------------------
MASTER_CSV = "cities.csv"      # contains all cities


def create_subset_csv(n, output_file="temp_cities.csv"):
    df = pd.read_csv(MASTER_CSV)

    subset = df.iloc[:n]

    subset.to_csv(output_file, index=False)

    return output_file


results = []

# -------------------------------------------------------
# CSV to store benchmark results
# -------------------------------------------------------

with open("benchmark_results.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Cities",
        "Runtime_sec",
        "Optimal_Distance"
    ])

    for n in range(10, 101, 10):

        print(f"\nRunning benchmark for {n} cities...")

        filename = create_subset_csv(n)

        runtime, distance = main(filename)

        results.append((n, runtime, distance))

        writer.writerow([
            n,
            runtime,
            distance
        ])

        print(
            f"{n:3d} cities | "
            f"Runtime = {runtime:.4f} sec | "
            f"Distance = {distance:.4f}"
        )

# -------------------------------------------------------
# Plot Runtime
# -------------------------------------------------------

cities = [r[0] for r in results]
runtimes = [r[1] for r in results]

plt.figure(figsize=(8,5))

plt.plot(
    cities,
    runtimes,
    marker='o',
    linewidth=2
)

plt.xlabel("Number of Cities")
plt.ylabel("Runtime (seconds)")
plt.title("Exact TSP Solver Scalability")
plt.grid(True)

plt.savefig(
    "exact_tsp_scalability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -------------------------------------------------------
# Plot Tour Length
# -------------------------------------------------------

distances = [r[2] for r in results]

plt.figure(figsize=(8,5))

plt.plot(
    cities,
    distances,
    marker='o',
    linewidth=2
)

plt.xlabel("Number of Cities")
plt.ylabel("Optimal Tour Length (km)")
plt.title("Optimal Tour Length vs Number of Cities")
plt.grid(True)

plt.savefig(
    "exact_tsp_distance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nBenchmark complete.")
print("benchmark_results.csv saved.")