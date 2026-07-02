import random
import matplotlib.pyplot as plt
import csv
from Benchmark.tsp_bm import main




def generate_csv(filename, n):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["city", "x", "y"])

        for i in range(n):
            writer.writerow([
                f"City{i+1}",
                random.uniform(0, 1),   # x coordinate
                random.uniform(0, 1)    # y coordinate
            ])


results = []

NUM_TRIALS = 10        # Number of random instances per city size

for n in range(10, 201, 10):

    runtimes = []

    for trial in range(NUM_TRIALS):

        filename = "temp_cities.csv"
        generate_csv(filename, n)

        runtime = main(filename)      # Pass the filename
        runtimes.append(runtime)

    avg_runtime = sum(runtimes) / NUM_TRIALS
    results.append((n, avg_runtime))

    print(f"{n} cities -> Average Runtime = {avg_runtime:.4f} sec")


# Plot scalability
cities = [x[0] for x in results]
times = [x[1] for x in results]

plt.figure(figsize=(8,5))
plt.plot(cities, times, marker='o')
plt.xlabel("Number of Cities")
plt.ylabel("Avg_Runtime (seconds)")
plt.title("Scalability of TSP Benchmark")
plt.grid(True)
plt.show()