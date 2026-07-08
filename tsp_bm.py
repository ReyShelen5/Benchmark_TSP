import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pulp


# ============================================================
# Benchmark settings
# ============================================================

                   # coordinate range: 0 to 100

OUTPUT_DIR = Path("exact_tsp_benchmark_csv")
INSTANCE_DIR = OUTPUT_DIR / "instances"
ROUTE_DIR = OUTPUT_DIR / "exact_routes"

SUMMARY_FILE = OUTPUT_DIR / "all_exact_tsp_results.csv"

OUTPUT_DIR.mkdir(exist_ok=True)
INSTANCE_DIR.mkdir(exist_ok=True)
ROUTE_DIR.mkdir(exist_ok=True)


# ============================================================
# Read Coordinates froma file
# ============================================================

def load_coordinates(csv_file):
    df = pd.read_csv(csv_file)

    # Assumes columns: x, y
    coords = df[['x', 'y']].to_numpy(dtype=float)

    return coords


# ============================================================
# Distance matrix
# ============================================================

EARTH_RADIUS = 6371.0

def haversine(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat/2)**2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon/2)**2
    )

    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return EARTH_RADIUS*c


def compute_distance_matrix(coords):

    n = len(coords)

    dist = np.zeros((n,n))

    for i in range(n):
        for j in range(n):

            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]

            dist[i,j] = haversine(
                lat1, lon1,
                lat2, lon2
            )

    return dist


# ============================================================
# Find subtours / connected components
# ============================================================

def find_components(n, selected_edges):
    adjacency = {i: [] for i in range(n)}

    for i, j in selected_edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    visited = set()
    components = []

    for start in range(n):
        if start in visited:
            continue

        stack = [start]
        component = []

        while stack:
            node = stack.pop()

            if node in visited:
                continue

            visited.add(node)
            component.append(node)

            for neigh in adjacency[node]:
                if neigh not in visited:
                    stack.append(neigh)

        components.append(component)

    return components


# ============================================================
# Reconstruct optimal tour
# ============================================================

def reconstruct_tour(n, selected_edges):
    if n == 1:
        return [0]

    adjacency = {i: [] for i in range(n)}

    for i, j in selected_edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    tour = [0]
    previous = None
    current = 0

    while True:
        neighbors = adjacency[current]

        if previous is None:
            next_city = neighbors[0]
        else:
            candidates = [node for node in neighbors if node != previous]

            if not candidates:
                break

            next_city = candidates[0]

        if next_city == 0:
            break

        tour.append(next_city)

        previous = current
        current = next_city

    return tour


# ============================================================
# Exact TSP solver using MILP + DFJ subtour elimination
# ============================================================

def solve_exact_tsp(coords):
    n = len(coords)
    dist = compute_distance_matrix(coords)

    # Special case: 2 cities
    if n == 2:
        return {
            "status": "Optimal",
            "objective": float(2.0 * dist[0, 1]),
            "tour": [0, 1],
            "selected_edges": [(0, 1)],
            "iterations": 0,
        }

    # Create MILP model
    model = pulp.LpProblem("Exact_TSP_DFJ", pulp.LpMinimize)

    # Binary edge variable x_ij = 1 if edge between city i and j is selected
    x = {}

    for i in range(n):
        for j in range(i + 1, n):
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")

    # Objective: minimize total tour distance
    model += pulp.lpSum(dist[i, j] * x[i, j] for i, j in x)

    # Degree constraint: every city must have exactly 2 incident selected edges
    for k in range(n):
        incident_edges = []

        for i, j in x:
            if i == k or j == k:
                incident_edges.append(x[i, j])

        model += pulp.lpSum(incident_edges) == 2

    solver = pulp.PULP_CBC_CMD(msg=False)

    iteration = 0

    while True:
        iteration += 1

        model.solve(solver)
        status = pulp.LpStatus[model.status]

        if status != "Optimal":
            return {
                "status": status,
                "objective": None,
                "tour": None,
                "selected_edges": None,
                "iterations": iteration,
            }

        selected_edges = []

        for i, j in x:
            if pulp.value(x[i, j]) > 0.5:
                selected_edges.append((i, j))

        components = find_components(n, selected_edges)

        # If only one component exists, then we have a valid TSP tour
        if len(components) == 1:
            tour = reconstruct_tour(n, selected_edges)

            return {
                "status": "Optimal",
                "objective": float(pulp.value(model.objective)),
                "tour": tour,
                "selected_edges": selected_edges,
                "iterations": iteration,
            }

        # Add subtour elimination constraints
        for component in components:
            if len(component) < n:
                subtour_edges = []

                for a in range(len(component)):
                    for b in range(a + 1, len(component)):
                        i = component[a]
                        j = component[b]

                        if i < j:
                            subtour_edges.append(x[i, j])
                        else:
                            subtour_edges.append(x[j, i])

                model += pulp.lpSum(subtour_edges) <= len(component) - 1


# ============================================================
# Save coordinate CSV
# ============================================================

def save_instance_csv(coords, n):
    df = pd.DataFrame({
        "city_id": np.arange(len(coords)),
        "x": coords[:, 0],
        "y": coords[:, 1],
    })

    filename = INSTANCE_DIR / f"cities_N{n}.csv"
    df.to_csv(filename, index=False)

    return filename


# ============================================================
# Save route CSV
# ============================================================

def save_route_csv(coords, tour, dist, n):
    closed_tour = tour + [tour[0]]

    route_rows = []

    for step in range(len(closed_tour) - 1):
        from_city = closed_tour[step]
        to_city = closed_tour[step + 1]

        route_rows.append({
            "step": step + 1,
            "from_city": from_city,
            "to_city": to_city,
            "from_x": coords[from_city, 0],
            "from_y": coords[from_city, 1],
            "to_x": coords[to_city, 0],
            "to_y": coords[to_city, 1],
            "edge_distance": dist[from_city, to_city],
        })

    route_df = pd.DataFrame(route_rows)

    filename = ROUTE_DIR / f"exact_route_N{n}.csv"
    route_df.to_csv(filename, index=False)

    return filename


# ============================================================
# Run benchmark
# ============================================================

def main(csv_file):
    summary_rows = []

    coords = load_coordinates(csv_file)
    n = len(coords)

    dist = compute_distance_matrix(coords)

    instance_file = csv_file

    start_time = time.time()
    solution = solve_exact_tsp(coords)
    runtime = time.time() - start_time

    if solution["status"] == "Optimal":
                route_file = save_route_csv(
                    coords=coords,
                    tour=solution["tour"],
                    dist=dist,
                    n=n
                )

                closed_tour = solution["tour"] + [solution["tour"][0]]

                print(f"Status        : {solution['status']}")
                print(f"Optimal length: {solution['objective']:.6f}")
                print(f"Tour          : {closed_tour}")
                print(f"Runtime       : {runtime:.6f} sec")

                summary_rows.append({
                                "N": n,
                                "status": solution["status"],
                                "optimal_tour_length": solution["objective"],
                                "optimal_tour_order": json.dumps(closed_tour),
                                "runtime_sec": runtime,
                                "subtour_iterations": solution["iterations"],
                                "instance_csv": str(instance_file),
                                "route_csv": str(route_file),
                            })

    else:
        print(f"Status: {solution['status']}")
        print("No optimal solution found.")

        summary_rows.append({
                    "N": n,
                    "status": solution["status"],
                    "optimal_tour_length": None,
                    "optimal_tour_order": None,
                    "runtime_sec": runtime,
                    "subtour_iterations": solution["iterations"],
                    "instance_csv": str(instance_file),
                    "route_csv": None,
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\n======================================")
    print("Exact TSP benchmark completed.")
    print(f"Summary saved at: {SUMMARY_FILE}")
    print(f"City coordinate CSVs saved in: {INSTANCE_DIR}")
    print(f"Exact route CSVs saved in: {ROUTE_DIR}")
    print("======================================")
    if solution["status"] == "Optimal":
        return runtime, solution["objective"]
    else:
        return runtime, None

if __name__ == "__main__":
    main("cities.csv")

##########  you will need install pulp library pip install pulp