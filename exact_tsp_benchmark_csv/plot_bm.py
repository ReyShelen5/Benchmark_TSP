import pandas as pd
import matplotlib.pyplot as plt


def show_benchmark(route_csv):

    # -------------------------------------------------
    # Read optimal route CSV
    # -------------------------------------------------
    df = pd.read_csv(route_csv)

    # -------------------------------------------------
    # Collect unique cities and coordinates
    # -------------------------------------------------
    cities = {}

    for _, row in df.iterrows():

        cities[int(row["from_city"])] = (
            row["from_x"],
            row["from_y"]
        )

        cities[int(row["to_city"])] = (
            row["to_x"],
            row["to_y"]
        )

    total_distance = df["edge_distance"].sum()

    # -------------------------------------------------
    # Print Information
    # -------------------------------------------------

    print("\n===================================")
    print("OPTIMAL ROUTE")
    print("===================================")

    print(f"\nTotal Distance : {total_distance:.4f} KM\n")

    print("Input Cities")
    print("-----------------------------------")

    for city in sorted(cities):

        x, y = cities[city]

        print(f"City {city:2d} : ({x:.4f}, {y:.4f})")

    print("\nOptimal Tour")
    print("-----------------------------------")

    for _, row in df.iterrows():

        print(
            f"Step {int(row['step']):2d} -> "
            f"City {int(row['from_city']):2d}"
            f" -> City {int(row['to_city']):2d}"
            f"   Distance = {row['edge_distance']:.4f} KM"
        )

    print("\nTour Sequence")
    print("-----------------------------------")

    sequence = [int(df.iloc[0]["from_city"])]

    for _, row in df.iterrows():
        sequence.append(int(row["to_city"]))

    print(" -> ".join(map(str, sequence)))

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    plt.figure(figsize=(9, 9))

    # Plot cities
    for city in sorted(cities):

        x, y = cities[city]

        plt.plot(
            x,
            y,
            'yo',
            markersize=10
        )

        plt.text(
            x,
            y,
            f"{city}\n({x:.2f}, {y:.2f})",
            fontsize=8,
            color="blue",
            ha="left",
            va="bottom"
        )

    # Plot edges
    for _, row in df.iterrows():

        x1 = row["from_x"]
        y1 = row["from_y"]

        x2 = row["to_x"]
        y2 = row["to_y"]

        plt.plot(
            [x1, x2],
            [y1, y2],
            'k-',
            linewidth=1.5
        )

        # Arrow
        plt.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color="red",
                lw=1.5
            )
        )

        # Edge distance
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        plt.text(
            mx,
            my,
            f"{row['edge_distance']:.1f}",
            fontsize=7,
            color="darkgreen",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8
            )
        )

    # Highlight start city
    start = df.iloc[0]

    plt.scatter(
        start["from_x"],
        start["from_y"],
        s=250,
        c="lime",
        edgecolors="black",
        zorder=5,
        label="Start"
    )

    # Highlight end city
    end = df.iloc[-1]

    plt.scatter(
        end["to_x"],
        end["to_y"],
        s=250,
        c="orange",
        edgecolors="black",
        zorder=5,
        label="End"
    )

    plt.title(
        f"Optimal Benchmark Tour\nTotal Distance = {total_distance:.4f} KM",
        fontsize=14
    )

    plt.xlabel("Latitude")
    plt.ylabel("Longitude")

    plt.grid(True)

    plt.axis("equal")

    plt.legend()

    plt.tight_layout()

    plt.show()
show_benchmark("exact_routes\exact_route_N30.csv")