import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_snapshot(filename):
    """
    Snapshot file format:
        # x y psi
        x y psi
    """
    data = np.loadtxt(filename, comments="#")
    x = data[:, 0]
    y = data[:, 1]
    psi = data[:, 2]

    xs = np.unique(x)
    ys = np.unique(y)

    nx = len(xs)
    ny = len(ys)

    X, Y = np.meshgrid(xs, ys)
    Z = np.full((ny, nx), np.nan)

    x_to_i = {val: i for i, val in enumerate(xs)}
    y_to_j = {val: j for j, val in enumerate(ys)}

    for xi, yi, zi in zip(x, y, psi):
        i = x_to_i[xi]
        j = y_to_j[yi]
        Z[j, i] = zi

    return X, Y, Z


def load_energy(filename):
    """
    Energy file format:
        # t kinetic potential total center_value max_abs_psi
    """
    return np.loadtxt(filename, comments="#")


def plot_snapshot_heatmap(snapshot_file, output_dir):
    X, Y, Z = load_snapshot(snapshot_file)

    base = os.path.splitext(os.path.basename(snapshot_file))[0]
    out = os.path.join(output_dir, base + "_heatmap.png")

    plt.figure(figsize=(7, 6))
    plt.pcolormesh(X, Y, Z, shading="auto")
    plt.colorbar(label=r"$\Psi(x,y,t)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(base)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    return out


def plot_snapshot_surface(snapshot_file, output_dir):
    X, Y, Z = load_snapshot(snapshot_file)

    base = os.path.splitext(os.path.basename(snapshot_file))[0]
    out = os.path.join(output_dir, base + "_surface.png")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$\Psi$")
    ax.set_title(base)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    return out


def plot_energy(energy_file, output_dir):
    data = load_energy(energy_file)

    t = data[:, 0]
    kinetic = data[:, 1]
    potential = data[:, 2]
    total = data[:, 3]
    center_value = data[:, 4]
    max_abs_psi = data[:, 5]

    base = os.path.splitext(os.path.basename(energy_file))[0]

    out1 = os.path.join(output_dir, base + "_energy.png")
    plt.figure(figsize=(8, 5))
    plt.plot(t, kinetic, label="kinetic")
    plt.plot(t, potential, label="potential")
    plt.plot(t, total, label="total")
    plt.xlabel("t")
    plt.ylabel("energy")
    plt.title(base)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    out2 = os.path.join(output_dir, base + "_amplitude.png")
    plt.figure(figsize=(8, 5))
    plt.plot(t, max_abs_psi, label=r"$\max |\Psi|$")
    plt.plot(t, np.abs(center_value), label=r"$|c_\mathrm{center}|$")
    plt.xlabel("t")
    plt.ylabel("amplitude")
    plt.title(base)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()

    return out1, out2


def plot_comparison_energy(energy_files, output_dir):
    plt.figure(figsize=(8, 5))

    for energy_file in energy_files:
        data = load_energy(energy_file)
        t = data[:, 0]
        max_abs_psi = data[:, 5]

        label = os.path.basename(energy_file).replace("_energy.txt", "")
        plt.plot(t, max_abs_psi, label=label)

    out = os.path.join(output_dir, "comparison_max_amplitude.png")
    plt.xlabel("t")
    plt.ylabel(r"$\max |\Psi|$")
    plt.title("Comparison of forced oscillations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    return out


def plot_last_snapshots_for_each_frequency(output_dir, base_path):
    """
    Finds the final snapshot for each omega_* prefix and plots heatmap + surface.
    """
    produced = []

    prefixes = [
        "omega_pi_over_L",
        "omega_2pi_over_L",
        "omega_pi_over_2L",
    ]

    for prefix in prefixes:
        files = sorted(glob.glob(os.path.join(base_path, prefix + "_snapshot_*.txt")))
        if not files:
            print(f"No snapshots found for {prefix}")
            continue

        last_file = files[-1]
        produced.append(plot_snapshot_heatmap(last_file, output_dir))
        produced.append(plot_snapshot_surface(last_file, output_dir))

    return produced


def plot_selected_snapshots(output_dir, max_per_frequency=5, base_path="/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug"):
    """
    Plots a few snapshots spread through time for each frequency.
    Useful for illustrating membrane vibrations.
    """
    produced = []

    prefixes = [
        "omega_pi_over_L",
        "omega_2pi_over_L",
        "omega_pi_over_2L",
    ]

    for prefix in prefixes:
        files = sorted(glob.glob(os.path.join(base_path, prefix + "_snapshot_*.txt")))
        if not files:
            continue

        if len(files) <= max_per_frequency:
            chosen = files
        else:
            ids = np.linspace(0, len(files) - 1, max_per_frequency, dtype=int)
            chosen = [files[i] for i in ids]

        for f in chosen:
            produced.append(plot_snapshot_heatmap(f, output_dir))

    return produced


def main():
    parser = argparse.ArgumentParser(
        description="Plot FEM membrane forced-oscillation results."
    )
    parser.add_argument(
        "--output",
        default="plots",
        help="Directory where plots will be saved."
    )
    parser.add_argument(
        "--all-surfaces",
        action="store_true",
        help="Generate 3D surface plots for all selected snapshots, not only final ones."
    )
    args = parser.parse_args()

    ensure_dir(args.output)

    produced = []

    # 1. Energy and amplitude plots for each frequency
    base_path = "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug"

    energy_files = sorted(
        glob.glob(os.path.join(base_path, "omega_*_energy.txt"))
    )
    
    if energy_files:
        for f in energy_files:
            produced.extend(plot_energy(f, args.output))

        produced.append(plot_comparison_energy(energy_files, args.output))
    else:
        print("No omega_*_energy.txt files found.")

    # 2. Final membrane shapes
    produced.extend(plot_last_snapshots_for_each_frequency(args.output, base_path))

    # 3. A few time snapshots for vibration illustration
    selected = []

    prefixes = [
        "omega_pi_over_L",
        "omega_2pi_over_L",
        "omega_pi_over_2L",
    ]

    for prefix in prefixes:
        files = sorted(glob.glob(os.path.join(base_path, prefix + "_snapshot_*.txt")))
        if not files:
            continue

        ids = np.linspace(0, len(files) - 1, min(5, len(files)), dtype=int)
        selected.extend([files[i] for i in ids])

    for f in selected:
        produced.append(plot_snapshot_heatmap(f, args.output))
        if args.all_surfaces:
            produced.append(plot_snapshot_surface(f, args.output))

    print("\nGenerated plots:")
    for p in produced:
        print("  " + p)


if __name__ == "__main__":
    main()
