import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug"


def load_csv(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def plot_energy_vs_time():
    df = load_csv("energy_vs_time_omega_pi.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(df["t"], df["energy"])
    plt.xlabel("t")
    plt.ylabel("E(t)")
    plt.title(r"Energy vs time for $\omega=\pi$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plot_energy_vs_time_omega_pi.png"), dpi=300)
    plt.show()


def plot_center_motion():
    df = load_csv("energy_vs_time_omega_pi.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(df["t"], df["center"])
    plt.xlabel("t")
    plt.ylabel(r"$u(c,c,t)$")
    plt.title(r"Central node motion for $\omega=\pi$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plot_center_motion_omega_pi.png"), dpi=300)
    plt.show()


def plot_average_energy_vs_omega():
    df = load_csv("avg_energy_vs_omega.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(df["omega"], df["average_energy"])
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\langle E\rangle$")
    plt.title(r"Average energy vs driving frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plot_avg_energy_vs_omega.png"), dpi=300)
    plt.show()


def plot_damping_comparison():
    df = load_csv("avg_energy_vs_omega_damping.csv")

    plt.figure(figsize=(8, 5))

    for two_d, group in df.groupby("twoD"):
        plt.plot(
            group["omega"],
            group["average_energy"],
            label=rf"$2d={two_d}$"
        )

    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\langle E\rangle$")
    plt.title("Influence of damping on resonant peaks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "plot_damping_comparison.png"), dpi=300)
    plt.show()


def print_possible_resonances():
    df = load_csv("avg_energy_vs_omega.csv")

    # Simple local-maximum detector
    peaks = []

    for i in range(1, len(df) - 1):
        e_prev = df.loc[i - 1, "average_energy"]
        e_now = df.loc[i, "average_energy"]
        e_next = df.loc[i + 1, "average_energy"]

        if e_now > e_prev and e_now > e_next:
            peaks.append((df.loc[i, "omega"], e_now))

    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)

    print("\nStrongest numerical resonant peaks:")
    for omega, energy in peaks[:10]:
        print(f"omega = {omega:.6f},  <E> = {energy:.6e}")


if __name__ == "__main__":
    plot_energy_vs_time()
    plot_center_motion()
    plot_average_energy_vs_omega()
    plot_damping_comparison()
    print_possible_resonances()