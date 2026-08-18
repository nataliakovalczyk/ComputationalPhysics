import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug"


def load_csv(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def save_plot(filename):
    path = os.path.join(BASE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"Saved: {path}")


def plot_pure_advection_center():
    df = load_csv("pure_advection_stats.csv")

    plt.figure(figsize=(7, 6))
    plt.plot(df["center_x"], df["center_y"])
    plt.scatter(df["center_x"].iloc[0], df["center_y"].iloc[0], label="start")
    plt.xlabel(r"$x_c$")
    plt.ylabel(r"$y_c$")
    plt.title("Pure advection: packet center trajectory")
    plt.legend()
    plt.grid(True)
    save_plot("plot_pure_advection_center.png")


def plot_pure_advection_shape_error():
    df = load_csv("pure_advection_stats.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(df["t"], df["shape_error"])
    plt.xlabel(r"$t$")
    plt.ylabel("relative shape error")
    plt.title("Pure advection: shape preservation")
    plt.grid(True)
    save_plot("plot_pure_advection_shape_error.png")


def plot_pure_diffusion_min_max():
    df = load_csv("pure_diffusion_stats.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(df["t"], df["min_u"], label=r"$\min(u)$")
    plt.plot(df["t"], df["max_u"], label=r"$\max(u)$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u$")
    plt.title("Pure diffusion: minimum and maximum values")
    plt.legend()
    plt.grid(True)
    save_plot("plot_pure_diffusion_min_max.png")


def plot_advection_diffusion_center():
    df = load_csv("advection_diffusion_stats.csv")

    plt.figure(figsize=(7, 6))
    plt.plot(df["center_x"], df["center_y"], label="numerical center")
    plt.scatter(df["center_x"].iloc[0], df["center_y"].iloc[0], label="start")

    plt.xlabel(r"$x_c$")
    plt.ylabel(r"$y_c$")
    plt.title("Advection-diffusion: packet center trajectory")
    plt.legend()
    plt.grid(True)
    save_plot("plot_advection_diffusion_center.png")


def plot_advection_diffusion_velocity():
    df = load_csv("advection_diffusion_stats.csv")

    vx_num = (df["center_x"].iloc[-1] - df["center_x"].iloc[0]) / (
        df["t"].iloc[-1] - df["t"].iloc[0]
    )
    vy_num = (df["center_y"].iloc[-1] - df["center_y"].iloc[0]) / (
        df["t"].iloc[-1] - df["t"].iloc[0]
    )

    print("\nAdvection-diffusion numerical velocity estimate:")
    print(f"vx ≈ {vx_num:.6f}")
    print(f"vy ≈ {vy_num:.6f}")
    print("Expected: vx = 1, vy = 1")

    plt.figure(figsize=(8, 5))
    plt.plot(df["t"], df["center_x"], label=r"$x_c(t)$")
    plt.plot(df["t"], df["center_y"], label=r"$y_c(t)$")
    plt.plot(df["t"], df["center_x"].iloc[0] + df["t"], "--", label=r"expected $x_c(0)+t$")
    plt.plot(df["t"], df["center_y"].iloc[0] + df["t"], "--", label=r"expected $y_c(0)+t$")
    plt.xlabel(r"$t$")
    plt.ylabel("center coordinate")
    plt.title("Advection-diffusion: center motion")
    plt.legend()
    plt.grid(True)
    save_plot("plot_advection_diffusion_center_motion.png")


def plot_mass_conservation():
    files = {
        "pure advection": "pure_advection_stats.csv",
        "pure diffusion": "pure_diffusion_stats.csv",
        "advection-diffusion": "advection_diffusion_stats.csv",
    }

    plt.figure(figsize=(8, 5))

    for label, filename in files.items():
        df = load_csv(filename)
        plt.plot(df["t"], df["mass"], label=label)

    plt.xlabel(r"$t$")
    plt.ylabel("mass")
    plt.title("Mass conservation")
    plt.legend()
    plt.grid(True)
    save_plot("plot_mass_conservation.png")


if __name__ == "__main__":
    plot_pure_advection_center()
    plot_pure_advection_shape_error()
    plot_pure_diffusion_min_max()
    plot_advection_diffusion_center()
    plot_advection_diffusion_velocity()
    plot_mass_conservation()