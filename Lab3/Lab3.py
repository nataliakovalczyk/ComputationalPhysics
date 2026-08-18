import numpy as np
import matplotlib.pyplot as plt
import math
import os

YELLOW = (197/255, 138/255, 2/255)
DARK_GREY = (98/255, 112/255, 132/255)
BLUE = (79/255, 79/255, 238/255)
RED = (202/255, 43/255, 79/255)

def save_plot(savepath):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {savepath}")

def plot_orbit(filename, title, savepath=None):
    data = np.loadtxt(filename, delimiter=",")
    x = data[:, 0]
    y = data[:, 1]

    plt.figure()
    plt.plot(x, y, linewidth=1.5, color=DARK_GREY, label="Mercury Orbit")
    plt.scatter([0], [0], s=200, color=YELLOW, label="Sun", zorder=5)
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if savepath:
        save_plot(savepath)


def plot_precession(orbit_file, extrema_file, title, savepath=None):
    data = np.loadtxt(orbit_file, delimiter=",")
    x = data[:, 0]
    y = data[:, 1]

    ext = np.loadtxt(extrema_file, delimiter=",", dtype=str)
    types = ext[:, 0]
    xe = ext[:, 2].astype(float)
    ye = ext[:, 3].astype(float)

    peri_mask = types == "perihelion"
    aphe_mask = types == "aphelion"

    x_peri = xe[peri_mask]
    y_peri = ye[peri_mask]
    x_aphe = xe[aphe_mask]
    y_aphe = ye[aphe_mask]

    plt.figure()
    plt.plot(x, y, linewidth=1.5, color=DARK_GREY)

    if len(x_peri):
        plt.scatter(x_peri, y_peri, s=50, color=BLUE, marker='o', label="perihelion")
    if len(x_aphe):
        plt.scatter(x_aphe, y_aphe, s=50, color=RED, marker='o', label="aphelion")

    for xp, yp in zip(x_peri, y_peri):
        plt.plot([0, xp], [0, yp], color=BLUE, linewidth=1.0, zorder=3)
    for xa, ya in zip(x_aphe, y_aphe):
        plt.plot([0, xa], [0, ya], color=RED, linewidth=1.0, zorder=3)

    plt.scatter([0], [0], s=250, color=YELLOW, edgecolor="black", label="Sun")

    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if savepath:
        save_plot(savepath)


def plot_alpha_omega(filename, title, savepath=None):
    data = np.loadtxt(filename, delimiter=",", comments="#")
    alpha = data[:, 0]
    omega = data[:, 1]

    a = np.sum(alpha * omega) / np.sum(alpha**2)
    alpha_fit = np.linspace(0, 1.05 * np.max(alpha), 200)
    omega_fit = a * alpha_fit

    alpha_real = 1.1e-8
    omega_real = a * alpha_real
    arcsec_per_century = omega_real * (180 / math.pi) * 3600 * 100

    plt.figure()
    plt.scatter(alpha, omega, color=BLUE, s=50, label="Data (αj, ωj)")
    plt.plot(alpha_fit, omega_fit, color=RED, linewidth=2.0,
             label=f"Fit ω = a·α\n a = {a:.4f} rad/yr")
    plt.xlabel("α")
    plt.ylabel("ω [rad/year]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.text(
        0.5 * np.max(alpha),
        0.8 * np.max(omega),
        f"ω(α=1.1e−8) = {arcsec_per_century:.2f}″/century",
        fontsize=9,
        color=DARK_GREY,
    )

    if savepath:
        save_plot(savepath)

base = "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug"

plot_orbit(f"{base}/test_no_relativistic.csv",
           "Mercury Orbit — No Relativistic Effects (0.95 T)",
           f"{base}/plot_no_relativistic.png")

plot_orbit(f"{base}/test_of_stability.csv",
           "Mercury Orbit — Stability Test (100 T)",
           f"{base}/plot_stability.png")

plot_orbit(f"{base}/precession_alpha001.csv",
           "Mercury Orbit - Precession Alpha001",
           f"{base}/plot_precession_alpha001.png")

plot_precession(
    f"{base}/precession_alpha001.csv",
    f"{base}/extrema_precession.csv",
    "Mercury Precession (α = 0.01)",
    f"{base}/plot_precession_extrema.png"
)

plot_alpha_omega(
    f"{base}/alpha_omega.csv",
    "Linear Fit of ω = a·α",
    f"{base}/plot_alpha_omega_fit.png"
)

plt.show()
