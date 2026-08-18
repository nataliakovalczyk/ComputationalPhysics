import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

folder = "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/"

files = {
    "original": folder + "time_observables_original.csv",
    "reverse": folder + "time_observables_reverse.csv",
    "x_oscillation": folder + "time_observables_x_oscillation.csv",
    "y_oscillation": folder + "time_observables_y_oscillation.csv",
    "eigenstate_0": folder + "time_observables_eigenstate_0.csv",
    "eigenstate_1": folder + "time_observables_eigenstate_1.csv",
    "eigenstate_2": folder + "time_observables_eigenstate_2.csv",
}

data = {name: pd.read_csv(path) for name, path in files.items()}

pink = "#ff69b4"
purple = "#8a2be2"
orange = "#ff8c00"

T = 2 * np.pi


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(folder + filename, dpi=300)
    plt.show()


# -------------------------------------------------
# 1. Norm and energy for original initial condition
# -------------------------------------------------
original = data["original"]

plt.figure(figsize=(7, 5))
plt.plot(original["t"], original["norm"], color=pink)
plt.xlabel("t")
plt.ylabel("Norm")
plt.title("Norm as a function of time")
plt.grid(True)
save_plot("plot_norm_original.png")

plt.figure(figsize=(7, 5))
plt.plot(original["t"], original["energy"], color=purple)
plt.xlabel("t")
plt.ylabel(r"$\langle E \rangle$")
plt.title("Average energy as a function of time")
plt.grid(True)
save_plot("plot_energy_original.png")


# -------------------------------------------------
# 2. <x>(t) and <y>(t) for original packet
# -------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(original["t"], original["x_avg"], color=pink, label=r"$\langle x \rangle(t)$")
plt.plot(original["t"], original["y_avg"], color=purple, label=r"$\langle y \rangle(t)$")
plt.axvline(T, color=orange, linestyle="--", label=r"$2\pi$")
plt.axvline(2 * T, color=orange, linestyle=":", label=r"$4\pi$")
plt.xlabel("t")
plt.ylabel("Average position")
plt.title(r"$\langle x \rangle(t)$ and $\langle y \rangle(t)$")
plt.legend()
plt.grid(True)
save_plot("plot_x_y_original.png")


# -------------------------------------------------
# 3. Trajectory <y>(<x>) for original packet
# -------------------------------------------------
plt.figure(figsize=(6, 6))
plt.plot(original["x_avg"], original["y_avg"], color=purple)
plt.scatter(original["x_avg"].iloc[0], original["y_avg"].iloc[0],
            color=pink, label="start")
plt.scatter(original["x_avg"].iloc[-1], original["y_avg"].iloc[-1],
            color=orange, label="end")
plt.xlabel(r"$\langle x \rangle$")
plt.ylabel(r"$\langle y \rangle$")
plt.title(r"Trajectory: $\langle y \rangle(\langle x \rangle)$")
plt.axis("equal")
plt.legend()
plt.grid(True)
save_plot("plot_trajectory_original.png")


# -------------------------------------------------
# 4. Original vs reverse rotation
# -------------------------------------------------
reverse = data["reverse"]

plt.figure(figsize=(6, 6))
plt.plot(original["x_avg"], original["y_avg"], color=pink, label="original")
plt.plot(reverse["x_avg"], reverse["y_avg"], color=purple, label="reverse")
plt.scatter(original["x_avg"].iloc[0], original["y_avg"].iloc[0],
            color=orange, label="start")
plt.xlabel(r"$\langle x \rangle$")
plt.ylabel(r"$\langle y \rangle$")
plt.title("Original and opposite direction of rotation")
plt.axis("equal")
plt.legend()
plt.grid(True)
save_plot("plot_original_vs_reverse_rotation.png")


# -------------------------------------------------
# 5. Pure x oscillation
# -------------------------------------------------
xosc = data["x_oscillation"]

plt.figure(figsize=(7, 5))
plt.plot(xosc["t"], xosc["x_avg"], color=pink, label=r"$\langle x \rangle(t)$")
plt.plot(xosc["t"], xosc["y_avg"], color=purple, label=r"$\langle y \rangle(t)$")
plt.xlabel("t")
plt.ylabel("Average position")
plt.title("Oscillation in x direction without rotation")
plt.legend()
plt.grid(True)
save_plot("plot_x_oscillation_time.png")

plt.figure(figsize=(6, 6))
plt.plot(xosc["x_avg"], xosc["y_avg"], color=orange)
plt.xlabel(r"$\langle x \rangle$")
plt.ylabel(r"$\langle y \rangle$")
plt.title("Trajectory for pure x oscillation")
plt.axis("equal")
plt.grid(True)
save_plot("plot_x_oscillation_trajectory.png")


# -------------------------------------------------
# 6. Pure y oscillation
# -------------------------------------------------
yosc = data["y_oscillation"]

plt.figure(figsize=(7, 5))
plt.plot(yosc["t"], yosc["x_avg"], color=pink, label=r"$\langle x \rangle(t)$")
plt.plot(yosc["t"], yosc["y_avg"], color=purple, label=r"$\langle y \rangle(t)$")
plt.xlabel("t")
plt.ylabel("Average position")
plt.title("Oscillation in y direction without rotation")
plt.legend()
plt.grid(True)
save_plot("plot_y_oscillation_time.png")

plt.figure(figsize=(6, 6))
plt.plot(yosc["x_avg"], yosc["y_avg"], color=orange)
plt.xlabel(r"$\langle x \rangle$")
plt.ylabel(r"$\langle y \rangle$")
plt.title("Trajectory for pure y oscillation")
plt.axis("equal")
plt.grid(True)
save_plot("plot_y_oscillation_trajectory.png")


# -------------------------------------------------
# 7. Eigenstate initial conditions
# -------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(data["eigenstate_0"]["t"], data["eigenstate_0"]["energy"],
         color=pink, label=r"$\Psi_0$")
plt.plot(data["eigenstate_1"]["t"], data["eigenstate_1"]["energy"],
         color=purple, label=r"$\Psi_1$")
plt.plot(data["eigenstate_2"]["t"], data["eigenstate_2"]["energy"],
         color=orange, label=r"$\Psi_2$")
plt.xlabel("t")
plt.ylabel(r"$\langle E \rangle$")
plt.title("Energy for eigenstate initial conditions")
plt.legend()
plt.grid(True)
save_plot("plot_eigenstate_energy.png")

plt.figure(figsize=(7, 5))
plt.plot(data["eigenstate_0"]["t"], data["eigenstate_0"]["norm"],
         color=pink, label=r"$\Psi_0$")
plt.plot(data["eigenstate_1"]["t"], data["eigenstate_1"]["norm"],
         color=purple, label=r"$\Psi_1$")
plt.plot(data["eigenstate_2"]["t"], data["eigenstate_2"]["norm"],
         color=orange, label=r"$\Psi_2$")
plt.xlabel("t")
plt.ylabel("Norm")
plt.title("Norm for eigenstate initial conditions")
plt.legend()
plt.grid(True)
save_plot("plot_eigenstate_norm.png")

plt.figure(figsize=(7, 5))
plt.plot(data["eigenstate_0"]["t"], data["eigenstate_0"]["x_avg"],
         color=pink, label=r"$\langle x \rangle$, $\Psi_0$")
plt.plot(data["eigenstate_1"]["t"], data["eigenstate_1"]["x_avg"],
         color=purple, label=r"$\langle x \rangle$, $\Psi_1$")
plt.plot(data["eigenstate_2"]["t"], data["eigenstate_2"]["x_avg"],
         color=orange, label=r"$\langle x \rangle$, $\Psi_2$")
plt.xlabel("t")
plt.ylabel(r"$\langle x \rangle$")
plt.title(r"$\langle x \rangle(t)$ for eigenstate initial conditions")
plt.legend()
plt.grid(True)
save_plot("plot_eigenstate_x_avg.png")

plt.figure(figsize=(7, 5))
plt.plot(data["eigenstate_0"]["t"], data["eigenstate_0"]["y_avg"],
         color=pink, label=r"$\langle y \rangle$, $\Psi_0$")
plt.plot(data["eigenstate_1"]["t"], data["eigenstate_1"]["y_avg"],
         color=purple, label=r"$\langle y \rangle$, $\Psi_1$")
plt.plot(data["eigenstate_2"]["t"], data["eigenstate_2"]["y_avg"],
         color=orange, label=r"$\langle y \rangle$, $\Psi_2$")
plt.xlabel("t")
plt.ylabel(r"$\langle y \rangle$")
plt.title(r"$\langle y \rangle(t)$ for eigenstate initial conditions")
plt.legend()
plt.grid(True)
save_plot("plot_eigenstate_y_avg.png")


# -------------------------------------------------
# 8. Numerical periodicity check for original packet
# -------------------------------------------------
def closest_row(df, target_t):
    idx = (df["t"] - target_t).abs().idxmin()
    return df.loc[idx]


row_0 = closest_row(original, 0.0)
row_T = closest_row(original, T)
row_2T = closest_row(original, 2 * T)

print("Values near t = 0:")
print(row_0[["t", "norm", "energy", "x_avg", "y_avg"]])

print("\nValues near t = 2π:")
print(row_T[["t", "norm", "energy", "x_avg", "y_avg"]])

print("\nValues near t = 4π:")
print(row_2T[["t", "norm", "energy", "x_avg", "y_avg"]])

print("\nDifference between t = 0 and t = 2π:")
print(row_T[["norm", "energy", "x_avg", "y_avg"]] -
      row_0[["norm", "energy", "x_avg", "y_avg"]])

print("\nDifference between t = 0 and t = 4π:")
print(row_2T[["norm", "energy", "x_avg", "y_avg"]] -
      row_0[["norm", "energy", "x_avg", "y_avg"]])