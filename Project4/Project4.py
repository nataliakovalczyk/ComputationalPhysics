# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
#
#
# # Exact ground-state energy for omega_x = 1, omega_y = 2
# E_exact = 1.5
#
# # Alpha factors used in the C++ code
# alpha_factors = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.05]
#
# # Folder containing your .dat files
# data_folder = Path("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug")
#
# # Output folder for plots
# output_folder = Path("plots")
# output_folder.mkdir(exist_ok=True)
#
#
# def load_energy_data(alpha_factor):
#     """
#     Loads file of the form:
#         energy_alpha_0.900000.dat
#     with two columns:
#         iteration energy
#     """
#     filename = data_folder / f"energy_alpha_{alpha_factor:.6f}.dat"
#
#     if not filename.exists():
#         print(f"Missing file: {filename}")
#         return None
#
#     data = np.loadtxt(filename)
#
#     if data.ndim == 1:
#         data = data.reshape(1, -1)
#
#     iterations = data[:, 0]
#     energies = data[:, 1]
#
#     return iterations, energies
#
#
# # ------------------------------------------------------------
# # Plot 1: Energy expectation value vs iteration number
# # ------------------------------------------------------------
#
# plt.figure(figsize=(9, 6))
#
# for factor in alpha_factors:
#     result = load_energy_data(factor)
#
#     if result is None:
#         continue
#
#     iterations, energies = result
#
#     if np.any(~np.isfinite(energies)):
#         print(f"Non-finite values found for alpha factor {factor}")
#         continue
#
#     plt.plot(
#         iterations,
#         energies,
#         label=rf"$\alpha = {factor}\alpha_c$"
#     )
#
# plt.axhline(
#     E_exact,
#     linestyle="--",
#     linewidth=1.5,
#     label=rf"Exact $E_0 = {E_exact}$"
# )
#
# plt.xlabel("Iteration number")
# plt.ylabel(r"Energy expectation value $\langle E \rangle$")
# plt.title("Ground-state convergence for different values of alpha")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
#
# plt.savefig(output_folder / "ground_state_energy_vs_iteration.png", dpi=300)
# plt.show()
#
#
# # ------------------------------------------------------------
# # Plot 2: Zoomed plot near the converged value
# # ------------------------------------------------------------
#
# plt.figure(figsize=(9, 6))
#
# for factor in alpha_factors:
#     result = load_energy_data(factor)
#
#     if result is None:
#         continue
#
#     iterations, energies = result
#
#     if np.any(~np.isfinite(energies)):
#         continue
#
#     plt.plot(
#         iterations,
#         energies,
#         label=rf"$\alpha = {factor}\alpha_c$"
#     )
#
# plt.axhline(
#     E_exact,
#     linestyle="--",
#     linewidth=1.5,
#     label=rf"Exact $E_0 = {E_exact}$"
# )
#
# plt.xlabel("Iteration number")
# plt.ylabel(r"Energy expectation value $\langle E \rangle$")
# plt.title("Ground-state convergence, zoom near final energy")
# plt.ylim(1.45, 1.75)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
#
# plt.savefig(output_folder / "ground_state_energy_zoom.png", dpi=300)
# plt.show()
#
#
# # ------------------------------------------------------------
# # Plot 3: Error relative to exact energy
# # ------------------------------------------------------------
#
# plt.figure(figsize=(9, 6))
#
# for factor in alpha_factors:
#     result = load_energy_data(factor)
#
#     if result is None:
#         continue
#
#     iterations, energies = result
#
#     if np.any(~np.isfinite(energies)):
#         continue
#
#     error = np.abs(energies - E_exact)
#
#     plt.semilogy(
#         iterations,
#         error,
#         label=rf"$\alpha = {factor}\alpha_c$"
#     )
#
# plt.xlabel("Iteration number")
# plt.ylabel(r"$|\langle E\rangle - E_\mathrm{exact}|$")
# plt.title("Ground-state energy error")
# plt.grid(True, which="both")
# plt.legend()
# plt.tight_layout()
#
# plt.savefig(output_folder / "ground_state_error_vs_iteration.png", dpi=300)
# plt.show()
#
#
# # ------------------------------------------------------------
# # Table: convergence summary
# # ------------------------------------------------------------
#
# print()
# print("Convergence summary")
# print("-" * 70)
# print(f"{'alpha/alpha_c':>15} {'iterations':>15} {'final energy':>15} {'error':>15}")
# print("-" * 70)
#
# for factor in alpha_factors:
#     result = load_energy_data(factor)
#
#     if result is None:
#         continue
#
#     iterations, energies = result
#
#     finite_mask = np.isfinite(energies)
#
#     if not np.all(finite_mask):
#         print(f"{factor:15.3f} {'unstable':>15}")
#         continue
#
#     final_iteration = int(iterations[-1])
#     final_energy = energies[-1]
#     error = abs(final_energy - E_exact)
#
#     print(
#         f"{factor:15.3f} "
#         f"{final_iteration:15d} "
#         f"{final_energy:15.8f} "
#         f"{error:15.8e}"
#     )

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

number_of_states = 4

# Exact energies for the first few oscillator states:
# State 0: (nx, ny) = (0, 0), E = 1.5
# State 1: (nx, ny) = (1, 0), E = 2.5
# State 2 and 3: degenerate pair (2, 0) and (0, 1), E = 3.5
exact_energies = [1.5, 2.5, 3.5, 3.5]

data_folder = Path("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug")
output_folder = Path("plots_excited_states")
output_folder.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def load_energy_data(state_index):
    filename = data_folder / f"energy_state_{state_index}.dat"

    if not filename.exists():
        print(f"Missing file: {filename}")
        return None

    data = np.loadtxt(filename)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    iterations = data[:, 0]
    energies = data[:, 1]

    return iterations, energies


def load_wavefunction_data(state_index):
    filename = data_folder / f"psi_state_{state_index}.dat"

    if not filename.exists():
        print(f"Missing file: {filename}")
        return None

    data = np.loadtxt(filename)

    x = data[:, 0]
    y = data[:, 1]
    psi = data[:, 2]

    # Recover grid size from number of points
    number_of_points = len(psi)
    N = int(np.sqrt(number_of_points))

    if N * N != number_of_points:
        raise ValueError(
            f"Cannot reshape wavefunction for state {state_index}. "
            f"Number of data points is {number_of_points}, not a perfect square."
        )

    X = x.reshape(N, N)
    Y = y.reshape(N, N)
    PSI = psi.reshape(N, N)

    return X, Y, PSI


# ------------------------------------------------------------
# Plot 1: energy convergence for all states
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

for k in range(number_of_states):
    result = load_energy_data(k)

    if result is None:
        continue

    iterations, energies = result

    plt.plot(
        iterations,
        energies,
        label=rf"State {k + 1}, final $E={energies[-1]:.5f}$"
    )

for k, E_exact in enumerate(exact_energies):
    plt.axhline(
        E_exact,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6
    )

plt.xlabel("Iteration number")
plt.ylabel(r"Energy expectation value $\langle E \rangle$")
plt.ylim(1,20)
plt.title("Convergence of the four lowest-energy states, zoomed near final energy")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(output_folder / "excited_states_energy_convergence.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# Plot 2: separate convergence plot for each state
# ------------------------------------------------------------

for k in range(number_of_states):
    result = load_energy_data(k)

    if result is None:
        continue

    iterations, energies = result
    E_exact = exact_energies[k]

    plt.figure(figsize=(8, 5))

    plt.plot(
        iterations,
        energies,
        label=rf"Numerical $E={energies[-1]:.6f}$"
    )

    plt.axhline(
        E_exact,
        linestyle="--",
        linewidth=1.5,
        label=rf"Exact $E={E_exact}$"
    )

    plt.xlabel("Iteration number")
    plt.ylabel(r"Energy expectation value $\langle E \rangle$")
    plt.ylim(1, 20)
    plt.title(rf"Convergence of state {k + 1}, zoomed near final energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_folder / f"energy_convergence_state_{k + 1}.png", dpi=300)
    plt.show()


# ------------------------------------------------------------
# Plot 3: wavefunction visualizations Psi_K(x, y)
# ------------------------------------------------------------

for k in range(number_of_states):
    result = load_wavefunction_data(k)

    if result is None:
        continue

    X, Y, PSI = result

    plt.figure(figsize=(7, 6))

    contour = plt.contourf(
        X,
        Y,
        PSI,
        levels=80
    )

    plt.colorbar(contour, label=rf"$\Psi_{k + 1}(x,y)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(rf"Wavefunction $\Psi_{k + 1}(x,y)$")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(output_folder / f"wavefunction_state_{k + 1}.png", dpi=300)
    plt.show()


# ------------------------------------------------------------
# Table: final numerical energies
# ------------------------------------------------------------

print()
print("Final energies")
print("-" * 75)
print(f"{'State':>8} {'Expected dominant state':>25} {'Exact E':>12} {'Numerical E':>15} {'Error':>12}")
print("-" * 75)

dominant_states = [
    "(0, 0)",
    "(1, 0)",
    "(2, 0) or (0, 1)",
    "(0, 1) or (2, 0)"
]

for k in range(number_of_states):
    result = load_energy_data(k)

    if result is None:
        continue

    iterations, energies = result

    numerical_E = energies[-1]
    exact_E = exact_energies[k]
    error = abs(numerical_E - exact_E)

    print(
        f"{k + 1:8d} "
        f"{dominant_states[k]:>25} "
        f"{exact_E:12.6f} "
        f"{numerical_E:15.8f} "
        f"{error:12.4e}"
    )