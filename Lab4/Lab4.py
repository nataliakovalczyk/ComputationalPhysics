import numpy as np
import matplotlib.pyplot as plt
import os


def plot_solution(numeric_path, exact_path=None, title=None, save_dir="."):
    num = np.loadtxt(numeric_path, delimiter=",")
    t = num[:, 0]
    N0, N1, N2 = num[:, 1], num[:, 2], num[:, 3]

    has_exact = exact_path is not None
    if has_exact:
        ex = np.loadtxt(exact_path, delimiter=",")
        te = ex[:, 0]
        N0e, N1e, N2e = ex[:, 1], ex[:, 2], ex[:, 3]

    plt.figure(figsize=(8, 6))
    plt.plot(t, N0, label="N0 (num)")
    plt.plot(t, N1, label="N1 (num)")
    plt.plot(t, N2, label="N2 (num)")
    if has_exact:
        plt.plot(te, N0e, "--", label="N0 (exact)")
        plt.plot(te, N1e, "--", label="N1 (exact)")
        plt.plot(te, N2e, "--", label="N2 (exact)")
    plt.xscale("log")
    plt.xlabel("t")
    plt.ylabel("N")
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)

    base_name = os.path.splitext(os.path.basename(numeric_path))[0]
    safe_title = title.replace(" ", "_").replace(",", "").replace("=", "") if title else base_name
    solution_filename = os.path.join(save_dir, f"{safe_title}_solution.png")
    plt.tight_layout()
    plt.savefig(solution_filename, dpi=300)
    print(f"Saved plot: {solution_filename}")

    dt = np.diff(t)
    t_mid = t[:-1] + 0.5 * dt

    plt.figure(figsize=(6, 4))
    plt.loglog(t_mid, dt)
    plt.xlabel("t")
    plt.ylabel("Δt (reconstructed)")
    plt.grid(True)
    dt_filename = os.path.join(save_dir, f"{safe_title}_dt.png")
    plt.tight_layout()
    plt.savefig(dt_filename, dpi=300)
    print(f"Saved Δt plot: {dt_filename}")

    plt.close("all")


plot_solution(
    "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/test_correctness.csv",
    "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/exact_correctness.csv",
    title="λ0=1, λ1=5, λ2=50, TOL=1e-4",
)

plot_solution(
    "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/tolerance_-6.csv",
    "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/exact_tolerance.csv",
    title="λ0=100, λ1=1, λ2=0.01, TOL=1e-6",
)

plot_solution(
    "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/tolerance_-3.csv",
    "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/exact_tolerance.csv",
    title="λ0=100, λ1=1, λ2=0.01, TOL=1e-3",
)
