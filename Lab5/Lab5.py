import numpy as np
import matplotlib.pyplot as plt
import glob


def plot_map(filename):
    # Load x, y, value columns
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Determine grid size N from unique x or y values
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    N = len(unique_x)

    # Reshape into N×N grid
    X = unique_x.reshape(N, 1).repeat(N, axis=1)
    Y = unique_y.reshape(1, N).repeat(N, axis=0)
    Z = z.reshape(N, N)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(X, Y, Z, cmap='seismic', shading='auto')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.title("V(x,y)")
    plt.tight_layout()
    plt.savefig("case_b_potential.png")
    plt.show()

def load_logfile(filename):
    data = np.loadtxt(filename)
    k = data[:, 0]
    S = data[:, 1]
    dS = data[:, 2]
    return k, S, dS


def plot_S_and_dS(logfiles):
    plt.figure(figsize=(8, 5))
    for file in logfiles:
        k, S, dS = load_logfile(file)
        label = file.replace("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/task3_", "").replace("00000_log.dat", "")
        plt.semilogx(k, S, label=label)

    plt.xlabel("Iteration k")
    plt.ylabel("Functional S")
    plt.title("S vs iteration")
    plt.grid(True)
    plt.legend()
    plt.savefig("S.png")

    plt.figure(figsize=(8, 5))
    for file in logfiles:
        k, S, dS = load_logfile(file)
        label = file.replace("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/task3_", "").replace("00000_log.dat", "")
        plt.loglog(k, dS, label=label)

    plt.xlabel("Iteration k (log)")
    plt.ylabel("Relative change dS (log)")
    plt.title("Relative change in S")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("Relative_change.png")

    plt.show()

if __name__ == "__main__":
    plot_map("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/.dat")

    # plot_map("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/case_a_error.dat")
    # logfiles = sorted(glob.glob("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/task3_omega_*_log.dat"))
    # plot_S_and_dS(logfiles)