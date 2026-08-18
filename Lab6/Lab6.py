import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------
# Path to the directory that contains p1/, p2/, ..., p6/
# -----------------------------------------------------
BASE_DIR = "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug"


# -----------------------------------------------------
def load_steps(path):
    arr = np.loadtxt(path)
    n = arr[:,0]
    t = arr[:,1]
    T_sensor = arr[:,2]
    E_sup = arr[:,3]
    E_win = arr[:,4]
    return n, t, T_sensor, E_sup, E_win


def load_field(path):
    return np.loadtxt(path)


def plot_T_sensor(time, T, outname):
    plt.figure(figsize=(7,5))
    plt.plot(time, T, 'k-', linewidth=2)
    plt.xlabel("time")
    plt.ylabel("T(ic,jc)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()


def plot_heatmap(field, outname):
    plt.figure(figsize=(7,6))

    N = field.shape[0]
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    # colored heatmap
    cs = plt.contourf(Y, X, field,
                      levels=20,
                      cmap='jet')

    plt.colorbar(cs)

    # --- NEW: labels ---
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.savefig(outname + ".png", dpi=180)
    plt.close()



def plot_energies(time, E_sup, E_win, outname):
    plt.figure(figsize=(7,5))
    plt.plot(time, E_sup, 'r-', label="E_supplied", linewidth=2)
    plt.plot(time, E_win, 'b-', label="E_window", linewidth=2)
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()


def process_case(case_name, heat_times):
    dirname = os.path.join(BASE_DIR, case_name)
    print(f"Processing: {dirname}")

    steps_path = os.path.join(dirname, "steps.dat")
    if not os.path.exists(steps_path):
        print(f"steps.dat not found in {dirname}, skipping.")
        return

    n, t, T_sensor, E_sup, E_win = load_steps(steps_path)

    # T_sensor vs time
    plot_T_sensor(t, T_sensor, f"{case_name}_Tsensor.png")

    # Heatmaps at the requested times
    for target_time in heat_times:
        n_index = int(round(target_time / 10.0))  # dt = 10
        field_path = os.path.join(dirname, f"T_{n_index:06d}.dat")
        if os.path.exists(field_path):
            field = load_field(field_path)
            plot_heatmap(field, f"{case_name}_T_{target_time}")
        else:
            print(f"WARNING: {field_path} not found.")


if __name__ == "__main__":

    times123 = [10, 100, 1000, 10000]
    times4 = [1830, 2170, 2510, 2970]

    # --- p1, p2, p3 ---
    for case in ["p1", "p2", "p3"]:
        process_case(case, times123)

    # --- p4 ---
    process_case("p4", times4)

    # --- p5 & p6 only energies ---
    for case in ["p5", "p6"]:
        dirname = os.path.join(BASE_DIR, case)
        steps_path = os.path.join(dirname, "steps.dat")
        if os.path.exists(steps_path):
            n, t, T_sensor, E_sup, E_win = load_steps(steps_path)
            plot_energies(t, E_sup, E_win, f"{case}_energies.png")
        else:
            print(f"steps.dat missing for {case}")

    print("All plots generated.")
