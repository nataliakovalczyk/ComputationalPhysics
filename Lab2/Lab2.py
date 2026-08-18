import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/energy_phase_space.csv")

# energies
plt.figure(figsize=(8,4.5))
plt.plot(df["t"], df["Ekin"], label="E_kin")
plt.plot(df["t"], df["Epot"], label="E_pot")
plt.plot(df["t"], df["Etot"], label="E_tot", linestyle="--")
plt.xlabel("t")
plt.ylabel("Energy")
plt.title("Energies vs time")
plt.legend()
plt.tight_layout()
plt.savefig("energies.png", dpi=200)

# trajectory
plt.figure(figsize=(8,4.5))
plt.plot(df["t"], df["x"], label="x(t)")
plt.plot(df["t"], df["v"], label="v(t)")
plt.xlabel("t")
plt.ylabel("x, v")
plt.title("Time-domain trajectories")
plt.legend()
plt.tight_layout()
plt.savefig("time_series.png", dpi=200)

# phase space
plt.figure(figsize=(5,5))
plt.plot(df["x"], df["v"])
plt.xlabel("x")
plt.ylabel("v")
plt.title("Phase space: v vs x")
plt.tight_layout()
plt.savefig("phase_space.png", dpi=200)

print("Saved: energies.png, time_series.png, phase_space.png")

# alpha changed

df1 = pd.read_csv("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/oscillator_damped.csv")

# numeric vs exact
plt.figure(figsize=(8,4.5))
plt.plot(df1["t"], df1["x_num"], label="x(t) numeric")
plt.plot(df1["t"], df1["x_exact"], label="x(t) exact", linestyle="--")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Underdamped oscillator: numeric vs exact (α = 0.1)")
plt.legend()
plt.tight_layout()
plt.savefig("x_compare.png", dpi=200)

# phase space
plt.figure(figsize=(5,5))
plt.plot(df1["x_num"], df1["v_num"])
plt.xlabel("x")
plt.ylabel("v")
plt.title("Phase space: v vs x (α = 0.1)")
plt.tight_layout()
plt.savefig("phase_space_damped.png", dpi=200)

# energies
plt.figure(figsize=(8,4.5))
plt.plot(df1["t"], df1["Ekin"], label="E_kin")
plt.plot(df1["t"], df1["Epot"], label="E_pot")
plt.plot(df1["t"], df1["Etot"], label="E_tot", linestyle="--")
plt.xlabel("t")
plt.ylabel("Energy")
plt.title("Energy decay with damping (α = 0.1)")
plt.legend()
plt.tight_layout()
plt.savefig("energies_damped.png", dpi=200)

print("Saved: x_compare.png, phase_space_damped.png, energies_damped.png")

# four alphas
df2 = pd.read_csv("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/trajectories.csv")

plt.figure(figsize=(9,5))
for col in df2.columns:
    if col == "t":
        continue
    plt.plot(df2["t"], df2[col], label=col.replace("x_alpha_","α="))

plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Damped oscillator: x(t) for different α (Fext=0)")
plt.legend()
plt.tight_layout()
plt.savefig("x_all_alphas.png", dpi=200)
print("Saved x_all_alphas.png")

# test before sweep

df = pd.read_csv("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/driven_test.csv")
plt.figure(figsize=(9,4.5))
plt.plot(df["t"], df["x"], lw=1)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Driven damped oscillator: α=1.0, F0=1, Ω_ext=0.5 ω0")
plt.tight_layout()
plt.savefig("driven_test_xt.png", dpi=200)

# resonance curves
sweep = pd.read_csv("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/amplitude_sweep.csv")
plt.figure(figsize=(8,5))
for col in sweep.columns:
    if col.startswith("A_alpha_"):
        label = r"$\alpha={}$".format(col.split("_")[-1])
        plt.plot(sweep["Omega"], sweep[col], label=label)

if "A_analytic_alpha_1.0" in sweep.columns:
    plt.plot(sweep["Omega"], sweep["A_analytic_alpha_1.0"],
             linestyle="--", label="analytic α=1.0")

plt.xlabel(r"$\Omega_{\mathrm{ext}}$")
plt.ylabel(r"$x_{\max}$ (last detected peak)")
plt.yscale("log")
plt.title("Driven amplitude vs driving frequency")
plt.legend()
plt.tight_layout()
plt.savefig("resonance_curves.png", dpi=200)

print("Saved: driven_test_xt.png, resonance_curves.png")