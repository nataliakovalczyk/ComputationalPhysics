import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL MATPLOTLIB STYLE
# ============================================================
T_UNIT = 119.0      # reduced → Kelvin
IT_SCALE = 1e4      # for x-axis labeling

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 11,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "figure.dpi": 120,
})

COL = {
    "red":   "#ed553b",
    "blue":  "#20639b",
    "black": "#173f5f",
    "green": "#3caea3",
}

BASE = "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug"
L = 25.0


# task 1
data = np.loadtxt(f"{BASE}/pos_vel_init.dat")
x, y, vx, vy = data.T

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(x, y, s=18, color=COL["red"], label="Particles")
ax.quiver(x, y, vx, vy, color=COL["black"], scale_units="xy", scale=1.0)
ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0],
        color=COL["blue"], label="Box")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Initial positions and velocities")
ax.set_aspect("equal")
ax.set_xlim(-2, L + 2)
ax.set_ylim(-2, L + 2)

ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("task1.png")
plt.show()


# task 2

energy = np.loadtxt(f"{BASE}/energies_2a.dat")
traj   = np.loadtxt(f"{BASE}/traj.dat")

it  = energy[:,0]
Ekin, Epot, Etot, Tred = energy[:,2:6].T
T_K = Tred * T_UNIT

x, y = traj[:,2], traj[:,3]

# Remove jumps across PBC
dx, dy = np.abs(np.diff(x)), np.abs(np.diff(y))
jumps = (dx > L/2) | (dy > L/2)
x[1:][jumps] = np.nan
y[1:][jumps] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) Energy vs iterations
axE = axes[0]
axE.plot(it/IT_SCALE, Ekin, color=COL["red"],   label=r"$E_{kin}$")
axE.plot(it/IT_SCALE, Epot, color=COL["blue"],  label=r"$E_{pot}$")
axE.plot(it/IT_SCALE, Etot, color=COL["black"], label=r"$E_{tot}$")

axE.set_xlabel(r"$it \times 10^4$")
axE.set_ylabel("Energy")
axE.set_title("Energy conservation (NVE)")
axE.legend(loc="upper right")

axT = axE.twinx()
axT.plot(it/IT_SCALE, T_K, color=COL["green"], label=r"$T(t)$")
axT.set_ylabel("Temperature [K]")
axT.set_ylim(260, 400)
axT.legend(loc="upper left")

# (b) Trajectory
ax = axes[1]
ax.plot(x, y, color=COL["red"])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"Trajectory of atom $i=N/2$")
ax.set_aspect("equal")
ax.set_xlim(0, L)
ax.set_ylim(0, L)

plt.tight_layout()
plt.savefig("task2.png")
plt.show()


# task 3

Qvals = [1.0, 0.1, 0.01]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, Q in zip(axes, Qvals):
    data = np.loadtxt(f"{BASE}/task3_Q_{Q:.6f}.dat")

    it = data[:,0]
    Ekin, Epot, Etot, Tred = data[:,2:6].T
    T_K = Tred * T_UNIT

    ax.plot(it/IT_SCALE, Ekin, color=COL["red"],   label=r"$E_{kin}$")
    ax.plot(it/IT_SCALE, Epot, color=COL["blue"],  label=r"$E_{pot}$")
    ax.plot(it/IT_SCALE, Etot, color=COL["black"], label=r"$E_{tot}$")

    axT = ax.twinx()
    axT.plot(it/IT_SCALE, T_K, color=COL["green"], label=r"$T(t)$")
    axT.set_ylim(260, 400)
    axT.set_ylabel("Temperature [K]")

    ax.set_xlabel(r"$it \times 10^4$")
    ax.set_ylabel("Energy")
    ax.set_title(rf"$Q={Q}$")

    ax.legend(loc="upper right")
    axT.legend(loc="upper left")

plt.tight_layout()
plt.savefig("task3.png")
plt.show()



# task 4

Nit_list = [1000, 10000, 100000]
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, Nit in zip(axes, Nit_list):
    data = np.loadtxt(f"{BASE}/task4_Nit_{Nit}.dat")
    v, vhist, f_exact = data.T
    dv = v[1] - v[0]

    ax.bar(v, vhist, width=dv, facecolor="none",
           edgecolor=COL["red"], label="Histogram")
    ax.plot(v, f_exact, color=COL["black"], label="Exact")

    ax.set_xlabel("v")
    ax.set_ylabel(r"$f_v(v)$")
    ax.set_title(rf"$N_{{it}}={Nit}$")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("task4.png")
plt.show()


#  task 5

fastT = np.loadtxt(f"{BASE}/task5_T_fast.dat")
slowT = np.loadtxt(f"{BASE}/task5_T_slow.dat")

it_fast = fastT[:,0] / 1e4
T_fast  = fastT[:,2]
it_slow = slowT[:,0] / 1e4
T_slow  = slowT[:,2]

pos_fast = np.loadtxt(f"{BASE}/task5_pos_fast.dat")
pos_slow = np.loadtxt(f"{BASE}/task5_pos_slow.dat")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.plot(it_fast, T_fast, color=COL["red"],  label=r"$N_{it}=10^4$")
ax.plot(it_slow, T_slow, color=COL["blue"], label=r"$N_{it}=10^5$")
ax.set_xlabel(r"$it \times 10^4$")
ax.set_ylabel("T")
ax.set_title("Cooling dynamics")
ax.legend(loc="upper right")

axes[1].scatter(pos_fast[:,0], pos_fast[:,1], color=COL["red"])
axes[1].set_title("Fast cooling")
axes[1].set_aspect("equal")

axes[2].scatter(pos_slow[:,0], pos_slow[:,1], color=COL["blue"])
axes[2].set_title("Slow cooling")
axes[2].set_aspect("equal")

plt.tight_layout()
plt.savefig("task5.png")
plt.show()
