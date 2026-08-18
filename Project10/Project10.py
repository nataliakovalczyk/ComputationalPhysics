import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/diffusion_integral.csv")

plt.figure(figsize=(8,5))

plt.plot(data["t"], data["I"],
         label="Numerical integral $I(t)$",
         linewidth=2)

plt.plot(data["t"], data["st"],
         "--",
         label="$s t$",
         linewidth=2)

plt.xlabel("t")
plt.ylabel("Integral")
plt.title("Mass conservation test")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("integral_vs_st.png", dpi=300)
plt.show()


plt.figure(figsize=(8,5))

plt.plot(
    data["t"][1:],
    data["relative_difference"][1:],
    linewidth=2
)

plt.xlabel("t")
plt.ylabel(r"$(I-st)/(st)$")
plt.title("Relative error")
plt.grid()

plt.tight_layout()
plt.savefig("relative_error.png", dpi=300)
plt.show()


files = [
    "diffusion_snapshot_000000.txt",
    "diffusion_snapshot_001000.txt",
    "diffusion_snapshot_003000.txt",
    "diffusion_snapshot_004000.txt"
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, file in zip(axes.flat, files):

    data = np.loadtxt(file)

    x = data[:,0]
    y = data[:,1]
    u = data[:,2]

    nx = len(np.unique(x))
    ny = len(np.unique(y))

    U = u.reshape(nx, ny)

    im = ax.imshow(
        U.T,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal"
    )

    ax.set_title(file)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

# leave room on the right for colorbar
fig.subplots_adjust(right=0.88)

# dedicated colorbar axis
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
fig.colorbar(im, cax=cbar_ax, label="u")

plt.savefig("snapshots_evolution.png", dpi=300)
plt.show()

data = np.loadtxt("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/D_field.txt")

x = data[:,0]
y = data[:,1]
D = data[:,2]

nx = len(np.unique(x))
ny = len(np.unique(y))

D = D.reshape(nx, ny)

plt.figure(figsize=(7,6))

from matplotlib.colors import LogNorm

plt.imshow(
    D.T,
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    aspect="equal",
    norm=LogNorm(vmin=1e-4, vmax=1)
)

plt.colorbar(label="D(x,y)")
plt.title("Spatial diffusion coefficient")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.savefig("D_field.png", dpi=300)
plt.show()