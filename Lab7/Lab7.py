import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/energies.txt")

t = data[:, 0]
Ekin = data[:, 1]
Epot = data[:, 2]
Etot = data[:, 3]

plt.figure(figsize=(8, 5))
plt.plot(t, Ekin, label="Kinetic")
plt.plot(t, Epot, label="Potential")
plt.plot(t, Etot, label="Total", linewidth=2)

plt.xlabel("time t")
plt.ylabel("Energy")
plt.title("Energies vs time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("energies_5.png")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/u_map.txt")

t = data[:, 0]
u = data[:, 1:]

plt.figure(figsize=(9, 5))
plt.imshow(
    u,
    extent=[0, 1, t[-1], t[0]],   # x from 0 to 1, time reversed so t=0 at top
    aspect="auto",
    cmap="seismic"
)

plt.colorbar(label="u(x,t)")
plt.xlabel("Position x")
plt.ylabel("Time t")
plt.title("Displacement u(x,t) — Heatmap")
plt.tight_layout()
plt.savefig("u_map_5.png")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/modes.txt")

t = data[:, 0]

# Columns: t, b1, d1, e1, b2, d2, e2, ...
b = data[:, 1::3]  # start at 1, step 3 → b1, b2, b3, ...
d = data[:, 2::3]  # start at 2, step 3 → d1, d2, d3, ...
e = data[:, 3::3]  # start at 3, step 3 → e1, e2, e3, ...

Kmodes = b.shape[1]

plt.figure(figsize=(8, 5))
for k in range(Kmodes):
    plt.plot(t, e[:, k], label=f"e{k+1}")
plt.xlabel("t")
plt.ylabel("Energy per mode")
plt.title("Mode Energy Contributions $e_k(t)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("modes_5.png")
plt.show()

