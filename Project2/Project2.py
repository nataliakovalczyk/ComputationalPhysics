import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(
    "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/DI3_nodes_comparison.txt",
)

node = data[:, 0]
x = data[:, 1]
y = data[:, 2]
psi_fem = data[:, 3]
psi_exact = data[:, 4]
diff = data[:, 5]

plt.figure(figsize=(10, 6))
plt.plot(node, psi_fem, 'o-', label='FEM')
plt.plot(node, psi_exact, 's--', label='Exact')
plt.xlabel("Node index")
plt.ylabel("Potential")
plt.title("Potential in nodes: FEM vs exact")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("D.I.3.FEM_vs_exact.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(node, diff, 'o-')
plt.xlabel("Node index")
plt.ylabel("FEM - Exact")
plt.title("Difference between FEM and exact solution in nodes")
plt.grid(True)
plt.tight_layout()
plt.savefig("D.I.3.FEM_vs_exact_diff.png")
plt.show()

data1 = np.loadtxt(
    "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/DI4_element_scan.txt",
)

x = data1[:, 3]
y = data1[:, 4]
psi_fem = data1[:, 5]
psi_exact = data1[:, 6]
diff = data1[:, 7]

plt.figure(figsize=(8, 6))
sc = plt.scatter(x, y, c=psi_fem)
plt.colorbar(sc, label="FEM potential")
plt.xlabel("x")
plt.ylabel("y")
plt.title("D.I.4 FEM potential")
plt.grid(True)
plt.tight_layout()
plt.savefig("D.I.4_potential.png")
plt.show()

plt.figure(figsize=(8, 6))
sc = plt.scatter(x, y, c=diff)
plt.colorbar(sc, label="FEM - Exact")
plt.xlabel("x")
plt.ylabel("y")
plt.title("D.I.4 Difference: FEM - Exact")
plt.grid(True)
plt.tight_layout()
plt.savefig("D.I.4_difference.png")
plt.show()

data2 = np.loadtxt(
    "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/DI5_max_error_vs_N.txt",
)

N = data2[:, 0]
err = data2[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(N, err, 'o-')
plt.xlabel("N")
plt.ylabel("Maximum absolute error")
plt.title("D.I.5 Max |FEM - Exact| as a function of N")
plt.grid(True)
plt.tight_layout()
plt.savefig("D.I.5_error.png")
plt.show()

data3 = np.loadtxt(
    "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/DII4_element_scan.txt",
)

x2 = data3[:, 3]
y2 = data3[:, 4]
psi_fem2 = data3[:, 5]
psi_exact2 = data3[:, 6]
diff2 = data3[:, 7]

plt.figure(figsize=(8, 6))
sc = plt.scatter(x2, y2, c=psi_fem2)
plt.colorbar(sc, label="FEM potential")
plt.xlabel("x")
plt.ylabel("y")
plt.title("D.II.1 Biparabolic FEM potential")
plt.grid(True)
plt.tight_layout()
plt.savefig("D.II.1_fem_potential.png")
plt.show()

plt.figure(figsize=(8, 6))
sc = plt.scatter(x2, y2, c=diff2)
plt.colorbar(sc, label="FEM - Exact")
plt.xlabel("x")
plt.ylabel("y")
plt.title("D.II.1 Biparabolic difference: FEM - Exact")
plt.grid(True)
plt.tight_layout()
plt.savefig("D.II.1_diff_potential.png")
plt.show()

data4 = np.loadtxt(
    "/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/DII5_max_error_vs_N.txt",
)

N2 = data4[:, 0]
err2 = data4[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(N2, err2, 'o-')
plt.xlabel("N")
plt.ylabel("Maximum absolute error")
plt.title("D.II.2 Biparabolic max |FEM - Exact| as a function of N")
plt.grid(True)
plt.tight_layout()
plt.savefig("D.II.2_error.png")
plt.show()