import pandas as pd
import matplotlib.pyplot as plt

for filename, title in [
    ("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/resonant_probabilities.csv", "Resonant driving"),
    ("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/half_resonant_probabilities.csv", "Half-resonant driving")
]:
    df = pd.read_csv(filename)

    plt.figure()
    plt.plot(df["t"], df["P0"], label="ground state")
    plt.plot(df["t"], df["P1"], label="1st excited")
    plt.plot(df["t"], df["P2"], label="2nd excited")
    plt.plot(df["t"], df["leakage_above_first_excited"], label="leakage")

    plt.xlabel("t")
    plt.ylabel("probability")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename.replace(".csv", ".png"), dpi=300)
    plt.show()

df = pd.read_csv("/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/leakage_vs_amplitude.csv")

plt.figure()
plt.plot(df["eF"], df["max_leakage"], "o-")
plt.xlabel("field amplitude eF")
plt.ylabel("maximum leakage")
plt.title("Leakage versus electric field amplitude")
plt.grid()
plt.savefig("leakage_vs_amplitude.png", dpi=300)
plt.show()