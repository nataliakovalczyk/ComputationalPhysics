import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

L = 5.0
Nwave = 10

def load_spectrum_vs_n(filename):
    df = pd.read_csv(
        filename,
        comment="#",
        delim_whitespace=True,
        header=None,
        names=["basis", "N", "state", "nx", "ny", "E_num", "E_exact", "abs_error"]
    )
    return df

def load_spectrum_fixed_n(filename):
    df = pd.read_csv(
        filename,
        comment="#",
        delim_whitespace=True,
        header=None,
        names=["state", "nx", "ny", "E_num", "E_exact", "abs_error"]
    )
    return df

def load_wavefunction(filename):
    df = pd.read_csv(
        filename,
        comment="#",
        delim_whitespace=True,
        header=None,
        names=["node", "x", "y", "psi"]
    )
    return df

def plot_spectrum_comparison():
    df_bilin = load_spectrum_fixed_n(f"bilinear_spectrum_N{Nwave}.txt")
    df_bipara = load_spectrum_fixed_n(f"biparabolic_spectrum_N{Nwave}.txt")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_bilin["state"], df_bilin["E_exact"], "o-", label="Exact")
    ax.plot(df_bilin["state"], df_bilin["E_num"], "s--", label="Bilinear")
    ax.plot(df_bipara["state"], df_bipara["E_num"], "d--", label="Biparabolic")

    ax.set_xlabel("State index")
    ax.set_ylabel("Energy")
    ax.set_title(f"First 10 positive eigenvalues (N = {Nwave})")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("plot_spectrum_comparison.png", dpi=200)
    plt.show()

def plot_convergence_energy(df, basis_name, states=(1, 2, 3, 4, 5)):
    fig, ax = plt.subplots(figsize=(8, 5))

    for s in states:
        sub = df[df["state"] == s].sort_values("N")
        ax.plot(sub["N"], sub["E_num"], "o-", label=f"state {s}")

        e_exact = sub["E_exact"].iloc[0]
        ax.axhline(e_exact, linestyle="--", linewidth=1)

    ax.set_xlabel("N")
    ax.set_ylabel("Energy")
    ax.set_title(f"Convergence of eigenvalues with N ({basis_name})")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"plot_convergence_energy_{basis_name}.png", dpi=200)
    plt.show()

def plot_convergence_error(df, basis_name, states=(1, 2, 3, 4, 5)):
    fig, ax = plt.subplots(figsize=(8, 5))

    for s in states:
        sub = df[df["state"] == s].sort_values("N")
        ax.plot(sub["N"], sub["abs_error"], "o-", label=f"state {s}")

    ax.set_xlabel("N")
    ax.set_ylabel(r"$|E_{\mathrm{num}} - E_{\mathrm{exact}}|$")
    ax.set_title(f"Convergence error vs N ({basis_name})")
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"plot_convergence_error_{basis_name}.png", dpi=200)
    plt.show()

def plot_wavefunction(filename, title, outname):
    df = load_wavefunction(filename)

    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())

    pivot = df.pivot_table(index="y", columns="x", values="psi", aggfunc="mean")
    pivot = pivot.reindex(index=np.sort(pivot.index.values), columns=np.sort(pivot.columns.values))

    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values

    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.contourf(X, Y, Z, levels=30)
    fig.colorbar(c, ax=ax, label=r"$\psi(x,y)$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outname, dpi=200)
    plt.show()

def plot_all_wavefunctions():
    for basis in ["bilinear", "biparabolic"]:
        for state in [1, 2, 3, 4]:
            filename = f"{basis}_N{Nwave}_state_{state}.txt"
            if Path(filename).exists():
                plot_wavefunction(
                    filename,
                    f"{basis.capitalize()} basis, state {state}, N={Nwave}",
                    f"plot_{basis}_state_{state}.png"
                )

def make_summary_tables():
    df_bilin = load_spectrum_fixed_n(f"bilinear_spectrum_N{Nwave}.txt")
    df_bipara = load_spectrum_fixed_n(f"biparabolic_spectrum_N{Nwave}.txt")

    merged = pd.DataFrame({
        "state": df_bilin["state"],
        "nx": df_bilin["nx"],
        "ny": df_bilin["ny"],
        "E_exact": df_bilin["E_exact"],
        "E_bilinear": df_bilin["E_num"],
        "err_bilinear": df_bilin["abs_error"],
        "E_biparabolic": df_bipara["E_num"],
        "err_biparabolic": df_bipara["abs_error"],
    })

    merged.to_csv("table_spectrum_comparison.csv", index=False)
    print("\nSpectrum comparison table:\n")
    print(merged.to_string(index=False))

def main():
    df_bilin = load_spectrum_vs_n("bilinear_spectrum_vs_N.txt")
    df_bipara = load_spectrum_vs_n("biparabolic_spectrum_vs_N.txt")

    make_summary_tables()

    plot_spectrum_comparison()

    plot_convergence_energy(df_bilin, "bilinear")
    plot_convergence_energy(df_bipara, "biparabolic")

    plot_convergence_error(df_bilin, "bilinear")
    plot_convergence_error(df_bipara, "biparabolic")

    plot_all_wavefunctions()

if __name__ == "__main__":
    main()