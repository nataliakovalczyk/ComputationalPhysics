
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>

// ============================================================
// 2D membrane vibrations, finite-difference version
//
// PDE:
//      d^2 u/dt^2 = v^2 Laplacian(u) - 2d du/dt + force
//
// Assignment parameters:
//      v = 1
//      L = 1
//      default 2d = 1
//      dx = L/N
//      v dt = dx/2
//      integrate until t = 80
//      force at central node: sin(omega t)
//
// Outputs:
//      energy_vs_time_omega_pi.csv
//      avg_energy_vs_omega.csv
//      avg_energy_vs_omega_damping.csv
//      snapshots_omega_pi/snapshot_*.txt
// ============================================================

struct SimulationParams {
    double L = 1.0;
    double v = 1.0;

    // assignment uses 2d = 1 by default
    double twoD = 1.0;

    int N = 80;                  // number of intervals in each direction
    double tMax = 80.0;
    double tAverageStart = 10.0;

    int snapshotEvery = 200;     // save every this many time steps
};

using Grid = std::vector<std::vector<double>>;

Grid MakeGrid(int n, double value = 0.0) {
    return Grid(n, std::vector<double>(n, value));
}

void WriteSnapshot(
        const std::string& filename,
        const Grid& u,
        double dx
) {
    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Cannot open snapshot file: " + filename);
    }

    fout << std::fixed << std::setprecision(12);
    fout << "# x y u\n";

    int n = static_cast<int>(u.size());

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fout << i * dx << " "
                 << j * dx << " "
                 << u[i][j] << "\n";
        }
        fout << "\n";
    }
}

double MaxAbs(const Grid& u) {
    double result = 0.0;

    for (const auto& row : u) {
        for (double value : row) {
            result = std::max(result, std::fabs(value));
        }
    }

    return result;
}

// Energy:
// E = 0.5 integral [(du/dt)^2 + (du/dx)^2 + (du/dy)^2] dx dy
//
// Time derivative: backward difference (u^n - u^{n-1}) / dt.
// Spatial derivatives: central difference inside the membrane.
// Integral: simple rectangular rule over internal nodes.
double ComputeEnergy(
        const Grid& uNow,
        const Grid& uPrev,
        double dx,
        double dt
) {
    int n = static_cast<int>(uNow.size());

    double sum = 0.0;

    for (int i = 1; i < n - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            double ut = (uNow[i][j] - uPrev[i][j]) / dt;

            double ux = (uNow[i + 1][j] - uNow[i - 1][j]) / (2.0 * dx);
            double uy = (uNow[i][j + 1] - uNow[i][j - 1]) / (2.0 * dx);

            sum += ut * ut + ux * ux + uy * uy;
        }
    }

    return 0.5 * dx * dx * sum;
}

struct RunResult {
    double omega;
    double averageEnergy;
};

RunResult RunSimulation(
        const SimulationParams& params,
        double omega,
        const std::string& energyFilename,
        const std::string& snapshotPrefix,
        bool saveEnergy,
        bool saveSnapshots
) {
    const int N = params.N;

    // There are N intervals, hence N+1 nodes including boundaries.
    const int nNodes = N + 1;

    const double L = params.L;
    const double v = params.v;
    const double twoD = params.twoD;

    const double dx = L / N;

    // Assignment condition: v dt = dx/2.
    const double dt = dx / (2.0 * v);

    const int nSteps = static_cast<int>(std::ceil(params.tMax / dt));

    const int c = N / 2;

    Grid uPrev = MakeGrid(nNodes, 0.0); // u(t - dt)
    Grid uNow  = MakeGrid(nNodes, 0.0); // u(t)
    Grid uNext = MakeGrid(nNodes, 0.0); // u(t + dt)

    const double courant2 = (dt * dt * v * v) / (dx * dx);

    std::ofstream energyFile;

    if (saveEnergy) {
        energyFile.open(energyFilename);
        if (!energyFile) {
            throw std::runtime_error("Cannot open energy file: " + energyFilename);
        }

        energyFile << std::setprecision(15);
        energyFile << "t,energy,center,max_abs_u\n";
    }

    double energyIntegral = 0.0;
    double previousEnergyForAverage = 0.0;
    bool havePreviousAveragePoint = false;
    double previousAverageTime = 0.0;

    for (int step = 0; step <= nSteps; ++step) {
        double t = step * dt;

        double energy = ComputeEnergy(uNow, uPrev, dx, dt);

        if (saveEnergy) {
            energyFile << t << ","
                       << energy << ","
                       << uNow[c][c] << ","
                       << MaxAbs(uNow) << "\n";
        }

        if (t >= params.tAverageStart && t <= params.tMax) {
            if (havePreviousAveragePoint) {
                double localDt = t - previousAverageTime;
                energyIntegral += 0.5 * (energy + previousEnergyForAverage) * localDt;
            }

            previousEnergyForAverage = energy;
            previousAverageTime = t;
            havePreviousAveragePoint = true;
        }

        if (saveSnapshots && (step % params.snapshotEvery == 0 || step == nSteps)) {
            std::ostringstream name;
            name << snapshotPrefix
                 << "_snapshot_"
                 << std::setw(6) << std::setfill('0') << step
                 << ".txt";

            WriteSnapshot(name.str(), uNow, dx);
        }

        if (step == nSteps) {
            break;
        }

        // Update only interior points.
        // Boundary remains fixed: u = 0.
        for (int i = 1; i < nNodes - 1; ++i) {
            for (int j = 1; j < nNodes - 1; ++j) {
                double lap =
                        uNow[i + 1][j] + uNow[i - 1][j]
                        + uNow[i][j + 1] + uNow[i][j - 1]
                        - 4.0 * uNow[i][j];

                double force = 0.0;
                if (i == c && j == c) {
                    force = dt * dt * std::sin(omega * t);
                }

                uNext[i][j] =
                        uNow[i][j] * (2.0 - twoD * dt)
                        + uPrev[i][j] * (twoD * dt - 1.0)
                        + courant2 * lap
                        + force;
            }
        }

        // Fixed boundary conditions.
        for (int i = 0; i < nNodes; ++i) {
            uNext[i][0] = 0.0;
            uNext[i][nNodes - 1] = 0.0;
            uNext[0][i] = 0.0;
            uNext[nNodes - 1][i] = 0.0;
        }

        uPrev.swap(uNow);
        uNow.swap(uNext);

        // Clear old uPrev storage now held by uNext.
        for (int i = 0; i < nNodes; ++i) {
            std::fill(uNext[i].begin(), uNext[i].end(), 0.0);
        }
    }

    double averageEnergy =
            energyIntegral / (params.tMax - params.tAverageStart);

    return {omega, averageEnergy};
}

void ScanOmega(
        SimulationParams params,
        double omegaMin,
        double omegaMax,
        int omegaCount,
        const std::string& filename
) {
    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Cannot open scan file: " + filename);
    }

    fout << std::setprecision(15);
    fout << "omega,average_energy\n";

    for (int k = 0; k < omegaCount; ++k) {
        double omega = omegaMin
                       + (omegaMax - omegaMin) * k / static_cast<double>(omegaCount - 1);

        RunResult result = RunSimulation(
                params,
                omega,
                "",
                "",
                false,
                false
        );

        fout << result.omega << "," << result.averageEnergy << "\n";

        std::cout << "omega = " << std::setw(12) << omega
                  << "   <E> = " << result.averageEnergy << "\n";
    }

    std::cout << "Saved: " << filename << "\n";
}

void ScanOmegaForDamping(
        SimulationParams params,
        const std::vector<double>& twoDValues,
        double omegaMin,
        double omegaMax,
        int omegaCount,
        const std::string& filename
) {
    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Cannot open damping scan file: " + filename);
    }

    fout << std::setprecision(15);
    fout << "twoD,omega,average_energy\n";

    for (double twoD : twoDValues) {
        params.twoD = twoD;

        std::cout << "\nDamping scan for 2d = " << twoD << "\n";

        for (int k = 0; k < omegaCount; ++k) {
            double omega = omegaMin
                           + (omegaMax - omegaMin) * k / static_cast<double>(omegaCount - 1);

            RunResult result = RunSimulation(
                    params,
                    omega,
                    "",
                    "",
                    false,
                    false
            );

            fout << twoD << ","
                 << result.omega << ","
                 << result.averageEnergy << "\n";

            std::cout << "2d = " << std::setw(8) << twoD
                      << "   omega = " << std::setw(12) << omega
                      << "   <E> = " << result.averageEnergy << "\n";
        }
    }

    std::cout << "Saved: " << filename << "\n";
}

int main() {
    try {
        SimulationParams params;

        // You can increase N for better accuracy.
        // N = 80 is a reasonable starting value.
        params.N = 80;

        params.L = 1.0;
        params.v = 1.0;
        params.twoD = 1.0;

        params.tMax = 80.0;
        params.tAverageStart = 10.0;

        // With N=80, dt=1/(2N)=0.00625, so 200 steps is every 1.25 time units.
        params.snapshotEvery = 200;

        const double omegaPi = M_PI;

        std::cout << std::fixed << std::setprecision(12);
        std::cout << "2D membrane FD simulation\n";
        std::cout << "N = " << params.N << "\n";
        std::cout << "L = " << params.L << "\n";
        std::cout << "v = " << params.v << "\n";
        std::cout << "2d = " << params.twoD << "\n";
        std::cout << "dx = " << params.L / params.N << "\n";
        std::cout << "dt = " << (params.L / params.N) / (2.0 * params.v) << "\n\n";

        // ------------------------------------------------------------
        // 1. Assignment question: omega = pi.
        //    The file energy_vs_time_omega_pi.csv lets you check whether
        //    the energy and center motion settle into periodic behavior.
        // ------------------------------------------------------------
        RunResult piResult = RunSimulation(
                params,
                omegaPi,
                "energy_vs_time_omega_pi.csv",
                "omega_pi",
                true,
                true
        );

        std::cout << "\nomega = pi run finished\n";
        std::cout << "<E> for omega = pi: " << piResult.averageEnergy << "\n";
        std::cout << "Saved: energy_vs_time_omega_pi.csv and omega_pi_snapshot_*.txt\n\n";

        // ------------------------------------------------------------
        // 2. Average energy scan over omega in (1.5, 15).
        //    Use enough points to resolve resonance peaks.
        // ------------------------------------------------------------
        const double omegaMin = 1.5;
        const double omegaMax = 15.0;
        const int omegaCount = 271; // step about 0.05

        ScanOmega(
                params,
                omegaMin,
                omegaMax,
                omegaCount,
                "avg_energy_vs_omega.csv"
        );

        // ------------------------------------------------------------
        // 3. Damping study.
        //    Smaller 2d -> higher and narrower peaks.
        //    Larger 2d  -> lower and broader peaks.
        // ------------------------------------------------------------
        std::vector<double> dampingValues = {
                0.2,
                0.5,
                1.0,
                2.0,
                4.0
        };

        ScanOmegaForDamping(
                params,
                dampingValues,
                omegaMin,
                omegaMax,
                omegaCount,
                "avg_energy_vs_omega_damping.csv"
        );

        std::cout << "\nFinished.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
