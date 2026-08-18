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
// 2D diffusion equation, finite-difference version
//
// PDE:
//      du/dt = div(D(x,y) grad u) + S(x,y)
//
// Flux form:
//      J = -D grad u
//
// Neumann boundary condition:
//      J · n = 0
//
// Assignment parameters:
//      grid: [-N:N] x [-N:N]
//      dx = dy = 1
//      dt = 1/8
//      D = 1 in the center
//      D = 0.005 in the second margin layer of width 5
//      D = 0.0001 in the outer margin layer of width 5
//      S = s delta(i,0) delta(j,0)
//      s = 0.01
//      u(x,y,0) = 0
//
// Outputs:
//      diffusion_integral.csv
//      D_field.txt
//      diffusion_snapshot_*.txt
// ============================================================

struct SimulationParams {
    int N = 51;                  // domain is [-N:N]
    double dx = 1.0;
    double dy = 1.0;
    double dt = 1.0 / 8.0;

    double sourceStrength = 0.01;

    double tMax = 500.0;
    int snapshotEvery = 200;
};

using Grid = std::vector<std::vector<double>>;

Grid MakeGrid(int n, double value = 0.0) {
    return Grid(n, std::vector<double>(n, value));
}

void WriteSnapshot(
        const std::string& filename,
        const Grid& u,
        int N,
        double dx,
        double dy
) {
    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Cannot open snapshot file: " + filename);
    }

    fout << std::fixed << std::setprecision(12);
    fout << "# x y u\n";

    int n = static_cast<int>(u.size());

    for (int i = 0; i < n; ++i) {
        double x = (i - N) * dx;

        for (int j = 0; j < n; ++j) {
            double y = (j - N) * dy;

            fout << x << " "
                 << y << " "
                 << u[i][j] << "\n";
        }

        fout << "\n";
    }
}

void InitializeDiffusionCoefficient(Grid& D) {
    int n = static_cast<int>(D.size());

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int distanceFromEdge = std::min({
                                                    i,
                                                    j,
                                                    n - 1 - i,
                                                    n - 1 - j
                                            });

            if (distanceFromEdge < 5) {
                D[i][j] = 0.0001;
            } else if (distanceFromEdge < 10) {
                D[i][j] = 0.005;
            } else {
                D[i][j] = 1.0;
            }
        }
    }
}

void InitializeSource(
        Grid& S,
        int center,
        double sourceStrength
) {
    S[center][center] = sourceStrength;
}

void ApplyNeumannBoundary(Grid& u) {
    int n = static_cast<int>(u.size());

    for (int j = 0; j < n; ++j) {
        u[0][j]     = u[1][j];
        u[n - 1][j] = u[n - 2][j];
    }

    for (int i = 0; i < n; ++i) {
        u[i][0]     = u[i][1];
        u[i][n - 1] = u[i][n - 2];
    }
}

double ComputeIntegral(
        const Grid& u,
        double dx,
        double dy
) {
    double sum = 0.0;

    for (const auto& row : u) {
        for (double value : row) {
            sum += value;
        }
    }

    return sum * dx * dy;
}

void EulerStep(
        const Grid& u,
        Grid& uNew,
        const Grid& D,
        const Grid& S,
        double dx,
        double dy,
        double dt
) {
    int n = static_cast<int>(u.size());

    for (int i = 1; i < n - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            double D_x_minus = 0.5 * (D[i][j] + D[i - 1][j]);
            double D_x_plus  = 0.5 * (D[i][j] + D[i + 1][j]);

            double D_y_minus = 0.5 * (D[i][j] + D[i][j - 1]);
            double D_y_plus  = 0.5 * (D[i][j] + D[i][j + 1]);

            double Fx_minus =
                    -D_x_minus * (u[i][j] - u[i - 1][j]) / dx;

            double Fx_plus =
                    -D_x_plus * (u[i + 1][j] - u[i][j]) / dx;

            double Fy_minus =
                    -D_y_minus * (u[i][j] - u[i][j - 1]) / dy;

            double Fy_plus =
                    -D_y_plus * (u[i][j + 1] - u[i][j]) / dy;

            double divergence =
                    (Fx_minus - Fx_plus) / dx
                    + (Fy_minus - Fy_plus) / dy;

            uNew[i][j] =
                    u[i][j]
                    + dt * (divergence + S[i][j]);
        }
    }

    ApplyNeumannBoundary(uNew);
}

void RunSimulation(const SimulationParams& params) {
    const int N = params.N;
    const int nNodes = 2 * N + 1;
    const int center = N;

    Grid u    = MakeGrid(nNodes, 0.0);
    Grid uNew = MakeGrid(nNodes, 0.0);
    Grid D    = MakeGrid(nNodes, 1.0);
    Grid S    = MakeGrid(nNodes, 0.0);

    InitializeDiffusionCoefficient(D);
    InitializeSource(S, center, params.sourceStrength);

    WriteSnapshot("D_field.txt", D, N, params.dx, params.dy);

    int nSteps = static_cast<int>(std::ceil(params.tMax / params.dt));

    std::ofstream integralFile("diffusion_integral.csv");
    if (!integralFile) {
        throw std::runtime_error("Cannot open diffusion_integral.csv");
    }

    integralFile << std::setprecision(15);
    integralFile << "t,I,st,difference,relative_difference,center,max_u\n";

    for (int step = 0; step <= nSteps; ++step) {
        double t = step * params.dt;

        double I = ComputeIntegral(u, params.dx, params.dy);
        double st = params.sourceStrength * t;
        double difference = I - st;

        double relativeDifference = 0.0;
        if (std::fabs(st) > 1e-14) {
            relativeDifference = difference / st;
        }

        double maxU = 0.0;
        for (const auto& row : u) {
            for (double value : row) {
                maxU = std::max(maxU, value);
            }
        }

        integralFile << t << ","
                     << I << ","
                     << st << ","
                     << difference << ","
                     << relativeDifference << ","
                     << u[center][center] << ","
                     << maxU << "\n";

        if (step % params.snapshotEvery == 0 || step == nSteps) {
            std::ostringstream name;
            name << "diffusion_snapshot_"
                 << std::setw(6) << std::setfill('0') << step
                 << ".txt";

            WriteSnapshot(name.str(), u, N, params.dx, params.dy);
        }

        if (step == nSteps) {
            break;
        }

        EulerStep(
                u,
                uNew,
                D,
                S,
                params.dx,
                params.dy,
                params.dt
        );

        u.swap(uNew);

        for (int i = 0; i < nNodes; ++i) {
            std::fill(uNew[i].begin(), uNew[i].end(), 0.0);
        }
    }
    
    std::cout << "Simulation finished.\n";
    std::cout << "Saved: diffusion_integral.csv\n";
    std::cout << "Saved: D_field.txt\n";
    std::cout << "Saved: diffusion_snapshot_*.txt\n";
}

int main() {
    try {
        SimulationParams params;

        params.N = 51;
        params.dx = 1.0;
        params.dy = 1.0;
        params.dt = 1.0 / 8.0;

        params.sourceStrength = 0.01;

        params.tMax = 500.0;
        params.snapshotEvery = 200;

        std::cout << std::fixed << std::setprecision(12);
        std::cout << "2D diffusion FD simulation\n";
        std::cout << "N = " << params.N << "\n";
        std::cout << "nodes = " << 2 * params.N + 1 << " x "
                  << 2 * params.N + 1 << "\n";
        std::cout << "dx = " << params.dx << "\n";
        std::cout << "dy = " << params.dy << "\n";
        std::cout << "dt = " << params.dt << "\n";
        std::cout << "source strength s = "
                  << params.sourceStrength << "\n";
        std::cout << "tMax = " << params.tMax << "\n\n";

        RunSimulation(params);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }


    return 0;
}