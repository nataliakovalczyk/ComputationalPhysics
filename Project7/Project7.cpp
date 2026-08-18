#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <sstream>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// ============================================================
// Classical wave equation on a square membrane, FEM version
//
// PDE:
//      d^2 Psi / dt^2 = v^2 Laplacian(Psi)
//
// FEM semi-discrete equation:
//      O c''(t) = -v^2 S c(t)
//
// Explicit central time scheme:
//      O c^{n+1} = (2O - v^2 dt^2 S)c^n - O c^{n-1}
//
// Boundary:
//      c_k = 0 on outer boundary
//      central node is driven: c_center(t) = sin(omega t)
//
// Mesh:
//      bilinear 4-node square elements, same style as previous assignments.
//      N = 10, L = 5, as in previous potential-well wave/eigenstate output.
// ============================================================

struct Node {
    double x, y;
};

struct Mesh {
    std::vector<Node> p;
    std::vector<std::vector<int>> nlg;
    int N;
    double L;
};

// ============================================================
// Bilinear basis
// ============================================================

double f1(double xi) { return 0.5 * (1.0 - xi); }
double f2(double xi) { return 0.5 * (1.0 + xi); }

double df1(double) { return -0.5; }
double df2(double) { return  0.5; }

double g(int i, double xi1, double xi2) {
    switch (i) {
        case 0: return f1(xi1) * f1(xi2);
        case 1: return f2(xi1) * f1(xi2);
        case 2: return f1(xi1) * f2(xi2);
        case 3: return f2(xi1) * f2(xi2);
        default: return 0.0;
    }
}

double dg_dxi1(int i, double xi1, double xi2) {
    switch (i) {
        case 0: return df1(xi1) * f1(xi2);
        case 1: return df2(xi1) * f1(xi2);
        case 2: return df1(xi1) * f2(xi2);
        case 3: return df2(xi1) * f2(xi2);
        default: return 0.0;
    }
}

double dg_dxi2(int i, double xi1, double xi2) {
    switch (i) {
        case 0: return f1(xi1) * df1(xi2);
        case 1: return f2(xi1) * df1(xi2);
        case 2: return f1(xi1) * df2(xi2);
        case 3: return f2(xi1) * df2(xi2);
        default: return 0.0;
    }
}

Mesh GenerateBilinearMesh(int N, double L) {
    double a = L / N;
    double tol = a / 10.0;

    int nElem = N * N;
    int nLocal = 4;

    Mesh mesh;
    mesh.N = N;
    mesh.L = L;
    mesh.nlg.assign(nElem, std::vector<int>(nLocal, -1));

    double rb[4][2] = {
            {0.0, 0.0},
            {a,   0.0},
            {0.0, a  },
            {a,   a  }
    };

    int inumer = 0;
    int elem = 0;

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            Node nd;
            nd.x = i * a;
            nd.y = j * a;
            mesh.p.push_back(nd);

            mesh.nlg[elem][0] = inumer;
            inumer++;
            elem++;
        }
    }

    int nelem = elem;

    for (int inr = 0; inr < nelem; ++inr) {
        for (int inw = 1; inw < nLocal; ++inw) {
            double xp = mesh.p[inr].x + rb[inw][0];
            double yp = mesh.p[inr].y + rb[inw][1];

            int ibyl = -1;

            for (int ie = 0; ie < inumer; ++ie) {
                double dr = std::fabs(xp - mesh.p[ie].x) + std::fabs(yp - mesh.p[ie].y);
                if (dr < tol) {
                    ibyl = ie;
                    break;
                }
            }

            if (ibyl == -1) {
                Node nd;
                nd.x = xp;
                nd.y = yp;
                mesh.p.push_back(nd);
                mesh.nlg[inr][inw] = inumer;
                inumer++;
            } else {
                mesh.nlg[inr][inw] = ibyl;
            }
        }
    }

    return mesh;
}

bool IsBoundaryNode(const Node& nd, double L, double eps = 1e-12) {
    return std::fabs(nd.x) < eps ||
           std::fabs(nd.x - L) < eps ||
           std::fabs(nd.y) < eps ||
           std::fabs(nd.y - L) < eps;
}

int FindCentralNode(const Mesh& mesh) {
    const double xc = mesh.L / 2.0;
    const double yc = mesh.L / 2.0;

    int best = -1;
    double bestDist = 1e100;

    for (int i = 0; i < static_cast<int>(mesh.p.size()); ++i) {
        double dx = mesh.p[i].x - xc;
        double dy = mesh.p[i].y - yc;
        double d2 = dx * dx + dy * dy;

        if (d2 < bestDist) {
            bestDist = d2;
            best = i;
        }
    }

    return best;
}

std::array<double, 4> GaussWeights4() {
    double sqrt30 = std::sqrt(30.0);
    return {
            (18.0 + sqrt30) / 36.0,
            (18.0 + sqrt30) / 36.0,
            (18.0 - sqrt30) / 36.0,
            (18.0 - sqrt30) / 36.0
    };
}

std::array<double, 4> GaussPoints4() {
    return {
            -std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0)),
            std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0)),
            std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0)),
            -std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0))
    };
}

std::array<std::array<double, 4>, 4> ComputeLocalStiffnessMatrix() {
    std::array<std::array<double, 4>, 4> S{};

    auto w = GaussWeights4();
    auto gamma = GaussPoints4();

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            double sum = 0.0;

            for (int l = 0; l < 4; ++l) {
                for (int n = 0; n < 4; ++n) {
                    double xi1 = gamma[l];
                    double xi2 = gamma[n];

                    double term =
                            dg_dxi1(j, xi1, xi2) * dg_dxi1(i, xi1, xi2) +
                            dg_dxi2(j, xi1, xi2) * dg_dxi2(i, xi1, xi2);

                    sum += w[l] * w[n] * term;
                }
            }

            S[j][i] = sum;
        }
    }

    return S;
}

std::array<std::array<double, 4>, 4> ComputeLocalOverlapMatrix(double a) {
    std::array<std::array<double, 4>, 4> O{};

    auto w = GaussWeights4();
    auto gamma = GaussPoints4();

    double jacobian = (a * a) / 4.0;

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            double sum = 0.0;

            for (int l = 0; l < 4; ++l) {
                for (int n = 0; n < 4; ++n) {
                    double xi1 = gamma[l];
                    double xi2 = gamma[n];

                    sum += w[l] * w[n] * g(i, xi1, xi2) * g(j, xi1, xi2);
                }
            }

            O[j][i] = jacobian * sum;
        }
    }

    return O;
}

void AssembleGlobalMatrices(
        const Mesh& mesh,
        const std::array<std::array<double, 4>, 4>& Sloc,
        const std::array<std::array<double, 4>, 4>& Oloc,
        Eigen::MatrixXd& S,
        Eigen::MatrixXd& O
) {
    int nNodes = static_cast<int>(mesh.p.size());

    S = Eigen::MatrixXd::Zero(nNodes, nNodes);
    O = Eigen::MatrixXd::Zero(nNodes, nNodes);

    for (size_t e = 0; e < mesh.nlg.size(); ++e) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int gi = mesh.nlg[e][i];
                int gj = mesh.nlg[e][j];

                S(gi, gj) += Sloc[i][j];
                O(gi, gj) += Oloc[i][j];
            }
        }
    }
}

std::vector<int> GetFreeNodes(const Mesh& mesh, int centerNode) {
    std::vector<int> freeNodes;

    for (int i = 0; i < static_cast<int>(mesh.p.size()); ++i) {
        if (!IsBoundaryNode(mesh.p[i], mesh.L) && i != centerNode) {
            freeNodes.push_back(i);
        }
    }

    return freeNodes;
}

Eigen::MatrixXd ExtractSubmatrix(
        const Eigen::MatrixXd& A,
        const std::vector<int>& rows,
        const std::vector<int>& cols
) {
    Eigen::MatrixXd B(rows.size(), cols.size());

    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        for (int j = 0; j < static_cast<int>(cols.size()); ++j) {
            B(i, j) = A(rows[i], cols[j]);
        }
    }

    return B;
}

Eigen::VectorXd ExtractColumn(
        const Eigen::MatrixXd& A,
        const std::vector<int>& rows,
        int col
) {
    Eigen::VectorXd b(rows.size());

    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        b(i) = A(rows[i], col);
    }

    return b;
}


double ComputeStableDt(
        const Eigen::MatrixXd& Sff,
        const Eigen::MatrixXd& Off,
        double v,
        double safety
) {
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(Sff, Off);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Generalized eigenvalue problem S u = lambda O u failed.");
    }

    double lambdaMax = solver.eigenvalues().maxCoeff();

    // For central difference applied to modal equation q'' + v^2 lambda q = 0:
    //      dt <= 2 / (v sqrt(lambdaMax))
    double dtCritical = 2.0 / (v * std::sqrt(lambdaMax));

    return safety * dtCritical;
}

void SaveSnapshot(
        const Mesh& mesh,
        const Eigen::VectorXd& cFull,
        const std::string& filename
) {
    std::ofstream fout(filename);

    if (!fout) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }

    fout << std::fixed << std::setprecision(12);
    fout << "# x y psi\n";

    for (int i = 0; i < static_cast<int>(mesh.p.size()); ++i) {
        fout << std::setw(18) << mesh.p[i].x
             << std::setw(18) << mesh.p[i].y
             << std::setw(18) << cFull(i)
             << "\n";
    }
}

void SaveEnergyHeader(const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Cannot open energy file: " + filename);
    }

    fout << "# t kinetic potential total center_value max_abs_psi\n";
}

void AppendEnergy(
        const std::string& filename,
        double t,
        const Eigen::VectorXd& velocityFree,
        const Eigen::VectorXd& cFree,
        const Eigen::MatrixXd& Off,
        const Eigen::MatrixXd& Sff,
        double centerValue,
        double maxAbsPsi,
        double v
) {
    double kinetic = 0.5 * velocityFree.dot(Off * velocityFree);
    double potential = 0.5 * v * v * cFree.dot(Sff * cFree);
    double total = kinetic + potential;

    std::ofstream fout(filename, std::ios::app);
    if (!fout) {
        throw std::runtime_error("Cannot open energy file: " + filename);
    }

    fout << std::fixed << std::setprecision(12)
         << std::setw(18) << t
         << std::setw(18) << kinetic
         << std::setw(18) << potential
         << std::setw(18) << total
         << std::setw(18) << centerValue
         << std::setw(18) << maxAbsPsi
         << "\n";
}

Eigen::VectorXd BuildFullVector(
        int nNodes,
        const std::vector<int>& freeNodes,
        const Eigen::VectorXd& cFree,
        int centerNode,
        double centerValue
) {
    Eigen::VectorXd cFull = Eigen::VectorXd::Zero(nNodes);

    for (int i = 0; i < static_cast<int>(freeNodes.size()); ++i) {
        cFull(freeNodes[i]) = cFree(i);
    }

    cFull(centerNode) = centerValue;

    return cFull;
}

// ============================================================
// One forced-oscillation run
// ============================================================

void RunForcedOscillation(
        const Mesh& mesh,
        const Eigen::MatrixXd& S,
        const Eigen::MatrixXd& O,
        const std::vector<int>& freeNodes,
        int centerNode,
        double omega,
        double dt,
        double tMax,
        int snapshotEvery,
        const std::string& prefix,
        double v
) {
    int nNodes = static_cast<int>(mesh.p.size());
    int nFree = static_cast<int>(freeNodes.size());

    Eigen::MatrixXd Sff = ExtractSubmatrix(S, freeNodes, freeNodes);
    Eigen::MatrixXd Off = ExtractSubmatrix(O, freeNodes, freeNodes);

    Eigen::VectorXd Sfc = ExtractColumn(S, freeNodes, centerNode);
    Eigen::VectorXd Ofc = ExtractColumn(O, freeNodes, centerNode);

    Eigen::MatrixXd A = 2.0 * Off - v * v * dt * dt * Sff;

    Eigen::LDLT<Eigen::MatrixXd> solver(Off);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("LDLT factorization of free-node mass matrix failed.");
    }

    int nSteps = static_cast<int>(std::ceil(tMax / dt));

    Eigen::VectorXd cPrev = Eigen::VectorXd::Zero(nFree); // c^{n-1}
    Eigen::VectorXd cNow  = Eigen::VectorXd::Zero(nFree); // c^n
    Eigen::VectorXd cNext = Eigen::VectorXd::Zero(nFree);

    std::string energyFile = prefix + "_energy.txt";
    SaveEnergyHeader(energyFile);

    for (int n = 0; n <= nSteps; ++n) {
        double t = n * dt;

        double qPrev = std::sin(omega * (t - dt));
        double qNow  = std::sin(omega * t);
        double qNext = std::sin(omega * (t + dt));

        if (n % snapshotEvery == 0 || n == nSteps) {
            Eigen::VectorXd cFull = BuildFullVector(nNodes, freeNodes, cNow, centerNode, qNow);

            std::ostringstream name;
            name << prefix << "_snapshot_" << std::setw(5) << std::setfill('0') << n << ".txt";
            SaveSnapshot(mesh, cFull, name.str());

            double maxAbsPsi = cFull.cwiseAbs().maxCoeff();

            Eigen::VectorXd velocityFree;
            if (n == 0) {
                velocityFree = Eigen::VectorXd::Zero(nFree);
            } else {
                velocityFree = (cNow - cPrev) / dt;
            }

            AppendEnergy(
                    energyFile,
                    t,
                    velocityFree,
                    cNow,
                    Off,
                    Sff,
                    qNow,
                    maxAbsPsi,
                    v
            );
        }

        // Right-hand side for free nodes:
        //
        // Off c_f^{n+1} =
        //      (2Off - v^2 dt^2 Sff)c_f^n
        //      - Off c_f^{n-1}
        //      + forced terms from prescribed center coefficient:
        //
        //      - Ofc q^{n+1}
        //      + (2Ofc - v^2 dt^2 Sfc) q^n
        //      - Ofc q^{n-1}
        Eigen::VectorXd rhs =
                A * cNow
                - Off * cPrev
                - Ofc * qNext
                + (2.0 * Ofc - v * v * dt * dt * Sfc) * qNow
                - Ofc * qPrev;

        cNext = solver.solve(rhs);

        cPrev = cNow;
        cNow = cNext;
    }

    std::cout << "Finished omega = " << omega
              << "   output prefix: " << prefix << "\n";
}

int main() {
    try {
        const double L = 5.0;
        const double v = 1.0;

        // Same mesh choice as previous potential-well output section.
        const int Nwave = 10;

        Mesh mesh = GenerateBilinearMesh(Nwave, L);

        int centerNode = FindCentralNode(mesh);
        std::vector<int> freeNodes = GetFreeNodes(mesh, centerNode);

        std::cout << "Mesh: bilinear, N = " << Nwave << ", L = " << L << "\n";
        std::cout << "Number of nodes: " << mesh.p.size() << "\n";
        std::cout << "Number of free nodes: " << freeNodes.size() << "\n";
        std::cout << "Driven central node: " << centerNode + 1
                  << " at x = " << mesh.p[centerNode].x
                  << ", y = " << mesh.p[centerNode].y << "\n";

        auto Sloc = ComputeLocalStiffnessMatrix();
        auto Oloc = ComputeLocalOverlapMatrix(L / Nwave);

        Eigen::MatrixXd S, O;
        AssembleGlobalMatrices(mesh, Sloc, Oloc, S, O);

        Eigen::MatrixXd Sff = ExtractSubmatrix(S, freeNodes, freeNodes);
        Eigen::MatrixXd Off = ExtractSubmatrix(O, freeNodes, freeNodes);

        const double safety = 0.90;
        double dt = ComputeStableDt(Sff, Off, v, safety);

        std::cout << std::setprecision(12);
        std::cout << "Stable dt used = " << dt << "\n";

        {
            std::ofstream fout("stability_info.txt");
            fout << std::fixed << std::setprecision(12);
            fout << "# N L v safety dt\n";
            fout << Nwave << " " << L << " " << v << " " << safety << " " << dt << "\n";
        }

        // Driving frequencies required in the assignment.
        const double omega1 = M_PI / L;
        const double omega2 = 2.0 * M_PI / L;
        const double omega3 = M_PI / (2.0 * L);

        // Run long enough to see whether amplitude settles or keeps growing.
        // No damping is present, so a true stationary state is generally not expected.
        const double tMax = 200.0;

        // Save roughly 200 snapshots per run.
        const int nSteps = static_cast<int>(std::ceil(tMax / dt));
        const int snapshotEvery = std::max(1, nSteps / 200);

        RunForcedOscillation(
                mesh, S, O, freeNodes, centerNode,
                omega1, dt, tMax, snapshotEvery,
                "omega_pi_over_L",
                v
        );

        RunForcedOscillation(
                mesh, S, O, freeNodes, centerNode,
                omega2, dt, tMax, snapshotEvery,
                "omega_2pi_over_L",
                v
        );

        RunForcedOscillation(
                mesh, S, O, freeNodes, centerNode,
                omega3, dt, tMax, snapshotEvery,
                "omega_pi_over_2L",
                v
        );

        std::cout << "\nDone.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
