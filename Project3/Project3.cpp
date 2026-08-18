#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <array>
#include <stdexcept>
#include <Eigen/Eigenvalues>

struct Node {
    double x, y;
};

struct Mesh {
    std::vector<Node> p;
    std::vector<std::vector<int>> nlg;
    int N;
    double L;
};

double f1(double xi) { return 0.5 * (1.0 - xi); }
double f2(double xi) { return 0.5 * (1.0 + xi); }

double df1(double) { return -0.5; }
double df2(double) { return  0.5; }


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

double g(int i, double xi1, double xi2) {
    switch (i) {
        case 0: return f1(xi1) * f1(xi2); // local node 1
        case 1: return f2(xi1) * f1(xi2); // local node 2
        case 2: return f1(xi1) * f2(xi2); // local node 3
        case 3: return f2(xi1) * f2(xi2); // local node 4
        default: return 0.0;
    }
}

double q1(double xi) { return xi * (xi - 1.0) / 2.0; }
double q2(double xi) { return (1.0 - xi) * (1.0 + xi); }
double q3(double xi) { return xi * (xi + 1.0) / 2.0; }

double dq1(double xi) { return xi - 0.5; }
double dq2(double xi) { return -2.0 * xi; }
double dq3(double xi) { return xi + 0.5; }

double dh_dxi1(int i, double xi1, double xi2) {
    switch (i) {
        case 0: return dq1(xi1) * q1(xi2);
        case 1: return dq3(xi1) * q1(xi2);
        case 2: return dq1(xi1) * q3(xi2);
        case 3: return dq3(xi1) * q3(xi2);
        case 4: return dq2(xi1) * q1(xi2);
        case 5: return dq3(xi1) * q2(xi2);
        case 6: return dq1(xi1) * q2(xi2);
        case 7: return dq2(xi1) * q3(xi2);
        case 8: return dq2(xi1) * q2(xi2);
        default: return 0.0;
    }
}

double dh_dxi2(int i, double xi1, double xi2) {
    switch (i) {
        case 0: return q1(xi1) * dq1(xi2);
        case 1: return q3(xi1) * dq1(xi2);
        case 2: return q1(xi1) * dq3(xi2);
        case 3: return q3(xi1) * dq3(xi2);
        case 4: return q2(xi1) * dq1(xi2);
        case 5: return q3(xi1) * dq2(xi2);
        case 6: return q1(xi1) * dq2(xi2);
        case 7: return q2(xi1) * dq3(xi2);
        case 8: return q2(xi1) * dq2(xi2);
        default: return 0.0;
    }
}

double h(int i, double xi1, double xi2) {
    switch (i) {
        case 0: return q1(xi1) * q1(xi2); // (0,0)  -> xi=(-1,-1)
        case 1: return q3(xi1) * q1(xi2); // (a,0)  -> xi=( 1,-1)
        case 2: return q1(xi1) * q3(xi2); // (0,a)  -> xi=(-1, 1)
        case 3: return q3(xi1) * q3(xi2); // (a,a)  -> xi=( 1, 1)
        case 4: return q2(xi1) * q1(xi2); // (a/2,0)
        case 5: return q3(xi1) * q2(xi2); // (a,a/2)
        case 6: return q1(xi1) * q2(xi2); // (0,a/2)
        case 7: return q2(xi1) * q3(xi2); // (a/2,a)
        case 8: return q2(xi1) * q2(xi2); // center
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

std::array<std::array<double, 4>, 4> ComputeLocalStiffnessMatrix() {
    std::array<std::array<double, 4>, 4> E{};

    double sqrt30 = std::sqrt(30.0);

    std::array<double, 4> w = {
            (18.0 + sqrt30) / 36.0,
            (18.0 + sqrt30) / 36.0,
            (18.0 - sqrt30) / 36.0,
            (18.0 - sqrt30) / 36.0
    };

    std::array<double, 4> gamma = {
            -std::sqrt(3.0/7.0 - 2.0/7.0 * std::sqrt(6.0/5.0)),
            std::sqrt(3.0/7.0 - 2.0/7.0 * std::sqrt(6.0/5.0)),
            std::sqrt(3.0/7.0 + 2.0/7.0 * std::sqrt(6.0/5.0)),
            -std::sqrt(3.0/7.0 + 2.0/7.0 * std::sqrt(6.0/5.0))
    };

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

            E[j][i] = sum;
        }
    }

    return E;
}

std::array<std::array<double, 4>, 4>
ComputeLocalOverlapMatrixBilinear(double a) {
    std::array<std::array<double, 4>, 4> O{};

    double sqrt30 = std::sqrt(30.0);

    std::array<double, 4> w = {
            (18.0 + sqrt30) / 36.0,
            (18.0 + sqrt30) / 36.0,
            (18.0 - sqrt30) / 36.0,
            (18.0 - sqrt30) / 36.0
    };

    std::array<double, 4> gamma = {
            -std::sqrt(3.0/7.0 - 2.0/7.0 * std::sqrt(6.0/5.0)),
            std::sqrt(3.0/7.0 - 2.0/7.0 * std::sqrt(6.0/5.0)),
            std::sqrt(3.0/7.0 + 2.0/7.0 * std::sqrt(6.0/5.0)),
            -std::sqrt(3.0/7.0 + 2.0/7.0 * std::sqrt(6.0/5.0))
    };

    double factor = (a * a) / 4.0;

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
            O[j][i] = factor * sum;
        }
    }

    return O;
}

void AssembleGlobalMatricesBilinear(
        const Mesh& mesh,
        const std::array<std::array<double, 4>, 4>& Hloc,
        const std::array<std::array<double, 4>, 4>& Oloc,
        std::vector<std::vector<double>>& H,
        std::vector<std::vector<double>>& O
) {
    int nNodes = static_cast<int>(mesh.p.size());
    H.assign(nNodes, std::vector<double>(nNodes, 0.0));
    O.assign(nNodes, std::vector<double>(nNodes, 0.0));

    for (size_t e = 0; e < mesh.nlg.size(); ++e) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int gi = mesh.nlg[e][i];
                int gj = mesh.nlg[e][j];
                H[gi][gj] += Hloc[i][j];
                O[gi][gj] += Oloc[i][j];
            }
        }
    }
}

bool IsBoundaryNode(const Node& nd, double L, double eps = 1e-12) {
    return (std::fabs(nd.x) < eps ||
            std::fabs(nd.x - L) < eps ||
            std::fabs(nd.y) < eps ||
            std::fabs(nd.y - L) < eps);
}

void ApplyBoundaryConditionsEigenproblem(
        const Mesh& mesh,
        std::vector<std::vector<double>>& H,
        std::vector<std::vector<double>>& O
) {
    int nNodes = static_cast<int>(mesh.p.size());

    for (int i = 0; i < nNodes; ++i) {
        if (IsBoundaryNode(mesh.p[i], mesh.L)) {
            for (int j = 0; j < nNodes; ++j) {
                H[i][j] = 0.0;
                H[j][i] = 0.0;
                O[i][j] = 0.0;
                O[j][i] = 0.0;
            }
            H[i][i] = -1.0;
            O[i][i] = 1.0;
        }
    }
}

std::vector<std::vector<double>> ComputeLocalStiffnessMatrixBiparabolic() {
    std::vector<std::vector<double>> E(9, std::vector<double>(9, 0.0));
    double sqrt30 = std::sqrt(30.0);

    std::array<double, 4> w = {
            (18.0 + sqrt30) / 36.0,
            (18.0 + sqrt30) / 36.0,
            (18.0 - sqrt30) / 36.0,
            (18.0 - sqrt30) / 36.0
    };

    std::array<double, 4> gamma = {
            -std::sqrt(3.0/7.0 - 2.0/7.0 * std::sqrt(6.0/5.0)),
            std::sqrt(3.0/7.0 - 2.0/7.0 * std::sqrt(6.0/5.0)),
            std::sqrt(3.0/7.0 + 2.0/7.0 * std::sqrt(6.0/5.0)),
            -std::sqrt(3.0/7.0 + 2.0/7.0 * std::sqrt(6.0/5.0))
    };

    for (int j = 0; j < 9; ++j) {
        for (int i = 0; i < 9; ++i) {
            double sum = 0.0;
            for (int l = 0; l < 4; ++l) {
                for (int n = 0; n < 4; ++n) {
                    double xi1 = gamma[l], xi2 = gamma[n];
                    double term = dh_dxi1(j, xi1, xi2) * dh_dxi1(i, xi1, xi2) +
                                  dh_dxi2(j, xi1, xi2) * dh_dxi2(i, xi1, xi2);
                    sum += w[l] * w[n] * term;
                }
            }
            E[j][i] = sum;
        }
    }
    return E;
}

Mesh GenerateBiparabolicMesh(int N, double L) {
    double a = L / N;
    double tol = 1e-8;
    Mesh mesh;
    mesh.N = N; mesh.L = L;
    mesh.nlg.assign(N * N, std::vector<int>(9, -1));

    double rb[9][2] = {
            {0,0}, {a,0}, {0,a}, {a,a},
            {a/2.0, 0}, {a, a/2.0}, {0, a/2.0},
            {a/2.0, a}, {a/2.0, a/2.0}
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
        for (int inw = 1; inw < 9; ++inw) {
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

std::vector<std::vector<double>>
ComputeLocalOverlapMatrixBiparabolic(double a) {
    std::vector<std::vector<double>> O(9, std::vector<double>(9, 0.0));

    double sqrt30 = std::sqrt(30.0);

    std::array<double, 4> w = {
            (18.0 + sqrt30) / 36.0,
            (18.0 + sqrt30) / 36.0,
            (18.0 - sqrt30) / 36.0,
            (18.0 - sqrt30) / 36.0
    };

    std::array<double, 4> gamma = {
            -std::sqrt(3.0/7.0 - 2.0/7.0 * std::sqrt(6.0/5.0)),
            std::sqrt(3.0/7.0 - 2.0/7.0 * std::sqrt(6.0/5.0)),
            std::sqrt(3.0/7.0 + 2.0/7.0 * std::sqrt(6.0/5.0)),
            -std::sqrt(3.0/7.0 + 2.0/7.0 * std::sqrt(6.0/5.0))
    };

    double factor = (a * a) / 4.0;

    for (int j = 0; j < 9; ++j) {
        for (int i = 0; i < 9; ++i) {
            double sum = 0.0;
            for (int l = 0; l < 4; ++l) {
                for (int n = 0; n < 4; ++n) {
                    double xi1 = gamma[l];
                    double xi2 = gamma[n];
                    sum += w[l] * w[n] * h(i, xi1, xi2) * h(j, xi1, xi2);
                }
            }
            O[j][i] = factor * sum;
        }
    }

    return O;
}

void AssembleGlobalMatricesBiparabolic(
        const Mesh& mesh,
        const std::vector<std::vector<double>>& Hloc,
        const std::vector<std::vector<double>>& Oloc,
        std::vector<std::vector<double>>& H,
        std::vector<std::vector<double>>& O
) {
    int nNodes = static_cast<int>(mesh.p.size());
    H.assign(nNodes, std::vector<double>(nNodes, 0.0));
    O.assign(nNodes, std::vector<double>(nNodes, 0.0));

    for (size_t e = 0; e < mesh.nlg.size(); ++e) {
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                int gi = mesh.nlg[e][i];
                int gj = mesh.nlg[e][j];
                H[gi][gj] += Hloc[i][j];
                O[gi][gj] += Oloc[i][j];
            }
        }
    }
}

double ExactEnergy(int nx, int ny, double L) {
    return 0.5 * (nx*nx + ny*ny) * M_PI * M_PI / (L*L);
}

struct ExactState {
    int nx;
    int ny;
    double E;
};

Eigen::MatrixXd ToEigenMatrix(const std::vector<std::vector<double>>& A) {
    int n = static_cast<int>(A.size());
    Eigen::MatrixXd M(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            M(i, j) = A[i][j];
        }
    }
    return M;
}

void SolveGeneralizedEigenproblem(
        const std::vector<std::vector<double>>& H,
        const std::vector<std::vector<double>>& O,
        Eigen::VectorXd& evals,
        Eigen::MatrixXd& evecs
) {
    Eigen::MatrixXd Hmat = ToEigenMatrix(H);
    Eigen::MatrixXd Omat = ToEigenMatrix(O);

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(Hmat, Omat);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Generalized eigenvalue solver failed.");
    }

    evals = solver.eigenvalues();
    evecs = solver.eigenvectors();
}

std::vector<ExactState> GenerateExactStates(double L, int nmax = 10) {
    std::vector<ExactState> states;
    for (int nx = 1; nx <= nmax; ++nx) {
        for (int ny = 1; ny <= nmax; ++ny) {
            states.push_back({nx, ny, ExactEnergy(nx, ny, L)});
        }
    }

    std::sort(states.begin(), states.end(),
              [](const ExactState& a, const ExactState& b) {
                  if (std::fabs(a.E - b.E) > 1e-12) return a.E < b.E;
                  if (a.nx != b.nx) return a.nx < b.nx;
                  return a.ny < b.ny;
              });

    return states;
}

std::vector<int> GetPositiveEigenIndices(const Eigen::VectorXd& evals, int howMany = 10) {
    std::vector<int> ids;
    for (int i = 0; i < evals.size(); ++i) {
        if (evals(i) > 1e-8) {
            ids.push_back(i);
            if (static_cast<int>(ids.size()) == howMany) break;
        }
    }
    return ids;
}

void PrintPositiveSpectrum(const Eigen::VectorXd& evals, int howMany = 10) {
    std::cout << "\nPositive eigenvalues:\n";
    int count = 0;
    for (int i = 0; i < evals.size() && count < howMany; ++i) {
        if (evals(i) > 1e-8) {
            std::cout << count + 1 << "   " << std::setprecision(12) << evals(i) << "\n";
            count++;
        }
    }
}

void SaveSpectrumWithExact(const Eigen::VectorXd& evals,
                           double L,
                           const std::string& filename,
                           int howMany = 10) {
    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Cannot open " + filename);
    }

    auto exact = GenerateExactStates(L, 10);
    auto ids = GetPositiveEigenIndices(evals, howMany);

    fout << std::fixed << std::setprecision(12);
    fout << "# state nx ny E_numerical E_exact abs_error\n";

    for (int k = 0; k < static_cast<int>(ids.size()); ++k) {
        int idx = ids[k];
        fout << std::setw(6)  << k + 1
             << std::setw(6)  << exact[k].nx
             << std::setw(6)  << exact[k].ny
             << std::setw(20) << evals(idx)
             << std::setw(20) << exact[k].E
             << std::setw(20) << std::fabs(evals(idx) - exact[k].E)
             << "\n";
    }
}

void AppendSpectrumVsN(const Eigen::VectorXd& evals,
                       double L,
                       int N,
                       const std::string& basisName,
                       const std::string& filename,
                       int howMany = 10) {
    std::ofstream fout(filename, std::ios::app);
    if (!fout) {
        throw std::runtime_error("Cannot open " + filename);
    }

    auto exact = GenerateExactStates(L, 10);
    auto ids = GetPositiveEigenIndices(evals, howMany);

    fout << std::fixed << std::setprecision(12);

    for (int k = 0; k < static_cast<int>(ids.size()); ++k) {
        int idx = ids[k];
        fout << basisName << " "
             << N << " "
             << k + 1 << " "
             << exact[k].nx << " "
             << exact[k].ny << " "
             << evals(idx) << " "
             << exact[k].E << " "
             << std::fabs(evals(idx) - exact[k].E)
             << "\n";
    }
}

void WriteSpectrumVsNHeader(const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Cannot open " + filename);
    }
    fout << "# basis N state nx ny E_numerical E_exact abs_error\n";
}

void SaveEigenvectorAtNodes(const Mesh& mesh,
                            const Eigen::VectorXd& psi,
                            const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Cannot open " + filename);
    }

    fout << std::fixed << std::setprecision(12);
    fout << "# node x y psi\n";

    for (size_t i = 0; i < mesh.p.size(); ++i) {
        fout << std::setw(8)  << i + 1
             << std::setw(18) << mesh.p[i].x
             << std::setw(18) << mesh.p[i].y
             << std::setw(18) << psi(static_cast<int>(i))
             << "\n";
    }
}

void SaveSelectedEigenstates(const Mesh& mesh,
                             const Eigen::VectorXd& evals,
                             const Eigen::MatrixXd& evecs,
                             const std::string& prefix,
                             int howMany = 4) {
    auto ids = GetPositiveEigenIndices(evals, howMany);

    for (int k = 0; k < static_cast<int>(ids.size()); ++k) {
        std::ostringstream name;
        name << prefix << "_state_" << (k + 1) << ".txt";
        SaveEigenvectorAtNodes(mesh, evecs.col(ids[k]), name.str());
    }
}

void RunBilinearCase(int N, double L,
                     Eigen::VectorXd& evals,
                     Eigen::MatrixXd& evecs) {
    Mesh mesh = GenerateBilinearMesh(N, L);

    auto Kloc = ComputeLocalStiffnessMatrix();

    std::array<std::array<double, 4>, 4> Hloc{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Hloc[i][j] = 0.5 * Kloc[i][j];
        }
    }

    auto Oloc = ComputeLocalOverlapMatrixBilinear(L / N);

    std::vector<std::vector<double>> H, O;
    AssembleGlobalMatricesBilinear(mesh, Hloc, Oloc, H, O);
    ApplyBoundaryConditionsEigenproblem(mesh, H, O);

    SolveGeneralizedEigenproblem(H, O, evals, evecs);
}

void RunBiparabolicCase(int N, double L,
                        Eigen::VectorXd& evals,
                        Eigen::MatrixXd& evecs) {
    Mesh mesh = GenerateBiparabolicMesh(N, L);

    auto Kloc = ComputeLocalStiffnessMatrixBiparabolic();

    std::vector<std::vector<double>> Hloc(9, std::vector<double>(9, 0.0));
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            Hloc[i][j] = 0.5 * Kloc[i][j];
        }
    }

    auto Oloc = ComputeLocalOverlapMatrixBiparabolic(L / N);

    std::vector<std::vector<double>> H, O;
    AssembleGlobalMatricesBiparabolic(mesh, Hloc, Oloc, H, O);
    ApplyBoundaryConditionsEigenproblem(mesh, H, O);

    SolveGeneralizedEigenproblem(H, O, evals, evecs);
}

int main() {
    double L = 5.0;
    std::vector<int> Nvalues = {2, 3, 4, 5, 6, 8, 10};
    int Nwave = 10;

    try {
        WriteSpectrumVsNHeader("bilinear_spectrum_vs_N.txt");
        WriteSpectrumVsNHeader("biparabolic_spectrum_vs_N.txt");

        for (int N : Nvalues) {
            Eigen::VectorXd evalsBilin, evalsBiPara;
            Eigen::MatrixXd evecsDummy1, evecsDummy2;

            RunBilinearCase(N, L, evalsBilin, evecsDummy1);
            AppendSpectrumVsN(evalsBilin, L, N, "bilinear", "bilinear_spectrum_vs_N.txt", 10);

            RunBiparabolicCase(N, L, evalsBiPara, evecsDummy2);
            AppendSpectrumVsN(evalsBiPara, L, N, "biparabolic", "biparabolic_spectrum_vs_N.txt", 10);

            std::cout << "Finished N = " << N << "\n";
        }

        {
            Eigen::VectorXd evals;
            Eigen::MatrixXd evecs;

            RunBilinearCase(Nwave, L, evals, evecs);
            PrintPositiveSpectrum(evals, 10);
            SaveSpectrumWithExact(evals, L, "bilinear_spectrum_N" + std::to_string(Nwave) + ".txt", 10);

            Mesh mesh = GenerateBilinearMesh(Nwave, L);
            SaveSelectedEigenstates(mesh, evals, evecs,
                                    "bilinear_N" + std::to_string(Nwave), 4);
        }

        {
            Eigen::VectorXd evals;
            Eigen::MatrixXd evecs;

            RunBiparabolicCase(Nwave, L, evals, evecs);
            PrintPositiveSpectrum(evals, 10);
            SaveSpectrumWithExact(evals, L, "biparabolic_spectrum_N" + std::to_string(Nwave) + ".txt", 10);

            Mesh mesh = GenerateBiparabolicMesh(Nwave, L);
            SaveSelectedEigenstates(mesh, evals, evecs,
                                    "biparabolic_N" + std::to_string(Nwave), 4);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}