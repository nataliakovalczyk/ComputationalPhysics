#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <array>
#include <stdexcept>


// ============================================================
// Bilinear
// ============================================================


struct Node {
    double x, y;
};

struct Mesh {
    std::vector<Node> p;                 // global nodes
    std::vector<std::vector<int>> nlg;   // local-to-global map
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

double ExactPotential(double x, double y) {
    double xc = -1.0;
    double yc =  0.0;
    double r = std::sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc));
    return -(1.0 / (2.0 * M_PI)) * std::log(r);
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

void SaveMesh(const Mesh& mesh, const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Error: cannot open file " << filename << "\n";
        return;
    }

    fout << std::fixed << std::setprecision(12);

    for (size_t k = 0; k < mesh.nlg.size(); ++k) {
        for (int i = 0; i < 4; ++i) {
            int g = mesh.nlg[k][i];
            fout << std::setw(20) << double(k + 1)
                 << std::setw(20) << double(i + 1)
                 << std::setw(20) << double(g + 1)
                 << std::setw(20) << mesh.p[g].x
                 << std::setw(20) << mesh.p[g].y
                 << "\n";
        }
    }

    fout.close();
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

void PrintLocalStiffnessMatrix(const std::array<std::array<double, 4>, 4>& E) {
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Local stiffness matrix E:\n\n";

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << std::setw(16) << E[i][j];
        }
        std::cout << "\n";
    }
}

void SaveLocalStiffnessMatrix(const std::array<std::array<double, 4>, 4>& E,
                              const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Error: cannot open file " << filename << "\n";
        return;
    }

    fout << std::fixed << std::setprecision(12);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            fout << std::setw(16) << E[i][j];
        }
        fout << "\n";
    }
    fout.close();
}

void AssembleGlobalMatrix(
        const Mesh& mesh,
        const std::array<std::array<double, 4>, 4>& E,
        std::vector<std::vector<double>>& S
) {
    int nNodes = static_cast<int>(mesh.p.size());
    S.assign(nNodes, std::vector<double>(nNodes, 0.0));

    for (size_t k = 0; k < mesh.nlg.size(); ++k) {
        for (int i1 = 0; i1 < 4; ++i1) {
            for (int i2 = 0; i2 < 4; ++i2) {
                int g1 = mesh.nlg[k][i1];
                int g2 = mesh.nlg[k][i2];
                S[g1][g2] += E[i1][i2];
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

void ApplyBoundaryConditions(
        const Mesh& mesh,
        std::vector<std::vector<double>>& S,
        std::vector<double>& b
) {
    int nNodes = static_cast<int>(mesh.p.size());
    b.assign(nNodes, 0.0);

    for (int i = 0; i < nNodes; ++i) {
        if (IsBoundaryNode(mesh.p[i], mesh.L)) {
            for (int j = 0; j < nNodes; ++j) {
                S[i][j] = 0.0;
            }
            S[i][i] = 1.0;
            b[i] = ExactPotential(mesh.p[i].x, mesh.p[i].y);
        }
    }
}

std::vector<double> SolveLinearSystem(std::vector<std::vector<double>> A,
                                      std::vector<double> b) {
    int n = static_cast<int>(b.size());

    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double maxVal = std::fabs(A[k][k]);

        for (int i = k + 1; i < n; ++i) {
            if (std::fabs(A[i][k]) > maxVal) {
                maxVal = std::fabs(A[i][k]);
                pivot = i;
            }
        }

        if (maxVal < 1e-14) {
            throw std::runtime_error("Matrix is singular or nearly singular.");
        }

        if (pivot != k) {
            std::swap(A[k], A[pivot]);
            std::swap(b[k], b[pivot]);
        }

        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            A[i][k] = 0.0;
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }

    return x;
}

void PrintNodeComparison(const Mesh& mesh, const std::vector<double>& c) {
    std::cout << "\nNode comparison (FEM vs exact):\n";
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "node              x              y"
              << "         psi_FEM       psi_exact      difference\n";

    for (size_t i = 0; i < mesh.p.size(); ++i) {
        double exact = ExactPotential(mesh.p[i].x, mesh.p[i].y);
        double diff = c[i] - exact;

        std::cout << std::setw(4)  << (i + 1)
                  << std::setw(15) << mesh.p[i].x
                  << std::setw(15) << mesh.p[i].y
                  << std::setw(15) << c[i]
                  << std::setw(15) << exact
                  << std::setw(15) << diff
                  << "\n";
    }
}

void SaveNodeComparison(const Mesh& mesh,
                        const std::vector<double>& c,
                        const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Error: cannot open file " << filename << "\n";
        return;
    }

    fout << std::fixed << std::setprecision(12);
    fout << "# node x y psi_FEM psi_exact difference\n";

    for (size_t i = 0; i < mesh.p.size(); ++i) {
        double exact = ExactPotential(mesh.p[i].x, mesh.p[i].y);
        double diff = c[i] - exact;

        fout << std::setw(8)  << (i + 1)
             << std::setw(18) << mesh.p[i].x
             << std::setw(18) << mesh.p[i].y
             << std::setw(18) << c[i]
             << std::setw(18) << exact
             << std::setw(18) << diff
             << "\n";
    }

    fout.close();
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

void SaveElementScanData(const Mesh& mesh,
                         const std::vector<double>& c,
                         const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Error: cannot open file " << filename << "\n";
        return;
    }

    fout << std::fixed << std::setprecision(12);
    fout << "# elem xi1 xi2 x y psi_FEM psi_exact difference\n";

    const double step = 0.2;
    const int nSteps = 10;

    for (size_t k = 0; k < mesh.nlg.size(); ++k) {
        for (int i1 = 0; i1 <= nSteps; ++i1) {
            double xi1 = -1.0 + i1 * step;

            for (int i2 = 0; i2 <= nSteps; ++i2) {
                double xi2 = -1.0 + i2 * step;

                double x = 0.0;
                double y = 0.0;
                double psi_fem = 0.0;

                for (int a = 0; a < 4; ++a) {
                    int gnode = mesh.nlg[k][a];
                    double Na = g(a, xi1, xi2);

                    x += mesh.p[gnode].x * Na;
                    y += mesh.p[gnode].y * Na;
                    psi_fem += c[gnode] * Na;
                }

                double psi_exact = ExactPotential(x, y);
                double diff = psi_fem - psi_exact;

                fout << std::setw(8)  << (k + 1)
                     << std::setw(16) << xi1
                     << std::setw(16) << xi2
                     << std::setw(16) << x
                     << std::setw(16) << y
                     << std::setw(16) << psi_fem
                     << std::setw(16) << psi_exact
                     << std::setw(16) << diff
                     << "\n";
            }
        }
    }

    fout.close();
}

double FEMSolutionInElement(const Mesh& mesh,
                            const std::vector<double>& c,
                            int elem,
                            double xi1,
                            double xi2) {
    double psi = 0.0;
    for (int a = 0; a < 4; ++a) {
        int gnode = mesh.nlg[elem][a];
        psi += c[gnode] * g(a, xi1, xi2);
    }
    return psi;
}

void PhysicalCoordinatesInElement(const Mesh& mesh,
                                  int elem,
                                  double xi1,
                                  double xi2,
                                  double& x,
                                  double& y) {
    x = 0.0;
    y = 0.0;

    for (int a = 0; a < 4; ++a) {
        int gnode = mesh.nlg[elem][a];
        double Na = g(a, xi1, xi2);
        x += mesh.p[gnode].x * Na;
        y += mesh.p[gnode].y * Na;
    }
}

double ComputeMaxError(const Mesh& mesh, const std::vector<double>& c) {
    const double step = 0.2;
    const int nSteps = 10;

    double maxErr = 0.0;

    for (size_t elem = 0; elem < mesh.nlg.size(); ++elem) {
        for (int i = 0; i <= nSteps; ++i) {
            double xi1 = -1.0 + i * step;

            for (int j = 0; j <= nSteps; ++j) {
                double xi2 = -1.0 + j * step;

                double x, y;
                PhysicalCoordinatesInElement(mesh, static_cast<int>(elem), xi1, xi2, x, y);

                double psi_fem = FEMSolutionInElement(mesh, c, static_cast<int>(elem), xi1, xi2);
                double psi_exact = ExactPotential(x, y);

                double err = std::fabs(psi_fem - psi_exact);
                if (err > maxErr) {
                    maxErr = err;
                }
            }
        }
    }

    return maxErr;
}

std::vector<double> SolveFEMForMesh(const Mesh& mesh,
                                    const std::array<std::array<double, 4>, 4>& E) {
    std::vector<std::vector<double>> S;
    AssembleGlobalMatrix(mesh, E, S);

    std::vector<double> b;
    ApplyBoundaryConditions(mesh, S, b);

    return SolveLinearSystem(S, b);
}

void SaveMaxErrorVsN(const std::vector<int>& Nvalues, const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout) {
        std::cerr << "Error: cannot open file " << filename << "\n";
        return;
    }

    fout << std::fixed << std::setprecision(12);
    fout << "# N max_abs_error\n";

    auto E = ComputeLocalStiffnessMatrix();

    for (int N : Nvalues) {
        double L = 5.0;

        Mesh mesh = GenerateBilinearMesh(N, L);
        std::vector<double> c = SolveFEMForMesh(mesh, E);
        double maxErr = ComputeMaxError(mesh, c);

        fout << std::setw(8) << N
             << std::setw(20) << maxErr
             << "\n";

        std::cout << "N = " << N << "   max error = " << maxErr << "\n";
    }

    fout.close();
}

// ============================================================
// Biparabolic
// ============================================================

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

double GetMaxError(const Mesh& mesh, const std::vector<double>& c) {
    double maxErr = 0.0;
    for (int k = 0; k < mesh.nlg.size(); ++k) {
        for (double xi1 = -1.0; xi1 <= 1.0; xi1 += 0.2) {
            for (double xi2 = -1.0; xi2 <= 1.0; xi2 += 0.2) {
                double x = 0, y = 0, psi_fem = 0;

                for (int i = 0; i < 4; ++i) {
                    int gn = mesh.nlg[k][i];
                    double gi = g(i, xi1, xi2);
                    x += mesh.p[gn].x * gi;
                    y += mesh.p[gn].y * gi;
                }

                for (int i = 0; i < 9; ++i) {
                    int gn = mesh.nlg[k][i];
                    psi_fem += c[gn] * h(i, xi1, xi2);
                }
                double err = std::fabs(psi_fem - ExactPotential(x, y));
                if (err > maxErr) maxErr = err;
            }
        }
    }
    return maxErr;
}

void AssembleGlobalBiparabolic(const Mesh& mesh,
                               const std::vector<std::vector<double>>& E,
                               std::vector<std::vector<double>>& S) {
    int nNodes = mesh.p.size();
    S.assign(nNodes, std::vector<double>(nNodes, 0.0));
    for (size_t k = 0; k < mesh.nlg.size(); ++k) {
        for (int i1 = 0; i1 < 9; ++i1) { // 9 nodes
            for (int i2 = 0; i2 < 9; ++i2) {
                int g1 = mesh.nlg[k][i1];
                int g2 = mesh.nlg[k][i2];
                S[g1][g2] += E[i1][i2];
            }
        }
    }
}

void SaveElementScanDataBiparabolic(const Mesh& mesh, const std::vector<double>& c, const std::string& filename) {
    std::ofstream fout(filename);
    fout << std::fixed << std::setprecision(12);
    fout << "# elem xi1 xi2 x y psi_FEM psi_exact difference\n";

    for (size_t k = 0; k < mesh.nlg.size(); ++k) {
        for (double xi1 = -1.0; xi1 <= 1.01; xi1 += 0.2) {
            for (double xi2 = -1.0; xi2 <= 1.01; xi2 += 0.2) {
                double x = 0.0, y = 0.0, psi_fem = 0.0;

                for (int a = 0; a < 4; ++a) {
                    int gnode = mesh.nlg[k][a];
                    double Na = g(a, xi1, xi2);
                    x += mesh.p[gnode].x * Na;
                    y += mesh.p[gnode].y * Na;
                }

                for (int a = 0; a < 9; ++a) {
                    int gnode = mesh.nlg[k][a];
                    psi_fem += c[gnode] * h(a, xi1, xi2);
                }

                double psi_exact = ExactPotential(x, y);
                fout << k+1 << " " << xi1 << " " << xi2 << " " << x << " " << y
                     << " " << psi_fem << " " << psi_exact << " " << (psi_fem - psi_exact) << "\n";
            }
        }
    }
}

void SaveMaxErrorVsNBiparabolic(const std::vector<int>& Nvalues, const std::string& filename) {
    std::ofstream fout(filename);
    auto E = ComputeLocalStiffnessMatrixBiparabolic(); // 9x9 matrix

    for (int N : Nvalues) {
        Mesh mesh = GenerateBiparabolicMesh(N, 5.0);

        // Assemble 9x9
        std::vector<std::vector<double>> S;
        AssembleGlobalBiparabolic(mesh, E, S);

        std::vector<double> b;
        ApplyBoundaryConditions(mesh, S, b);

        std::vector<double> c = SolveLinearSystem(S, b);
        double maxErr = GetMaxError(mesh, c);

        fout << N << " " << maxErr << "\n";
        std::cout << "N = " << N << " (Biparabolic) Max Error = " << maxErr << "\n";
    }
}

int main() {
    int N = 4;
    double L = 5.0;

    try {
        Mesh mesh = GenerateBilinearMesh(N, L);

        SaveMesh(mesh, "bilinear_N4.txt");

        auto E = ComputeLocalStiffnessMatrix();
        PrintLocalStiffnessMatrix(E);
        SaveLocalStiffnessMatrix(E, "local_stiffness_matrix.txt");

        std::vector<std::vector<double>> S;
        AssembleGlobalMatrix(mesh, E, S);

        std::vector<double> b;
        ApplyBoundaryConditions(mesh, S, b);

        std::vector<double> c = SolveLinearSystem(S, b);

        PrintNodeComparison(mesh, c);
        SaveNodeComparison(mesh, c, "DI3_nodes_comparison.txt");

        SaveElementScanData(mesh, c, "DI4_element_scan.txt");

        std::vector<int> Nvalues = {2, 3, 4, 5, 6, 8, 10};
        SaveMaxErrorVsN(Nvalues, "DI5_max_error_vs_N.txt");


        Mesh mesh_para = GenerateBiparabolicMesh(N, L);
        auto E_para = ComputeLocalStiffnessMatrixBiparabolic();

        std::vector<std::vector<double>> S_para;
        AssembleGlobalBiparabolic(mesh_para, E_para, S_para);

        std::vector<double> b_para;
        ApplyBoundaryConditions(mesh_para, S_para, b_para);

        std::vector<double> c_para = SolveLinearSystem(S_para, b_para);

        SaveElementScanDataBiparabolic(mesh_para, c_para, "DII4_element_scan.txt");
        SaveMaxErrorVsNBiparabolic(Nvalues, "DII5_max_error_vs_N.txt");

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}