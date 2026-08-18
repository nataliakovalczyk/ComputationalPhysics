#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <array>
#include <stdexcept>
#include <complex>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

using Complex = std::complex<double>;

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

    Mesh mesh;
    mesh.N = N;
    mesh.L = L;
    mesh.nlg.assign(N * N, std::vector<int>(4, -1));

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
        for (int inw = 1; inw < 4; ++inw) {
            double xp = mesh.p[inr].x + rb[inw][0];
            double yp = mesh.p[inr].y + rb[inw][1];

            int ibyl = -1;

            for (int ie = 0; ie < inumer; ++ie) {
                double dr = std::fabs(xp - mesh.p[ie].x)
                            + std::fabs(yp - mesh.p[ie].y);

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

std::array<std::array<double, 4>, 4> ComputeLocalKineticMatrix() {
    std::array<std::array<double, 4>, 4> K{};

    auto w = GaussWeights4();
    auto p = GaussPoints4();

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            double sum = 0.0;

            for (int l = 0; l < 4; ++l) {
                for (int n = 0; n < 4; ++n) {
                    double xi1 = p[l];
                    double xi2 = p[n];

                    double term =
                            dg_dxi1(j, xi1, xi2) * dg_dxi1(i, xi1, xi2)
                            + dg_dxi2(j, xi1, xi2) * dg_dxi2(i, xi1, xi2);

                    sum += w[l] * w[n] * term;
                }
            }

            K[j][i] = 0.5 * sum;
        }
    }

    return K;
}

std::array<std::array<double, 4>, 4> ComputeLocalOverlapMatrix(double a) {
    std::array<std::array<double, 4>, 4> O{};

    auto w = GaussWeights4();
    auto p = GaussPoints4();

    double factor = a * a / 4.0;

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            double sum = 0.0;

            for (int l = 0; l < 4; ++l) {
                for (int n = 0; n < 4; ++n) {
                    double xi1 = p[l];
                    double xi2 = p[n];

                    sum += w[l] * w[n]
                           * g(j, xi1, xi2)
                           * g(i, xi1, xi2);
                }
            }

            O[j][i] = factor * sum;
        }
    }

    return O;
}

std::array<std::array<double, 4>, 4>
ComputeLocalElectricMatrix(const Mesh& mesh, int elem, double eF) {
    std::array<std::array<double, 4>, 4> V{};

    double a = mesh.L / mesh.N;
    double factor = a * a / 4.0;

    auto w = GaussWeights4();
    auto p = GaussPoints4();

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            double sum = 0.0;

            for (int l = 0; l < 4; ++l) {
                for (int n = 0; n < 4; ++n) {
                    double xi1 = p[l];
                    double xi2 = p[n];

                    double x = 0.0;

                    for (int r = 0; r < 4; ++r) {
                        int globalNode = mesh.nlg[elem][r];
                        x += mesh.p[globalNode].x * g(r, xi1, xi2);
                    }

                    sum += w[l] * w[n]
                           * eF * x
                           * g(j, xi1, xi2)
                           * g(i, xi1, xi2);
                }
            }

            V[j][i] = factor * sum;
        }
    }

    return V;
}

Eigen::MatrixXd ToEigenMatrix(const std::vector<std::vector<double>>& A) {
    int n = static_cast<int>(A.size());
    Eigen::MatrixXd M(n, n);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M(i, j) = A[i][j];

    return M;
}

void AssembleMatrices(
        const Mesh& mesh,
        double eF,
        Eigen::MatrixXd& H,
        Eigen::MatrixXd& O,
        Eigen::MatrixXd& V
) {
    int nNodes = static_cast<int>(mesh.p.size());
    double a = mesh.L / mesh.N;

    std::vector<std::vector<double>> Hv(nNodes, std::vector<double>(nNodes, 0.0));
    std::vector<std::vector<double>> Ov(nNodes, std::vector<double>(nNodes, 0.0));
    std::vector<std::vector<double>> Vv(nNodes, std::vector<double>(nNodes, 0.0));

    auto Hloc = ComputeLocalKineticMatrix();
    auto Oloc = ComputeLocalOverlapMatrix(a);

    for (size_t e = 0; e < mesh.nlg.size(); ++e) {
        auto Vloc = ComputeLocalElectricMatrix(mesh, static_cast<int>(e), eF);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int gi = mesh.nlg[e][i];
                int gj = mesh.nlg[e][j];

                Hv[gi][gj] += Hloc[i][j];
                Ov[gi][gj] += Oloc[i][j];
                Vv[gi][gj] += Vloc[i][j];
            }
        }
    }

    for (int i = 0; i < nNodes; ++i) {
        if (IsBoundaryNode(mesh.p[i], mesh.L)) {
            for (int j = 0; j < nNodes; ++j) {
                Hv[i][j] = Hv[j][i] = 0.0;
                Ov[i][j] = Ov[j][i] = 0.0;
                Vv[i][j] = Vv[j][i] = 0.0;
            }

            Hv[i][i] = -1.0;
            Ov[i][i] = 1.0;
            Vv[i][i] = 0.0;
        }
    }

    H = ToEigenMatrix(Hv);
    O = ToEigenMatrix(Ov);
    V = ToEigenMatrix(Vv);
}

std::vector<int> GetPositiveEigenIndices(const Eigen::VectorXd& evals, int howMany) {
    std::vector<int> ids;

    for (int i = 0; i < evals.size(); ++i) {
        if (evals(i) > 1e-8) {
            ids.push_back(i);
            if (static_cast<int>(ids.size()) == howMany)
                break;
        }
    }

    return ids;
}

double NormO(
        const Eigen::VectorXcd& c,
        const Eigen::MatrixXcd& Oc
) {
    Complex value = c.adjoint() * Oc * c;
    return value.real();
}

Complex ProjectionO(
        const Eigen::VectorXd& eigenstate,
        const Eigen::VectorXcd& c,
        const Eigen::MatrixXcd& Oc
) {
    Eigen::VectorXcd e = eigenstate.cast<Complex>();
    return e.adjoint() * Oc * c;
}

void RunTimeEvolution(
        const std::string& filename,
        const Eigen::MatrixXd& H0real,
        const Eigen::MatrixXd& Oreal,
        const Eigen::MatrixXd& Vreal,
        const Eigen::VectorXd& evals,
        const Eigen::MatrixXd& evecs,
        const std::vector<int>& positiveIds,
        double omega,
        double dt,
        double tMax,
        double hbar,
        int trackedStates
) {
    int n = H0real.rows();

    Eigen::MatrixXcd H0 = H0real.cast<Complex>();
    Eigen::MatrixXcd O  = Oreal.cast<Complex>();
    Eigen::MatrixXcd V  = Vreal.cast<Complex>();

    Eigen::VectorXcd c = evecs.col(positiveIds[0]).cast<Complex>();

    std::ofstream fout(filename);
    fout << std::setprecision(15);

    fout << "t,norm";
    for (int k = 0; k < trackedStates; ++k)
        fout << ",P" << k;
    fout << ",leakage_above_first_excited\n";

    int steps = static_cast<int>(tMax / dt);

    for (int step = 0; step <= steps; ++step) {
        double t = step * dt;

        double norm = NormO(c, O);

        fout << t << "," << norm;

        double lowSum = 0.0;

        for (int k = 0; k < trackedStates; ++k) {
            Complex p = ProjectionO(evecs.col(positiveIds[k]), c, O);
            double prob = std::norm(p);

            fout << "," << prob;

            if (k <= 2)
                lowSum += prob;
        }

        double leakage = 1.0 - lowSum;
        fout << "," << leakage << "\n";

        if (step == steps)
            break;

        double tNext = t + dt;

        Eigen::MatrixXcd Ht     = H0 + V * std::sin(omega * t);
        Eigen::MatrixXcd HtNext = H0 + V * std::sin(omega * tNext);

        Complex I(0.0, 1.0);

        Eigen::MatrixXcd A = O + I * dt / (2.0 * hbar) * HtNext;
        Eigen::MatrixXcd B = O - I * dt / (2.0 * hbar) * Ht;

        Eigen::VectorXcd rhs = B * c;

        c = A.partialPivLu().solve(rhs);
    }

    std::cout << "Saved: " << filename << "\n";
}

double MaxLeakageDuringEvolution(
        const Eigen::MatrixXd& H0real,
        const Eigen::MatrixXd& Oreal,
        const Eigen::MatrixXd& Vreal,
        const Eigen::MatrixXd& evecs,
        const std::vector<int>& positiveIds,
        double omega,
        double dt,
        double tMax,
        double hbar
) {
    Eigen::MatrixXcd H0 = H0real.cast<Complex>();
    Eigen::MatrixXcd O  = Oreal.cast<Complex>();
    Eigen::MatrixXcd V  = Vreal.cast<Complex>();

    Eigen::VectorXcd c = evecs.col(positiveIds[0]).cast<Complex>();

    int steps = static_cast<int>(tMax / dt);
    double maxLeakage = 0.0;

    for (int step = 0; step <= steps; ++step) {
        double lowSum = 0.0;

        for (int k = 0; k <= 2; ++k) {
            Complex p = ProjectionO(evecs.col(positiveIds[k]), c, O);
            lowSum += std::norm(p);
        }

        double leakage = 1.0 - lowSum;
        if (leakage > maxLeakage)
            maxLeakage = leakage;

        if (step == steps)
            break;

        double t = step * dt;
        double tNext = t + dt;

        Eigen::MatrixXcd Ht     = H0 + V * std::sin(omega * t);
        Eigen::MatrixXcd HtNext = H0 + V * std::sin(omega * tNext);

        Complex I(0.0, 1.0);

        Eigen::MatrixXcd A = O + I * dt / (2.0 * hbar) * HtNext;
        Eigen::MatrixXcd B = O - I * dt / (2.0 * hbar) * Ht;

        c = A.partialPivLu().solve(B * c);
    }

    return maxLeakage;
}

int main() {
    double L = 5.0;
    int Nwave = 10;

    double hbar = 1.0;
    double dt = 0.25;
    double tMax = 300.0;

    int trackedStates = 10;

    try {
        Mesh mesh = GenerateBilinearMesh(Nwave, L);

        double eF = 1.0 / L;

        Eigen::MatrixXd H0, O, V;
        AssembleMatrices(mesh, eF, H0, O, V);

        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H0, O);

        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Generalized eigenproblem failed.");

        Eigen::VectorXd evals = solver.eigenvalues();
        Eigen::MatrixXd evecs = solver.eigenvectors();

        std::vector<int> ids = GetPositiveEigenIndices(evals, trackedStates);

        std::cout << "\nLowest positive eigenvalues:\n";
        for (int k = 0; k < trackedStates; ++k) {
            std::cout << k << "   E = "
                      << std::setprecision(12)
                      << evals(ids[k]) << "\n";
        }

        double E0 = evals(ids[0]);
        double E1 = evals(ids[1]);
        double omegaRes = (E1 - E0) / hbar;
        double omegaHalf = 0.5 * omegaRes;

        std::cout << "\nResonant omega = " << omegaRes << "\n";
        std::cout << "Half omega     = " << omegaHalf << "\n\n";

        RunTimeEvolution(
                "resonant_probabilities.csv",
                H0, O, V,
                evals, evecs, ids,
                omegaRes,
                dt, tMax, hbar,
                trackedStates
        );

        RunTimeEvolution(
                "half_resonant_probabilities.csv",
                H0, O, V,
                evals, evecs, ids,
                omegaHalf,
                dt, tMax, hbar,
                trackedStates
        );

        std::ofstream leakFile("leakage_vs_amplitude.csv");
        leakFile << "eF,max_leakage\n";

        std::vector<double> amplitudes = {
                0.25 / L,
                0.5  / L,
                1.0  / L,
                2.0  / L,
                4.0  / L
        };

        for (double amp : amplitudes) {
            Eigen::MatrixXd Htmp, Otmp, Vtmp;
            AssembleMatrices(mesh, amp, Htmp, Otmp, Vtmp);

            double leakage = MaxLeakageDuringEvolution(
                    Htmp, Otmp, Vtmp,
                    evecs, ids,
                    omegaRes,
                    dt, tMax, hbar
            );

            leakFile << amp << "," << leakage << "\n";

            std::cout << "eF = " << amp
                      << "   max leakage = " << leakage << "\n";
        }

        std::cout << "\nFinished.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}