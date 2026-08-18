
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <array>
#include <stdexcept>
#include <string>
#include <sstream>
#include <algorithm>

#include <Eigen/Dense>

// ============================================================
// Advection-diffusion equation in 2D, FEM version
//
// PDE:
//      du/dt = -vx du/dx - vy du/dy + D Laplacian(u)
//
// Periodic square domain:
//      (x,y) in (0,L) x (0,L)
//      u(x+mL,y+nL) = u(x,y)
//
// FEM matrix equation:
//      O dc/dt = (C - D S)c
//
// where:
//      O  - overlap/mass matrix
//      S  - stiffness matrix for -Laplacian
//      C  - advection matrix corresponding to -vx d/dx - vy d/dy
//
// Crank-Nicolson:
//      [O - dt/2 (C - D S)] c^{n+1}
//      =
//      [O + dt/2 (C - D S)] c^n
//
// Assignment parameters:
//      L = 5
//      dt = 0.02
//      u(x,y,0) = exp[-2(x-L/2)^2 - 2(y-L/2)^2]
//
// Deliverables:
//      1) pure advection: D=0, vx=vy=1
//      2) pure diffusion: D=0.1, vx=vy=0
//      3) advection-diffusion: D=0.1, vx=vy=1
//
// Outputs:
//      pure_advection_stats.csv
//      pure_diffusion_stats.csv
//      advection_diffusion_stats.csv
//      *_snapshot_*.txt
// ============================================================

struct Node {
    double x, y;
};

struct PeriodicMesh {
    std::vector<Node> p;
    std::vector<std::vector<int>> nlg;
    int N;
    double L;
    double a;
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

int PeriodicId(int i, int j, int N) {
    int ii = ((i % N) + N) % N;
    int jj = ((j % N) + N) % N;
    return jj * N + ii;
}

PeriodicMesh GeneratePeriodicBilinearMesh(int N, double L) {
    PeriodicMesh mesh;
    mesh.N = N;
    mesh.L = L;
    mesh.a = L / N;

    mesh.p.resize(N * N);

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            int id = PeriodicId(i, j, N);
            mesh.p[id] = Node{i * mesh.a, j * mesh.a};
        }
    }

    mesh.nlg.assign(N * N, std::vector<int>(4, -1));

    int elem = 0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            mesh.nlg[elem][0] = PeriodicId(i,     j,     N);
            mesh.nlg[elem][1] = PeriodicId(i + 1, j,     N);
            mesh.nlg[elem][2] = PeriodicId(i,     j + 1, N);
            mesh.nlg[elem][3] = PeriodicId(i + 1, j + 1, N);
            elem++;
        }
    }

    return mesh;
}

std::array<std::array<double, 4>, 4> ComputeLocalOverlapMatrix(double a) {
    std::array<std::array<double, 4>, 4> O{};

    auto w = GaussWeights4();
    auto p = GaussPoints4();

    double jacobian = a * a / 4.0;

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            double sum = 0.0;

            for (int l = 0; l < 4; ++l) {
                for (int n = 0; n < 4; ++n) {
                    double xi1 = p[l];
                    double xi2 = p[n];

                    sum += w[l] * w[n] * g(j, xi1, xi2) * g(i, xi1, xi2);
                }
            }

            O[j][i] = jacobian * sum;
        }
    }

    return O;
}

std::array<std::array<double, 4>, 4> ComputeLocalStiffnessMatrix() {
    std::array<std::array<double, 4>, 4> S{};

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

            S[j][i] = sum;
        }
    }

    return S;
}

std::array<std::array<double, 4>, 4> ComputeLocalDerivativeMatrix(double a, char direction) {
    std::array<std::array<double, 4>, 4> Dmat{};

    auto w = GaussWeights4();
    auto p = GaussPoints4();

    double factor = a / 2.0;

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            double sum = 0.0;

            for (int l = 0; l < 4; ++l) {
                for (int n = 0; n < 4; ++n) {
                    double xi1 = p[l];
                    double xi2 = p[n];

                    double derivative = 0.0;

                    if (direction == 'x') {
                        derivative = dg_dxi1(i, xi1, xi2);
                    } else if (direction == 'y') {
                        derivative = dg_dxi2(i, xi1, xi2);
                    } else {
                        throw std::runtime_error("Unknown derivative direction.");
                    }

                    sum += w[l] * w[n] * g(j, xi1, xi2) * derivative;
                }
            }

            Dmat[j][i] = factor * sum;
        }
    }

    return Dmat;
}

void AssembleGlobalMatrices(
        const PeriodicMesh& mesh,
        Eigen::MatrixXd& O,
        Eigen::MatrixXd& S,
        Eigen::MatrixXd& Dx,
        Eigen::MatrixXd& Dy
) {
    int nNodes = static_cast<int>(mesh.p.size());

    O  = Eigen::MatrixXd::Zero(nNodes, nNodes);
    S  = Eigen::MatrixXd::Zero(nNodes, nNodes);
    Dx = Eigen::MatrixXd::Zero(nNodes, nNodes);
    Dy = Eigen::MatrixXd::Zero(nNodes, nNodes);

    auto Oloc  = ComputeLocalOverlapMatrix(mesh.a);
    auto Sloc  = ComputeLocalStiffnessMatrix();
    auto Dxloc = ComputeLocalDerivativeMatrix(mesh.a, 'x');
    auto Dyloc = ComputeLocalDerivativeMatrix(mesh.a, 'y');

    for (size_t e = 0; e < mesh.nlg.size(); ++e) {
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                int gj = mesh.nlg[e][j];
                int gi = mesh.nlg[e][i];

                O(gj, gi)  += Oloc[j][i];
                S(gj, gi)  += Sloc[j][i];
                Dx(gj, gi) += Dxloc[j][i];
                Dy(gj, gi) += Dyloc[j][i];
            }
        }
    }
}

double InitialCondition(double x, double y, double L) {
    double dx = x - L / 2.0;
    double dy = y - L / 2.0;
    return std::exp(-2.0 * dx * dx - 2.0 * dy * dy);
}

Eigen::VectorXd BuildInitialVector(const PeriodicMesh& mesh) {
    int nNodes = static_cast<int>(mesh.p.size());
    Eigen::VectorXd c(nNodes);

    for (int i = 0; i < nNodes; ++i) {
        c(i) = InitialCondition(mesh.p[i].x, mesh.p[i].y, mesh.L);
    }

    return c;
}

double Mass(const Eigen::VectorXd& c, const Eigen::MatrixXd& O) {
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(c.size());
    return ones.dot(O * c);
}

std::pair<double, double> PeriodicCenterOfMass(
        const PeriodicMesh& mesh,
        const Eigen::VectorXd& c,
        const Eigen::MatrixXd& O
) {
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(c.size());
    Eigen::VectorXd weights = O * ones;

    double sx = 0.0, cx = 0.0;
    double sy = 0.0, cy = 0.0;

    for (int i = 0; i < c.size(); ++i) {
        double w = weights(i) * c(i);

        double thetaX = 2.0 * M_PI * mesh.p[i].x / mesh.L;
        double thetaY = 2.0 * M_PI * mesh.p[i].y / mesh.L;

        sx += w * std::sin(thetaX);
        cx += w * std::cos(thetaX);

        sy += w * std::sin(thetaY);
        cy += w * std::cos(thetaY);
    }

    double meanThetaX = std::atan2(sx, cx);
    double meanThetaY = std::atan2(sy, cy);

    if (meanThetaX < 0.0) meanThetaX += 2.0 * M_PI;
    if (meanThetaY < 0.0) meanThetaY += 2.0 * M_PI;

    double x = mesh.L * meanThetaX / (2.0 * M_PI);
    double y = mesh.L * meanThetaY / (2.0 * M_PI);

    return {x, y};
}

double ShapeErrorRelativeToInitialShifted(
        const PeriodicMesh& mesh,
        const Eigen::VectorXd& c,
        double shiftX,
        double shiftY
) {
    double num = 0.0;
    double den = 0.0;

    for (int k = 0; k < c.size(); ++k) {
        double x0 = mesh.p[k].x - shiftX;
        double y0 = mesh.p[k].y - shiftY;

        while (x0 < 0.0) x0 += mesh.L;
        while (x0 >= mesh.L) x0 -= mesh.L;
        while (y0 < 0.0) y0 += mesh.L;
        while (y0 >= mesh.L) y0 -= mesh.L;

        double exact = InitialCondition(x0, y0, mesh.L);
        double diff = c(k) - exact;

        num += diff * diff;
        den += exact * exact;
    }

    return std::sqrt(num / den);
}

void WriteSnapshot(
        const PeriodicMesh& mesh,
        const Eigen::VectorXd& c,
        const std::string& filename
) {
    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Cannot open snapshot file: " + filename);
    }

    fout << std::fixed << std::setprecision(12);
    fout << "# x y u\n";

    for (int j = 0; j < mesh.N; ++j) {
        for (int i = 0; i < mesh.N; ++i) {
            int id = PeriodicId(i, j, mesh.N);
            fout << mesh.p[id].x << " "
                 << mesh.p[id].y << " "
                 << c(id) << "\n";
        }
        fout << "\n";
    }
}

struct CaseParams {
    std::string name;
    double D;
    double vx;
    double vy;
    double tMax;
    int snapshotEvery;
};

void RunCase(
        const PeriodicMesh& mesh,
        const Eigen::MatrixXd& O,
        const Eigen::MatrixXd& S,
        const Eigen::MatrixXd& Dx,
        const Eigen::MatrixXd& Dy,
        double dt,
        const CaseParams& cp
) {
    std::cout << "\nRunning case: " << cp.name << "\n";
    std::cout << "D = " << cp.D
              << ", vx = " << cp.vx
              << ", vy = " << cp.vy
              << ", tMax = " << cp.tMax << "\n";

    Eigen::VectorXd c = BuildInitialVector(mesh);

    Eigen::MatrixXd C = -cp.vx * Dx - cp.vy * Dy;
    Eigen::MatrixXd Aop = C - cp.D * S;

    Eigen::MatrixXd LHS = O - 0.5 * dt * Aop;
    Eigen::MatrixXd RHS = O + 0.5 * dt * Aop;

    Eigen::PartialPivLU<Eigen::MatrixXd> solver(LHS);

    int nSteps = static_cast<int>(std::ceil(cp.tMax / dt));

    std::string statsFilename = cp.name + "_stats.csv";
    std::ofstream stats(statsFilename);

    if (!stats) {
        throw std::runtime_error("Cannot open stats file: " + statsFilename);
    }

    stats << std::setprecision(15);
    stats << "t,min_u,max_u,mass,center_x,center_y,shape_error\n";

    for (int step = 0; step <= nSteps; ++step) {
        double t = step * dt;

        double minU = c.minCoeff();
        double maxU = c.maxCoeff();
        double mass = Mass(c, O);
        auto center = PeriodicCenterOfMass(mesh, c, O);

        double shiftX = std::fmod(cp.vx * t, mesh.L);
        double shiftY = std::fmod(cp.vy * t, mesh.L);
        if (shiftX < 0.0) shiftX += mesh.L;
        if (shiftY < 0.0) shiftY += mesh.L;

        double shapeError = ShapeErrorRelativeToInitialShifted(mesh, c, shiftX, shiftY);

        stats << t << ","
              << minU << ","
              << maxU << ","
              << mass << ","
              << center.first << ","
              << center.second << ","
              << shapeError << "\n";

        if (step % cp.snapshotEvery == 0 || step == nSteps) {
            std::ostringstream filename;
            filename << cp.name
                     << "_snapshot_"
                     << std::setw(6) << std::setfill('0') << step
                     << ".txt";

            WriteSnapshot(mesh, c, filename.str());
        }

        if (step == nSteps) {
            break;
        }

        c = solver.solve(RHS * c);
    }

    std::cout << "Saved: " << statsFilename << "\n";
}

int main() {
    try {
        const double L = 5.0;
        const double dt = 0.02;

        // N = 25 is a reasonable starting value.
        // Increase to 30 or 40 for smoother results if runtime is acceptable.
        const int N = 25;

        PeriodicMesh mesh = GeneratePeriodicBilinearMesh(N, L);

        std::cout << std::fixed << std::setprecision(12);
        std::cout << "2D advection-diffusion FEM with periodic boundaries\n";
        std::cout << "N = " << N << "\n";
        std::cout << "nodes = " << mesh.p.size() << "\n";
        std::cout << "elements = " << mesh.nlg.size() << "\n";
        std::cout << "L = " << L << "\n";
        std::cout << "a = " << mesh.a << "\n";
        std::cout << "dt = " << dt << "\n";

        Eigen::MatrixXd O, S, Dx, Dy;
        AssembleGlobalMatrices(mesh, O, S, Dx, Dy);

        std::cout << "Matrices assembled.\n";

        // For vx=vy=1 and L=5:
        // first return to initial center:  t=5
        // second return to initial center: t=10
        CaseParams pureAdvection;
        pureAdvection.name = "pure_advection";
        pureAdvection.D = 0.0;
        pureAdvection.vx = 1.0;
        pureAdvection.vy = 1.0;
        pureAdvection.tMax = 10.0;
        pureAdvection.snapshotEvery = 25;

        CaseParams pureDiffusion;
        pureDiffusion.name = "pure_diffusion";
        pureDiffusion.D = 0.1;
        pureDiffusion.vx = 0.0;
        pureDiffusion.vy = 0.0;
        pureDiffusion.tMax = 20.0;
        pureDiffusion.snapshotEvery = 50;

        CaseParams advectionDiffusion;
        advectionDiffusion.name = "advection_diffusion";
        advectionDiffusion.D = 0.1;
        advectionDiffusion.vx = 1.0;
        advectionDiffusion.vy = 1.0;
        advectionDiffusion.tMax = 5.0;
        advectionDiffusion.snapshotEvery = 25;

        RunCase(mesh, O, S, Dx, Dy, dt, pureAdvection);
        RunCase(mesh, O, S, Dx, Dy, dt, pureDiffusion);
        RunCase(mesh, O, S, Dx, Dy, dt, advectionDiffusion);

        std::cout << "\nFinished.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
