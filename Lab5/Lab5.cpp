#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>


void apply_dirichlet_bc(std::vector<std::vector<double>>& v,
                        int N, double L, double Vmax,
                        int k1, int k2, int k3, int k4)
{
    double Delta = 2.0 * L / (N - 1);

    for (int j = 0; j < N; ++j) {
        double y = j * Delta;
        v[0][j]     = Vmax * std::sin(k1 * M_PI * y / (2.0 * L));
        // task 1
        //        v[N-1][j]   = Vmax * std::sin(k3 * M_PI * y / (2.0 * L));
    }

    for (int i = 0; i < N; ++i) {
        double x = i * Delta;
        v[i][0]   = Vmax * std::sin(k4 * M_PI * x / (2.0 * L));
        v[i][N-1] = Vmax * std::sin(k2 * M_PI * x / (2.0 * L));
    }
}

double stopping_criterion(int N,
                          const std::vector<std::vector<double>>& v,
                          const std::vector<std::vector<double>>& rho,
                          double L, double epsilon)
{
    double Delta  = 2.0 * L / (N - 1);
    double Delta2 = Delta * Delta;

    double s = 0.0;

    for (int i = 1; i <= N-2; ++i) {
        for (int j = 1; j <= N-2; ++j) {
            double Ex = -(v[i+1][j] - v[i-1][j]) / (2.0 * Delta);
            double Ey = -(v[i][j+1] - v[i][j-1]) / (2.0 * Delta);

            s += (0.5 * (Ex*Ex + Ey*Ey) - v[i][j] * rho[i][j] / epsilon) * Delta2;
        }
    }
    return s;
}

std::vector<std::vector<double>> checking_correctness(const std::vector<std::vector<double>>& v,
                                            const std::vector<std::vector<double>>& rho,
                                            int N, double L, double epsilon)
{
    double Delta  = 2.0 * L / (N - 1);
    double Delta2 = Delta * Delta;

    std::vector<std::vector<double>> r(N, std::vector<double>(N, 0.0));

    for (int i = 1; i <= N-2; ++i) {
        for (int j = 1; j <= N-2; ++j) {
            r[i][j] =
                    (v[i+1][j] - 2.0 * v[i][j] + v[i-1][j]) / Delta2 +
                    (v[i][j+1] - 2.0 * v[i][j] + v[i][j-1]) / Delta2 +
                    rho[i][j] / epsilon;
        }
    }
    return r;
}

void SOR_method(double L, int N, double epsilon,
                double TOL, int K_max, double omega,
                std::vector<std::vector<double>>& v,
                std::vector<std::vector<double>>& rho,
                double Vmax)
{
    double Delta  = 2.0 * L / (N - 1);
    double Delta2 = Delta * Delta;

    double s_old = stopping_criterion(N, v, rho, L, epsilon);

    for (int k = 1; k <= K_max; ++k) {

        for (int i = 1; i <= N-2; ++i) {
            for (int j = 1; j <= N-2; ++j) {

                double rhs = v[i+1][j] + v[i-1][j] +
                             v[i][j+1] + v[i][j-1] +
                             rho[i][j] * Delta2 / epsilon;

                v[i][j] = (1.0 - omega) * v[i][j] + 0.25 * omega * rhs;
            }
        }

        // Neumann BC
        for (int j = 1; j <= N-2; ++j) {
            v[N-1][j] = v[N-2][j];
        }

        // convergence check
        double s_new = stopping_criterion(N, v, rho, L, epsilon);
        double delta = std::fabs((s_new - s_old) / s_old);

        if (delta < TOL) {
            std::cout << "Converged in " << k
                      << " iterations, delta = " << delta << std::endl;
            break;
        }

        s_old = s_new;
    }
}

void save_map(const std::string& filename,
                    const std::vector<std::vector<double>>& vector,
                    int N, double L)
{
    double Delta = 2.0 * L / (N - 1);
    std::ofstream fout(filename);

    for (int i = 0; i < N; ++i) {
        double x = -L + i * Delta;
        for (int j = 0; j < N; ++j) {
            double y = -L + j * Delta;
            fout << x << " " << y << " " << vector[i][j] << "\n";
        }
    }
}

void SOR_with_logging(double L, int N, double epsilon,
                      double TOL, int K_max, double omega,
                      std::vector<std::vector<double>>& v,
                      const std::vector<std::vector<double>>& rho,
                      const std::string& logfile)
{
    double Delta  = 2.0 * L / (N - 1);
    double Delta2 = Delta * Delta;

    std::ofstream log(logfile);

    double S_old = stopping_criterion(N, v, rho, L, epsilon);

    for (int k = 1; k <= K_max; ++k) {

        for (int i = 1; i <= N-2; ++i) {
            for (int j = 1; j <= N-2; ++j) {

                double rhs = v[i+1][j] + v[i-1][j] +
                             v[i][j+1] + v[i][j-1] +
                             rho[i][j] * Delta2 / epsilon;

                v[i][j] = (1.0 - omega) * v[i][j] + 0.25 * omega * rhs;
            }
        }

        for (int j = 1; j <= N-2; ++j) {
            v[N-1][j] = v[N-2][j];
        }

        double S_new = stopping_criterion(N, v, rho, L, epsilon);
        double dS = std::fabs((S_new - S_old) / S_old);

        log << k << " " << S_new << " " << dS << "\n";

        if (dS < TOL) {
            std::cout << "omega=" << omega
                      << " converged in " << k
                      << " iterations\n";
            break;
        }

        S_old = S_new;
    }
}


int main() {
    const double L       = 4.0;
    const int    N       = 100;
    const double epsilon = 1.0;
    const double TOL     = 1e-8;
    const int    K_max   = 10000;
    const double omega   = 1.5;
    const double Vmax    = 1.0;

    std::vector<std::vector<double>> v   (N, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> rho (N, std::vector<double>(N, 0.0)); // A_rho = 0

//      task 1
//    apply_dirichlet_bc(v, N, L, Vmax, 1, -1, 1, -1);
//    SOR_method(L, N, epsilon, TOL, K_max, omega, v, rho, Vmax);
//    save_map("potential.dat", v, N, L);
//
//    std::vector<std::vector<double>> err = checking_correctness(v, rho, N, L, epsilon);
//    save_map("error_map.dat", err, N, L);

//      task 2
//    apply_dirichlet_bc(v, N, L, Vmax, 1, -1, 1, -1);
//    SOR_method(L, N, epsilon, TOL, K_max, omega, v, rho, Vmax);
//    save_map("potential_2.dat", v, N, L);
//
//    std::vector<std::vector<double>> err = checking_correctness(v, rho, N, L, epsilon);
//    save_map("error_map_2.dat", err, N, L);

//      task 3
//    std::vector<std::vector<double>> rho(N, std::vector<double>(N, 0.0)); // rho=0
//    std::vector<double> omega_values = {1.0, 1.3, 1.6, 1.9};
//
//    for (double omega : omega_values) {
//        std::vector<std::vector<double>> v(N, std::vector<double>(N, 0.0));
//
//        apply_dirichlet_bc(v, N, L, Vmax, 1, -1, 1, -1);
//
//        std::string logfile =
//                "task3_omega_" + std::to_string(omega) + "_log.dat";
//
//        SOR_with_logging(L, N, epsilon, TOL, K_max,
//                         omega, v, rho, logfile);
//    }

//      task 4
    double A_rho = 1.0;

    double Delta = 2.0 * L / (N - 1);
    for (int i = 0; i < N; ++i) {
        double x = -L + i * Delta;
        for (int j = 0; j < N; ++j) {
            double y = -L + j * Delta;
            rho[i][j] = A_rho * x * y * std::exp(-(x*x + y*y));
        }
    }
    for(int i=0;i<N;i++) for(int j=0;j<N;j++) v[i][j]=0.0;

    // k1 = k2 = k3 = k4 = 0
    apply_dirichlet_bc(v, N, L, Vmax, 0, 0, 0, 0);

    SOR_method(L, N, epsilon, TOL, K_max, omega, v, rho, Vmax);

    save_map("case_a_potential.dat", v, N, L);

    std::vector<std::vector<double>> errA =
            checking_correctness(v, rho, N, L, epsilon);

    save_map("case_a_error.dat", errA, N, L);


    for(int i=0;i<N;i++) for(int j=0;j<N;j++) v[i][j]=0.0;

    // k1 = 1, k2 = -1, k3 = 1, k4 = -1
    apply_dirichlet_bc(v, N, L, Vmax, 1, -1, 1, -1);

    SOR_method(L, N, epsilon, TOL, K_max, omega, v, rho, Vmax);

    save_map("case_b_potential.dat", v, N, L);

    std::vector<std::vector<double>> errB =
            checking_correctness(v, rho, N, L, epsilon);

    save_map("case_b_error.dat", errB, N, L);

    return 0;
}
