#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
using std::vector;

vector<double> trapezoidal_method(double lambda_0, double lambda_1, double lambda_2,
                                  double delta_t, vector<double> N)
{
    vector<double> N_new = N;

    const double tolerance   = 1e-10;
    const int IT_max   = 50;
    int it = 0;

    const double a0 = 0.5 * delta_t * lambda_0;
    const double a1 = 0.5 * delta_t * lambda_1;
    const double a2 = 0.5 * delta_t * lambda_2;
    const double b0 = a0;  // = (Δt/2)λ0
    const double b1 = a1;  // = (Δt/2)λ1

    double epsilon_max;

    do {
        ++it;

        // F(N_new)
        const double F0 = N_new[0] - N[0] + a0 * (N[0] + N_new[0]);
        const double F1 = N_new[1] - N[1] - 0.5 * delta_t *
                                            ( lambda_0*(N[0] + N_new[0]) - lambda_1*(N[1] + N_new[1]) );
        const double F2 = N_new[2] - N[2] - 0.5 * delta_t *
                                            ( lambda_1*(N[1] + N_new[1]) - lambda_2*(N[2] + N_new[2]) );

        double dN0 = -F0 / (1.0 + a0);
        double dN1 = (-F1 + b0 * dN0) / (1.0 + a1);
        double dN2 = (-F2 + b1 * dN1) / (1.0 + a2);

        N_new[0] += dN0;
        N_new[1] += dN1;
        N_new[2] += dN2;

        epsilon_max = std::max({std::abs(dN0), std::abs(dN1), std::abs(dN2)});
    } while (epsilon_max > tolerance && it < IT_max);

    return N_new; // ≡ N_{i+1}
}

void adaptive_time_step(double lambda_0, double lambda_1, double lambda_2,
                        double t_max, vector<double> N,
                        double p   = 2.0,
                        double S   = 0.9,
                        double TOL = 1e-6,
                        double delta_t = 1e-2,
                        double t = 0.0,
                        const std::string& filename = "out.csv")
{
    std::ofstream file(filename);

    while (t < t_max) {
        double epsilon_max;
        vector<double> N1(3), N2(3), Nhalf(3), e(3);

        double delta_t_old = 0.0;

        do {
            N1 = trapezoidal_method(lambda_0, lambda_1, lambda_2, delta_t, N);
            Nhalf = trapezoidal_method(lambda_0, lambda_1, lambda_2, 0.5 * delta_t, N);
            N2    = trapezoidal_method(lambda_0, lambda_1, lambda_2, 0.5 * delta_t, Nhalf);

            const double denominator = std::pow(2.0, p) - 1.0;
            for (int k = 0; k < 3; ++k) e[k] = (N2[k] - N1[k]) / denominator;

            epsilon_max = std::max({std::abs(e[0]), std::abs(e[1]), std::abs(e[2])});

            double delta_t_new = delta_t * S * std::pow(TOL / epsilon_max, 1.0 / (p + 1.0));
            delta_t_old = delta_t;
            delta_t = delta_t_new;

            if (epsilon_max == 0.0) break;

        } while (TOL / epsilon_max < 1.0);

        for (int k = 0; k < 3; ++k) N[k] = N2[k] + e[k];

        t += delta_t_old;

        file << t << "," << N[0] << "," << N[1] << "," << N[2] << "\n";
    }
    file.close();
}

void exact_solutions(double lambda_0, double lambda_1, double lambda_2,
                     const std::vector<double>& N,
                     double t_max, double delta_t = 1e-2,
                     const std::string& filename = "exact.csv")
{
    std::ofstream file(filename);

    const double N0_0 = N[0];
    const double N1_0 = N[1];
    const double N2_0 = N[2];

    for (double t = 0.0; t <= t_max; t += delta_t) {
        double N0 = N0_0 * std::exp(-lambda_0 * t);

        double denom01 = (lambda_0 - lambda_1);
        double term1_N1 = (-lambda_0 * N0_0 / denom01) * std::exp(-lambda_0 * t);
        double term2_N1 = (lambda_0 * (N0_0 + N1_0) - lambda_1 * N1_0) / denom01 * std::exp(-lambda_1 * t);
        double N1 = term1_N1 + term2_N1;

        double denom02 = (lambda_2 - lambda_0);
        double denom12 = (lambda_2 - lambda_1);
        double term1_N2 = (-lambda_0 * lambda_1 * N0_0) / ((lambda_0 - lambda_1) * denom02) * std::exp(-lambda_0 * t);
        double term2_N2 = (lambda_1 * (lambda_0 * (N0_0 + N1_0) - lambda_1 * N1_0)) /
                          ((lambda_0 - lambda_1) * denom12) * std::exp(-lambda_1 * t);

        double num = lambda_0 * lambda_1 * (N0_0 + N1_0 + N2_0)
                     - lambda_0 * lambda_2 * N2_0
                     - lambda_1 * lambda_2 * (N1_0 + N2_0)
                     + lambda_2 * lambda_2 * N2_0;
        double term3_N2 = num / (denom12 * denom02) * std::exp(-lambda_2 * t);

        double N2 = term1_N2 + term2_N2 + term3_N2;

        file << t << "," << N0 << "," << N1 << "," << N2 << "\n";
    }
}


int main() {
    double t_max = 200.0;

    std::vector<double> N = {1.0, 0.0, 0.0};

    adaptive_time_step(
            1.0, 5.0, 50.0, t_max, N,
            2.0,0.9,1e-4,1e-2,0.0,
            "test_correctness.csv");

    exact_solutions(1.0, 5.0, 50.0, N,
                    t_max, 1e-2, "exact_correctness.csv");

    adaptive_time_step(
            100.0, 1.0, 0.01, t_max, N,
            2.0,0.9,1e-6,1e-2,0.0,
            "tolerance_-6.csv");

    adaptive_time_step(
            100.0, 1.0, 0.01, t_max, N,
            2.0,0.9,1e-3,1e-2,0.0,
            "tolerance_-3.csv");

    exact_solutions(100.0, 1.0, 0.01, N,
                    t_max, 1e-2, "exact_tolerance.csv");
}
