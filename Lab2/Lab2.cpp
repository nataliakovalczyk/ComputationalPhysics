#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
// #include <lapacke.h>
#define k 1.0
#define m 1.0
#define omega_0 1.0
#define x_0 0.0
#define v_0 0.0


std::vector<double> calculate_rhs_ODE(
        const std::vector<double>& u,  // u[0] = x, u[1] = v
        double alpha, double F0, double Omega_ext, double t) {
    std::vector<double> rhs(2);
    rhs[0] = u[1];  // dx/dt = v
    rhs[1] = (-k * u[0] - alpha * u[1] + F0 * std::sin(Omega_ext * t)) / m; // dv/dt

    return rhs;
}

std::vector<double> RK4_method(
        const std::vector<double>& u,
        double t, double dt, double alpha, double F0, double Omega_ext) {

    const std::size_t n = u.size();
    std::vector<double> w(n), k1, k2, k3, k4, u_next(n);

    // k1
    k1 = calculate_rhs_ODE(u, alpha, F0, Omega_ext, t);

    // k2
    for (std::size_t i = 0; i < n; ++i) w[i] = u[i] + 0.5 * dt * k1[i];
    k2 = calculate_rhs_ODE(w, alpha, F0, Omega_ext, t + 0.5 * dt);

    // k3
    for (std::size_t i = 0; i < n; ++i) w[i] = u[i] + 0.5 * dt * k2[i];
    k3 = calculate_rhs_ODE(w, alpha, F0, Omega_ext, t + 0.5 * dt);

    // k4
    for (std::size_t i = 0; i < n; ++i) w[i] = u[i] + dt * k3[i];
    k4 = calculate_rhs_ODE(w, alpha, F0, Omega_ext, t + dt);

    // update u
    for (std::size_t i = 0; i < n; ++i)
    u_next[i] = u[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);

    return u_next;
}

// x(t) exact
double x_exact(double t, double x0, double v0, double alpha) {
    const double tau = 2.0 * m / alpha;

    const double Omega_d = std::sqrt(omega_0 * omega_0 - 1.0 / (tau * tau));
    const double phi = std::atan((x0 + v0 * tau) / (x0 * Omega_d * tau));
    const double amplitude = std::sqrt(
            x0 * x0 + std::pow((x0 + v0 * tau) / (Omega_d * tau), 2));
    return amplitude * std::exp(-t / tau) * std::cos(Omega_d * t - phi);
}

// last peak
double simulate_last_peak(double alpha, double F0, double Omega_ext,
                          double tmax, std::size_t N,
                          double x0 = 0.0, double v0 = 0.0) {
    const double dt = tmax / static_cast<double>(N);
    std::vector<double> u{ x0, v0 };

    double t = 0.0;
    double x_im1 = u[0];

    auto u_next = RK4_method(u, t, dt, alpha, F0, Omega_ext);
    t += dt;
    u = u_next;
    double x_i = u[0];

    double last_peak = std::abs(x_i);

    for (std::size_t i = 2; i <= N; ++i) {
        u_next = RK4_method(u, t, dt, alpha, F0, Omega_ext);
        t += dt;
        double x_ip1 = u_next[0];

        if (x_i > x_im1 && x_i > x_ip1) {
            last_peak = std::abs(x_i);
        }

        x_im1 = x_i;
        x_i = x_ip1;
        u = u_next;
    }
    return last_peak;
}

int main() {
    const double alpha = 0.0;
    const double F0 = 0.0;
    const double Omega_ext = 0.0;
    const double tmax = 50.0;
    const std::size_t N = 10000;
    const double dt = tmax / static_cast<double>(N);

    double x0 = 1.0;
    double v0 = 0.0;

    std::vector<double> u = {x0, v0};
    double t = 0.0;

    std::ofstream energy("energy_phase_space.csv");
    energy << std::fixed << std::setprecision(10);
    energy << "t,x,v,Ekin,Epot,Etot\n";

    double Ekin = 0.5 * m * u[1] * u[1];
    double Epot = 0.5 * k * u[0] * u[0];
    double Etot = Ekin + Epot;
    energy << t << "," << u[0] << "," << u[1] << ","
        << Ekin << "," << Epot << "," << Etot << "\n";

    for (std::size_t i = 0; i < N; ++i) {
        std::vector<double> u_next = RK4_method(u, t, dt, alpha, F0, Omega_ext);
        t += dt;
        u = u_next;

        Ekin = 0.5 * m * u[1] * u[1];
        Epot = 0.5 * k * u[0] * u[0];
        Etot = Ekin + Epot;

        energy << t << "," << u[0] << "," << u[1] << ","
            << Ekin << "," << Epot << "," << Etot << "\n";
    }

    energy.close();

    std::cout << "Data written to .csv\n";

    const double alpha1 = 0.1;        // friction
    const double F01 = 0.0;
    const double Omega_ext1 = 0.0;
    const double tmax1 = 50.0;
    const std::size_t N1 = 10000;
    const double dt1 = tmax1 / static_cast<double>(N1);

    double x01 = 1.0;
    double v01 = 0.0;

    std::vector<double> u1 = {x01, v01};
    double t1 = 0.0;
    std::ofstream damped("oscillator_damped.csv");
    damped << std::fixed << std::setprecision(10);
    damped << "t,x_num,v_num,Ekin,Epot,Etot,x_exact\n";

    double Ekin1 = 0.5 * m * u1[1] * u1[1];
    double Epot1 = 0.5 * k * u1[0] * u1[0];
    double Etot1 = Ekin1 + Epot1;
    damped << t1 << "," << u1[0] << "," << u1[1] << ","
        << Ekin1 << "," << Epot1 << "," << Etot1 << ","
        << x_exact(t1, x01, v01, alpha1) << "\n";

    for (std::size_t i = 0; i < N; ++i) {
        auto u_next1 = RK4_method(u1, t1, dt1, alpha1, F01, Omega_ext1);
        t1 += dt1;
        u1 = u_next1;

        Ekin1 = 0.5 * m * u1[1] * u1[1];
        Epot1 = 0.5 * k * u1[0] * u1[0];
        Etot1 = Ekin1 + Epot1;

        damped << t1 << "," << u1[0] << "," << u1[1] << ","
            << Ekin1 << "," << Epot1 << "," << Etot1 << ","
            << x_exact(t1, x01, v01, alpha1) << "\n";
    }

    damped.close();

    std::cout << "Data written to .csv\n";

    // four alphas
    const double F02 = 0.0, Omega_ext2 = 0.0;
    const double tmax2 = 50.0;

    const std::size_t N2 = 10000;
    const double dt2 = tmax2 / static_cast<double>(N2);

    const double x02 = 1.0, v02 = 0.0;

    const std::vector<double> alphas = {1e-4, 0.1, 0.5, 1.95};

    std::vector<std::vector<double>> u2(alphas.size(), std::vector<double>{x0, v0});
    double t2 = 0.0;

    std::ofstream friction("trajectories.csv");
    friction << std::fixed << std::setprecision(10);
    friction << "t";
    for (double a : alphas) friction << ",x_alpha_" << a;
    friction << "\n";

    friction << t2;
    for (std::size_t j = 0; j < alphas.size(); ++j) friction << "," << u2[j][0];
    friction << "\n";

    for (std::size_t i = 0; i < N; ++i) {
        t2 += dt2;
        for (std::size_t j = 0; j < alphas.size(); ++j) {
            u2[j] = RK4_method(u2[j], t2 - dt2, dt2, alphas[j], F02, Omega_ext2);
        }
        friction << t2;
        for (std::size_t j = 0; j < alphas.size(); ++j) friction << "," << u2[j][0];
        friction << "\n";
    }
    friction.close();

    std::cout << "Data written to .csv\n";

    // last task
    const double F03 = 1.0;
    const double tmax3 = 1000.0;
    const std::size_t N3 = 200000;
    const double dt3 = tmax3 / static_cast<double>(N3);
    const double x03 = 0.0, v03 = 0.0;

    {
        const double alpha3 = 1.0;
        const double Omega_ext3 = 0.5 * omega_0;

        std::ofstream driven("driven_test.csv");
        driven << std::fixed << std::setprecision(10);
        driven << "t,x\n";

        std::vector<double> u3{ x03, v03};
        double t3 = 0.0;
        driven << t3 << "," << u[0] << "\n";
        for (std::size_t i = 0; i < N3; ++i) {
            auto u_next3 = RK4_method(u3, t3, dt3, alpha3, F03, Omega_ext3);
            t3 += dt3;
            u3 = u_next3;
            driven << t3 << "," << u3[0] << "\n";
        }
        driven.close();
        std::cout << "Data written to .csv\n";
    }

    // resonance sweep
    const std::vector<double> alphas3 = {0.01, 0.1, 0.5, 1.0};

    std::ofstream sweep("amplitude_sweep.csv");
    sweep << std::fixed << std::setprecision(10);
    // header
    sweep << "Omega";
    for (double a : alphas3) sweep << ",A_alpha_" << a;
    sweep << ",A_analytic_alpha_1.0\n";

    for (double Omega3 = 0.10; Omega3 <= 2.000001; Omega3 += 0.01) {
        sweep << Omega3;

        double analytic_a1 = std::numeric_limits<double>::quiet_NaN();

        for (double alpha3 : alphas3) {
            double A_last = simulate_last_peak(alpha3, F03, Omega3, tmax3, N3, x03, v03);
            sweep << "," << A_last;

            if (std::abs(alpha3 - 1.0) < 1e-12) {
                analytic_a1 = F03 / std::sqrt(std::pow(k - m * Omega3 * Omega3, 2.0)
                                             + std::pow(alpha3 * Omega3, 2.0));
            }
        }
        sweep << "," << analytic_a1 << "\n";
        std::cout << "." << std::flush;
    }
    sweep.close();
    
    std::cout << "Data written to .csv\n";
    return 0;
}