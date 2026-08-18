#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <vector>
#include <filesystem>
#include <algorithm>

using Grid = std::vector<std::vector<double>>;

struct Params {
    double L  = 10.0;
    int    N  = 51;
    double dx;
    double dt   = 10.0;
    double tmax = 10000.0;
    int    nmax;
    double D = 0.1;
    double h   = 0.0;
    double h_w = 0.0;
    double T_inf = 273.0;
    double T_low  = 293.0;
    double T_high = 1e4;
    int    kmax = 30;
    double TOL  = 1e-8;
    double Smax = 2.0;
    int ic;
    int jc;
    int j1_window;
    int j2_window;

    Params() {
        dx   = L / (N - 1);
        nmax = static_cast<int>(tmax / dt);

        ic = static_cast<int>(2.0 / dx);
        jc = static_cast<int>(8.0 / dx);

        j1_window = static_cast<int>(6.0 / dx);
        j2_window = static_cast<int>(9.0 / dx);
    }
};

void apply_all_boundaries(const Params& p, Grid& T)
{
    int N = p.N;
    double D = p.D;
    double dx = p.dx;
    double Tinf = p.T_inf;

    {
        double A = (p.h * dx) / (D + p.h * dx);
        double B =        D   / (D + p.h * dx);

        // west wall
        for (int j = 1; j < N-1; ++j)
            T[0][j] = A * Tinf + B * T[1][j];

        // north wall
        for (int i = 1; i < N-1; ++i)
            T[i][N-1] = A * Tinf + B * T[i][N-2];

        // south wall
        for (int i = 1; i < N-1; ++i)
            T[i][0] = A * Tinf + B * T[i][1];
    }

    {
        double A = (p.h * dx) / (D + p.h * dx);
        double B =        D   / (D + p.h * dx);

        double A_w = (p.h_w * dx) / (D + p.h_w * dx);
        double B_w =        D     / (D + p.h_w * dx);

        for (int j = 1; j < N-1; ++j) {
            if (j >= p.j1_window && j <= p.j2_window)
                T[N-1][j] = A_w * Tinf + B_w * T[N-2][j];
            else
                T[N-1][j] = A * Tinf + B * T[N-2][j];
        }
    }

    T[0][0]       = 0.5 * (T[0][1]     + T[1][0]);
    T[0][N-1]     = 0.5 * (T[0][N-2]   + T[1][N-1]);
    T[N-1][0]     = 0.5 * (T[N-2][0]   + T[N-1][1]);
    T[N-1][N-1]   = 0.5 * (T[N-2][N-1] + T[N-1][N-2]);
}


void compute_R(const Params& p,
               const Grid& Tn,
               const Grid& S,
               Grid& R,
               double w_old)
{
    int N = p.N;
    double D  = p.D;
    double dt = p.dt;
    double dx = p.dx;

    double alpha = D * dt / (2.0 * dx * dx);
    double beta  = dt / 2.0;

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            R[i][j] = 0.0;

    for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {

            double laplace_T =
                    Tn[i+1][j] + Tn[i-1][j] +
                    Tn[i][j+1] + Tn[i][j-1] -
                    4.0 * Tn[i][j];

            R[i][j] = Tn[i][j]
                      + alpha * laplace_T
                      + beta  * w_old * S[i][j];
        }
    }
}

void gauss_seidel_CN_step(const Params& p,
                          const Grid& R,
                          const Grid& S,
                          Grid& T,
                          double w)
{
    int N = p.N;
    double D  = p.D;
    double dt = p.dt;
    double dx = p.dx;

    double alpha = D * dt / (2.0 * dx * dx);
    double beta  = dt / 2.0;
    double denom = 1.0 + 4.0 * alpha;

    for (int k = 0; k < p.kmax; ++k) {

        for (int i = 1; i < N-1; ++i) {
            for (int j = 1; j < N-1; ++j) {

                double Tnew =
                        ( R[i][j]
                          + alpha * ( T[i-1][j] + T[i+1][j]
                                      + T[i][j-1] + T[i][j+1] )
                          + beta * w * S[i][j] )
                        / denom;

                T[i][j] = Tnew;
            }
        }

        apply_all_boundaries(p, T);  // includes window + corners

        double c = 0.0;

        for (int i = 1; i < N-1; ++i) {
            for (int j = 1; j < N-1; ++j) {

                double Lap =
                        T[i+1][j] + T[i-1][j] +
                        T[i][j+1] + T[i][j-1] -
                        4.0 * T[i][j];

                double Lij =
                        T[i][j]
                        - alpha * Lap
                        - beta * w * S[i][j];

                double diff = Lij - R[i][j];

                c += diff * diff * dx * dx;
            }
        }

        if (std::sqrt(c) < p.TOL)
            break;
    }
}

void update_heater_switch(const Params& p,
                          double T_sensor,
                          int& w)
{
    if (T_sensor < p.T_low) {w = 1;}
    else if (T_sensor > p.T_high) {w = 0;}
}

double compute_E_supplied(const Params& p,
                          const Grid& S,
                          int w,
                          int w_old)
{
    int N = p.N;
    double dt = p.dt;
    double dx = p.dx;


    double sumS = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sumS += S[i][j];
        }
    }

    return 0.5 * dt * dx * dx * sumS * (w + w_old);
}

double compute_E_window(const Params& p, const Grid& T)
{
    int N = p.N;
    double dt = p.dt;
    double dx = p.dx;
    double h_w = p.h_w;
    double Tinf = p.T_inf;

    double E = 0.0;

    for (int j = p.j1_window; j <= p.j2_window; ++j) {
        double Tcell = T[N-1][j];

        double flux = h_w * (Tcell - Tinf);

        E += flux * dx;
    }

    return E * dt;
}


void init_temperature(const Params& p, Grid& T)
{
    int N = p.N;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            T[i][j] = p.T_inf;
        }
    }
}

void init_source(const Params& p, Grid& S)
{
    int N = p.N;
    double dx = p.dx;

    double x1 = 8.0;
    double x2 = 8.8;
    double y1 = 2.0;
    double y2 = 4.0;

    for (int i = 0; i < N; ++i) {
        double x = i * dx;

        for (int j = 0; j < N; ++j) {
            double y = j * dx;

            if (x >= x1 && x <= x2 &&
                y >= y1 && y <= y2)
                S[i][j] = p.Smax;   // IMPORTANT
            else
                S[i][j] = 0.0;
        }
    }
}


void save_step(const std::string& dir,
               int n, double t,
               double T_sensor,
               double E_supplied,
               double E_window)
{
    std::ofstream file(dir + "/steps.dat", std::ios::app);
    file << n << " " << t << " " << T_sensor << " "
         << E_supplied << " " << E_window << "\n";
}

void save_temperature_field(const std::string& dir,
                            int n, const Grid& T)
{
    std::ostringstream fname;
    fname << dir << "/T_" << std::setw(6) << std::setfill('0') << n << ".dat";

    std::ofstream file(fname.str());

    int N = T.size();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            file << T[i][j];
            if (j < N-1) file << " ";
        }
        file << "\n";
    }
}


void CN_algorithm(const std::string& dir,
                  const Params& p,
                  bool save_step_data,
                  const std::vector<double>& save_times)
{
    int N = p.N;
    Grid T (N, std::vector<double>(N));
    Grid Tn(N, std::vector<double>(N));
    Grid S (N, std::vector<double>(N));
    Grid R (N, std::vector<double>(N));

    init_temperature(p, T);
    init_source(p, S);

    double t = 0.0;
    int w = 1, w_old = 1;

    std::vector<int> save_steps;
    save_steps.reserve(save_times.size());

for (double target_t : save_times)
        save_steps.push_back((int)std::round(target_t / p.dt));

    for (int n = 0; n <= p.nmax; ++n)
    {
        Tn = T;

        double T_sensor_old = Tn[p.ic][p.jc];
        w_old = w;

        update_heater_switch(p, T_sensor_old, w);

        compute_R(p, Tn, S, R, w_old);

        gauss_seidel_CN_step(p, R, S, T, w);

        double T_sensor = T[p.ic][p.jc];

        double E_supplied = compute_E_supplied(p, S, w, w_old);
        double E_window   = compute_E_window(p, T);  // uses T[N-2][j] after fix

        t += p.dt;

        if (save_step_data)
            save_step(dir, n, t, T_sensor, E_supplied, E_window);

        if (std::find(save_steps.begin(), save_steps.end(), n) != save_steps.end())
            save_temperature_field(dir, n, T);
    }

}


int main()
{
    std::filesystem::create_directory("p1");
    std::filesystem::create_directory("p2");
    std::filesystem::create_directory("p3");
    std::filesystem::create_directory("p4");
    std::filesystem::create_directory("p5");
    std::filesystem::create_directory("p6");

    Params p1;

    Params p2;
    p2.h   = 0.002;
    p2.h_w = 0.002;

    Params p3;
    p3.h   = 0.002;
    p3.h_w = 0.01;

    Params p4;
    p4.h      = 0.002;
    p4.h_w    = 0.01;
    p4.T_high = 298;

    Params p5;
    p5.h      = 0.0;
    p5.h_w    = 1.0;
    p5.T_high = 298;

    Params p6;
    p6.h      = 0.0;
    p6.h_w   = 1.0;
    p6.T_high = 10000;

    CN_algorithm("p1", p1, true, {10,100,1000,10000});
    CN_algorithm("p2", p2, true, {10,100,1000,10000});
    CN_algorithm("p3", p3, true, {10,100,1000,10000});
    CN_algorithm("p4", p4, true, {1830,2170,2510,2970});
    CN_algorithm("p5", p5, true, {});
    CN_algorithm("p6", p6, true, {});

    std::cout << "All simulations completed.\n";
    return 0;
}