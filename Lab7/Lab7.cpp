#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <vector>
#include <filesystem>
#include <algorithm>

struct Params {
    double L      = 1.0;
    double rho0   = 1.0;
    double T      = 100.0;
    double c;

    int    N      = 301;
    double dx;
    double alpha  = 0.5;
    double dt;
    int    Nt     = 2000;

    int    Kmodes = 0;

    double gamma  = 0.0;
    double Fext   = 0.0;
    double Omega  = 0.0;
    int    ip     = 0;

    double u_init = 0.01;
    double xA;
    double v_init = 0.0;
    double sigma;

    Params() {
        recompute_derived();
        xA    = 0.5 * L;
        sigma = L / 20.0;
    }

    void recompute_derived() {
        c  = std::sqrt(T / rho0);
        dx = L / (N - 1);
        dt = alpha * dx / c;
    }

    double x(int i) const { return i * dx; }
    double t(int n) const { return n * dt; }
};

struct State {
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> a;
    std::vector<double> rho;
    std::vector<double> c;
};

void initialize_state(const Params& P, State& S){
    int N = P.N;

    S.u.assign(N, 0.0);
    S.v.assign(N, 0.0);
    S.a.assign(N, 0.0);
    S.rho.assign(N, 0.0);
    S.c.assign(N, 0.0);

    for (int i = 0; i < N; i++) {
        S.rho[i] = P.rho0;
        S.c[i]   = std::sqrt(P.T / S.rho[i]);
    }

    for (int i = 0; i < N; i++) {
        double x = P.x(i);
        double arg = (x - P.xA);
        S.u[i] = P.u_init * std::exp(-(arg * arg) / (2.0 * P.sigma * P.sigma));
    }

    for (int i = 0; i < N; i++) {
        double x = P.x(i);
        double arg = (x - P.xA);
        S.v[i] = (P.v_init * arg / (P.sigma * P.sigma)) * S.u[i];
    }

    S.u[0]     = 0.0;
    S.u[N - 1] = 0.0;

    S.v[0]     = 0.0;
    S.v[N - 1] = 0.0;
}

inline void fixed_BC(std::vector<double>& u){
    int N = u.size();
    u[0]     = 0.0;
    u[N - 1] = 0.0;
}

inline void open_BC(std::vector<double>& u){
    int N = u.size();
    u[0]     = u[1];
    u[N - 1] = u[N - 2];
}

void compute_acceleration(
        const Params& P,
        const std::vector<double>& u,
        const std::vector<double>& v,
        const std::vector<double>& c_i,
        std::vector<double>& a,
        double t){
    int N = P.N;

    for (int i = 1; i < N - 1; i++) {

        double lap = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (P.dx * P.dx);

        a[i] = (c_i[i] * c_i[i]) * lap - P.gamma * v[i];

        if (i == P.ip) {
            a[i] += P.Fext * std::sin(P.Omega * t);
        }
    }

    a[0]     = 0.0;
    a[N - 1] = 0.0;
}

void velocity_half_step(
        const Params& P,
        const std::vector<double>& v,
        const std::vector<double>& a,
        std::vector<double>& v_half){
    int N = P.N;

    if ((int)v_half.size() != N)
        v_half.assign(N, 0.0);

    for (int i = 1; i < N - 1; i++) {
        v_half[i] = v[i] + 0.5 * P.dt * a[i];
    }

    v_half[0]     = 0.0;
    v_half[N - 1] = 0.0;
}

void update_displacement(
        const Params& P,
        std::vector<double>& u,
        const std::vector<double>& v_half){
    int N = P.N;

    for (int i = 1; i < N - 1; i++) {
        u[i] += v_half[i] * P.dt;
    }
}

void velocity_full_step(
        const Params& P,
        std::vector<double>& v,
        const std::vector<double>& v_half,
        const std::vector<double>& a_new){
    int N = P.N;

    for (int i = 1; i < N - 1; i++) {
        v[i] = v_half[i] + 0.5 * P.dt * a_new[i];
    }

    v[0]     = 0.0;
    v[N - 1] = 0.0;
}

double compute_kinetic_energy(
        const Params& P,
        const std::vector<double>& rho,
        const std::vector<double>& v){
    double Ekin = 0.0;

    for (int i = 1; i < P.N - 1; i++) {
        double m_i = P.dx * rho[i];
        Ekin += 0.5 * m_i * v[i] * v[i];
    }

    return Ekin;
}

double compute_potential_energy(
        const Params& P,
        const std::vector<double>& u){
    double Epot = 0.0;

    for (int i = 1; i < P.N; i++) {
        double du_dx = (u[i] - u[i - 1]) / P.dx;
        Epot += 0.5 * P.T * P.dx * (du_dx * du_dx);
    }

    return Epot;
}

void compute_bk(const Params& P,
                const std::vector<double>& u,
                std::vector<double>& b){
    int N = P.N;
    int K = P.Kmodes;

    b.assign(K, 0.0);

    for (int k = 1; k <= K; k++) {
        double sum = 0.0;

        for (int i = 1; i < N - 1; i++) {
            sum += u[i] * std::sin(k * M_PI * i / (N - 1));
        }

        b[k - 1] = (2.0 / (N - 1)) * sum;;
    }
}

void compute_dk(const Params& P,
                const std::vector<double>& v,
                std::vector<double>& d){
    int N = P.N;
    int K = P.Kmodes;

    d.assign(K, 0.0);

    for (int k = 1; k <= K; k++) {
        double sum = 0.0;

        for (int i = 1; i < N - 1; i++) {
            sum += v[i] * std::sin(k * M_PI * i / (N - 1));
        }

        d[k - 1] = (2.0 / (N - 1)) * sum;;
    }
}

void compute_ek(const Params& P,
                const std::vector<double>& b,
                const std::vector<double>& d,
                std::vector<double>& e){
    int K = P.Kmodes;
    e.assign(K, 0.0);

    for (int k = 1; k <= K; k++) {
        double s = std::sin(k * M_PI / (P.N - 1));

        double term_pot = P.T * P.dx * (s * s) / (P.dx * P.dx);
        double term_kin = P.rho0 * P.dx;

        double bk = b[k - 1];
        double dk = d[k - 1];

        e[k - 1] = 0.25 * (P.N - 1) * (term_pot * bk * bk + term_kin * dk * dk);
    }
}


void run_simulation(const Params& P,
                    State& S,
                    bool use_fixed_bc,
                    std::ostream& energy_out,
                    std::ostream& u_out,
                    std::ostream* mode_out){
    int N  = P.N;
    int Nt = P.Nt;

    std::vector<double> a_old(N), a_new(N), v_half(N);

    compute_acceleration(P, S.u, S.v, S.c, a_old, 0.0);

    energy_out << "# t  Ekin  Epot  Etot\n";
    if (P.Kmodes > 0 && mode_out) {
        *mode_out << "# t ";
        for (int k = 1; k <= P.Kmodes; k++)
            *mode_out << "b" << k << " d" << k << " e" << k << " ";
        *mode_out << "\n";
    }

    for (int n = 0; n < Nt; n++) {

        double t = n * P.dt;

        velocity_half_step(P, S.v, a_old, v_half);
        update_displacement(P, S.u, v_half);

        if(use_fixed_bc) fixed_BC(S.u);
        else             open_BC(S.u);

        compute_acceleration(P, S.u, S.v, S.c, a_new, t + P.dt);
        velocity_full_step(P, S.v, v_half, a_new);

        double Ekin = compute_kinetic_energy(P, S.rho, S.v);
        double Epot = compute_potential_energy(P, S.u);
        double Etot = Ekin + Epot;

        energy_out << t << " " << Ekin << " " << Epot << " " << Etot << "\n";

        u_out << t;
        for (int i = 0; i < N; i++)
            u_out << " " << S.u[i];
        u_out << "\n";

        if (P.Kmodes > 0 && mode_out) {

            std::vector<double> b(P.Kmodes), d(P.Kmodes), e(P.Kmodes);

            compute_bk(P, S.u, b);
            compute_dk(P, S.v, d);
            compute_ek(P, b, d, e);

            *mode_out << t;
            for (int k = 0; k < P.Kmodes; k++)
                *mode_out << " " << b[k] << " " << d[k] << " " << e[k];
            *mode_out << "\n";
        }

        a_old = a_new;
    }
}

void set_nonhomogeneous_density(const Params& P, State& S){
    int N = P.N;
    double midpoint = 0.5 * P.L;

    for (int i = 0; i < N; i++) {

        double x = P.x(i);

        if (x <= midpoint)
            S.rho[i] = P.rho0;
        else
            S.rho[i] = 10.0 * P.rho0;

        S.c[i] = std::sqrt(P.T / S.rho[i]);
    }
}

std::vector<double> compute_mode_frequencies(const Params& P){
    std::vector<double> omega(P.Kmodes);

    for (int k = 1; k <= P.Kmodes; k++) {

        double cos_term = std::cos(k * M_PI / (P.N - 1));

        double inside = 2.0 * (1.0 - cos_term);

        omega[k - 1] = (P.c / P.dx) * std::sqrt(inside);
    }

    return omega;
}

void set_force_point(Params& P, int ip){
    if (ip < 1) ip = 1;
    if (ip > P.N - 2) ip = P.N - 2;

    P.ip = ip;
}

int main(){
    int task_id;
    std::cout << "Select task (1–6): ";
    std::cin  >> task_id;

    Params P;
    State S;

    std::ofstream energy_file("energies.txt");
    std::ofstream u_file("u_map.txt");
    std::ofstream mode_file;

    bool use_fixed_bc = true;

    std::vector<double> omega;
    switch (task_id) {
        case 1:
            std::cout << "Running Task 1...\n";
            P.gamma = 0.0;
            P.Fext  = 0.0;
            P.v_init = 0.0;
            use_fixed_bc = true;
            break;
        case 2:
            std::cout << "Running Task 2...\n";
            P.gamma = 0.0;
            P.Fext  = 0.0;
            P.v_init = 0.0;
            use_fixed_bc = false;
            break;
        case 3:
        {
            std::cout << "Running Task 3...\n";
            use_fixed_bc = false;

            std::cout << "Select velocity: (1) v=c/2, (2) v=c:\n";
            int opt; std::cin >> opt;

            if (opt == 1) P.v_init = P.c / 2.0;
            else          P.v_init = P.c;
        }
            break;
        case 4:
            std::cout << "Running Task 4...\n";
            P.v_init = P.c;
            P.xA     = P.L/4;
            use_fixed_bc = false;
            break;
        case 5:
        {
            std::cout << "Running Task 5...\n";

            P.gamma  = 150.0;
            P.v_init = 0.0;
            P.u_init = 0.01;
            P.Kmodes = 3;

            mode_file.open("modes.txt");
            use_fixed_bc = true;
        }
            break;
        case 6:
        {
            std::cout << "Running Task 6...\n";
            P.u_init = 0.0;
            P.v_init = 0.0;
            P.gamma  = 0.0;
            P.Fext   = 100.0;
            P.Kmodes = 3;
            P.Nt = 10000;
            use_fixed_bc = true;

            omega = compute_mode_frequencies(P);

            std::cout << "Choose case: (1) Ω=ω1 ip=30, (2) Ω=ω2 ip=30, (3) Ω=ω2 ip=center:\n";
            int c; std::cin >> c;

            if (c == 1) {
                P.Omega = omega[0];
                set_force_point(P, 30);
            }
            else if (c == 2) {
                P.Omega = omega[1];
                set_force_point(P, 30);
            }
            else {
                P.Omega = omega[1];
                set_force_point(P, (P.N - 1) / 2); // node of 2nd mode
            }

            mode_file.open("modes.txt");
        }
            break;

        default:
            std::cerr << "Invalid task ID.\n";
            return 1;
    }

    P.recompute_derived();
    initialize_state(P, S);


    if (task_id == 4) {
        set_nonhomogeneous_density(P, S);
    }

    if (task_id == 5) {
        // set displacement as sum of 3 eigenmodes
        for (int i = 0; i < P.N; i++) {
            double x = P.x(i);
            S.u[i] = P.u_init * (
                    std::sin(M_PI   * x / P.L)
                    + std::sin(2*M_PI * x / P.L)
                    + std::sin(3*M_PI * x / P.L)
            );
        }
        fixed_BC(S.u);
        std::fill(S.v.begin(), S.v.end(), 0.0);
    }

    run_simulation(P, S,
                   use_fixed_bc,
                   energy_file,
                   u_file,
                   (P.Kmodes > 0 ? &mode_file : nullptr));

    std::cout << "Simulation completed.\n";
    return 0;
}
