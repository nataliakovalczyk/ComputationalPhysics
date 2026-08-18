#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>

using namespace std;

using Grid = vector<vector<double>>;

double rhs(double x, double y) {
    return 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}

double exact_solution(double x, double y) {
    return sin(M_PI * x) * sin(M_PI * y);
}


Grid make_grid(int N, double value = 0.0) {
    return Grid(N + 1, vector<double>(N + 1, value));
}

void fill_rhs(Grid& f, int N, double dx) {
    for (int i = 0; i <= N; i++) {
        double x = i * dx;
        for (int j = 0; j <= N; j++) {
            double y = j * dx;
            f[i][j] = rhs(x, y);
        }
    }
}


double gauss_seidel_step(Grid& u, const Grid& f, int N, double dx) {
    double max_change = 0.0;
    double dx2 = dx * dx;

    for (int i = 1; i < N; i++) {
        for (int j = 1; j < N; j++) {
            double old_u = u[i][j];

            u[i][j] = 0.25 * (
                    u[i + 1][j] + u[i - 1][j] +
                    u[i][j + 1] + u[i][j - 1] +
                    dx2 * f[i][j]
            );

            max_change = max(max_change, fabs(u[i][j] - old_u));
        }
    }

    return max_change;
}

double sor_step(Grid& u, const Grid& f, int N, double dx, double omega) {
    double max_change = 0.0;
    double dx2 = dx * dx;

    for (int i = 1; i < N; i++) {
        for (int j = 1; j < N; j++) {
            double old_u = u[i][j];

            double gs_value = 0.25 * (
                    u[i + 1][j] + u[i - 1][j] +
                    u[i][j + 1] + u[i][j - 1] +
                    dx2 * f[i][j]
            );

            u[i][j] = (1.0 - omega) * old_u + omega * gs_value;

            max_change = max(max_change, fabs(u[i][j] - old_u));
        }
    }

    return max_change;
}

struct ResidualInfo {
    double max_abs;
    double rms;
};

Grid compute_residual_grid(const Grid& u, const Grid& f, int N, double dx) {
    Grid delta = make_grid(N, 0.0);
    double dx2 = dx * dx;

    for (int i = 1; i < N; i++) {
        for (int j = 1; j < N; j++) {
            delta[i][j] =
                    (u[i + 1][j] + u[i - 1][j] +
                     u[i][j + 1] + u[i][j - 1] -
                     4.0 * u[i][j]) / dx2
                    + f[i][j];
        }
    }

    return delta;
}

ResidualInfo compute_residual_info(const Grid& u, const Grid& f, int N, double dx) {
    double dx2 = dx * dx;
    double max_abs = 0.0;
    double sum_sq = 0.0;
    int count = 0;

    for (int i = 1; i < N; i++) {
        for (int j = 1; j < N; j++) {
            double r =
                    (u[i + 1][j] + u[i - 1][j] +
                     u[i][j + 1] + u[i][j - 1] -
                     4.0 * u[i][j]) / dx2
                    + f[i][j];

            max_abs = max(max_abs, fabs(r));
            sum_sq += r * r;
            count++;
        }
    }

    ResidualInfo info;
    info.max_abs = max_abs;
    info.rms = sqrt(sum_sq / count);
    return info;
}

double compute_energy(const Grid& u, const Grid& f, int N, double dx) {
    double S = 0.0;

    for (int i = 1; i < N; i++) {
        for (int j = 1; j < N; j++) {
            double ux = (u[i + 1][j] - u[i - 1][j]) / (2.0 * dx);
            double uy = (u[i][j + 1] - u[i][j - 1]) / (2.0 * dx);

            S += (0.5 * ux * ux + 0.5 * uy * uy - f[i][j] * u[i][j]) * dx * dx;
        }
    }

    return S;
}

struct ErrorInfo {
    double max_abs;
    double rms;
};

ErrorInfo compute_error_info(const Grid& u, int N, double dx) {
    double max_abs = 0.0;
    double sum_sq = 0.0;
    int count = 0;

    for (int i = 1; i < N; i++) {
        double x = i * dx;
        for (int j = 1; j < N; j++) {
            double y = j * dx;
            double e = u[i][j] - exact_solution(x, y);

            max_abs = max(max_abs, fabs(e));
            sum_sq += e * e;
            count++;
        }
    }

    ErrorInfo info;
    info.max_abs = max_abs;
    info.rms = sqrt(sum_sq / count);
    return info;
}

void save_history_header(ofstream& file) {
    file << "iter,max_change,S,max_abs_res,rms_res\n";
}

void save_history_row(ofstream& file, int iter, double max_change,
                      double S, const ResidualInfo& r) {
    file << iter << ","
         << max_change << ","
         << S << ","
         << r.max_abs << ","
         << r.rms << "\n";
}

void save_scalar_field(const Grid& g, int N, double dx, const string& filename) {
    ofstream file(filename);
    file << fixed << setprecision(12);
    file << "x,y,value\n";

    for (int i = 0; i <= N; i++) {
        double x = i * dx;
        for (int j = 0; j <= N; j++) {
            double y = j * dx;
            file << x << "," << y << "," << g[i][j] << "\n";
        }
    }
}

void save_solution_with_error(const Grid& u, int N, double dx, const string& filename) {
    ofstream file(filename);
    file << fixed << setprecision(12);
    file << "i,j,x,y,u_num,u_exact,error\n";

    for (int i = 0; i <= N; i++) {
        double x = i * dx;
        for (int j = 0; j <= N; j++) {
            double y = j * dx;
            double u_ex = exact_solution(x, y);
            double err = u[i][j] - u_ex;

            file << i << ","
                 << j << ","
                 << x << ","
                 << y << ","
                 << u[i][j] << ","
                 << u_ex << ","
                 << err << "\n";
        }
    }
}


struct RunResult {
    int N;
    double dx;
    int iterations;
    double S_final;
    double S_error;
    double max_abs_res;
    double rms_res;
    double max_abs_error;
    double rms_error;

    double omega;
};

RunResult solve_for_N(int N,
                      double tol_res,
                      int max_iter,
                      int snapshot_interval) {
    double dx = 3.0 / N;

    Grid u = make_grid(N, 0.0);
    Grid f = make_grid(N, 0.0);
    fill_rhs(f, N, dx);

    string prefix = "N" + to_string(N);

    ofstream history_file("history_" + prefix + ".csv");
    save_history_header(history_file);

    int iter = 0;
    ResidualInfo res_info{0.0, 0.0};

    for (iter = 0; iter < max_iter; iter++) {
        double max_change = gauss_seidel_step(u, f, N, dx);
        res_info = compute_residual_info(u, f, N, dx);
        double S = compute_energy(u, f, N, dx);

        save_history_row(history_file, iter, max_change, S, res_info);

        if (iter % snapshot_interval == 0) {
            Grid delta = compute_residual_grid(u, f, N, dx);
            string residual_snapshot_name =
                    "snapshot_residual_" + prefix + "_" + to_string(iter) + ".csv";
            save_scalar_field(delta, N, dx, residual_snapshot_name);
        }

        if (res_info.max_abs < tol_res) {
            break;
        }
    }

    Grid delta_final = compute_residual_grid(u, f, N, dx);
    save_scalar_field(delta_final, N, dx, "residual_final_" + prefix + ".csv");

    save_solution_with_error(u, N, dx, "solution_" + prefix + ".csv");

    double S_final = compute_energy(u, f, N, dx);
    double S_exact = -pow(3.0 * M_PI / 2.0, 2);
    ErrorInfo err_info = compute_error_info(u, N, dx);
    res_info = compute_residual_info(u, f, N, dx);

    RunResult result;
    result.N = N;
    result.dx = dx;
    result.iterations = iter + 1;
    result.S_final = S_final;
    result.S_error = fabs(S_final - S_exact);
    result.max_abs_res = res_info.max_abs;
    result.rms_res = res_info.rms;
    result.max_abs_error = err_info.max_abs;
    result.rms_error = err_info.rms;

    return result;
}

RunResult solve_for_N_with_omega(int N, double omega, double tol_res, int max_iter) {
    double dx = 3.0 / N;

    Grid u = make_grid(N, 0.0);
    Grid f = make_grid(N, 0.0);
    fill_rhs(f, N, dx);

    int iter = 0;
    ResidualInfo res_info{0.0, 0.0};

    for (iter = 0; iter < max_iter; iter++) {
        sor_step(u, f, N, dx, omega);
        res_info = compute_residual_info(u, f, N, dx);

        if (res_info.max_abs < tol_res) {
            break;
        }
    }

    double S_final = compute_energy(u, f, N, dx);
    double S_exact = -pow(3.0 * M_PI / 2.0, 2);
    ErrorInfo err_info = compute_error_info(u, N, dx);

    RunResult result;
    result.N = N;
    result.dx = dx;
    result.omega = omega;
    result.iterations = iter + 1;
    result.S_final = S_final;
    result.S_error = fabs(S_final - S_exact);
    result.max_abs_res = res_info.max_abs;
    result.rms_res = res_info.rms;
    result.max_abs_error = err_info.max_abs;
    result.rms_error = err_info.rms;

    return result;
}

double find_optimal_omega(int N, double tol_res, int max_iter, const string& filename) {
    ofstream file(filename);
    file << "omega,iterations,max_abs_res,S_error,max_abs_error\n";

    double best_omega = 1.0;
    int best_iterations = max_iter + 1;

    for (double omega = 1.00; omega <= 1.95; omega += 0.05) {
        RunResult r = solve_for_N_with_omega(N, omega, tol_res, max_iter);

        file << omega << ","
             << r.iterations << ","
             << r.max_abs_res << ","
             << r.S_error << ","
             << r.max_abs_error << "\n";

        if (r.max_abs_res < tol_res && r.iterations < best_iterations) {
            best_iterations = r.iterations;
            best_omega = omega;
        }
    }

    return best_omega;
}

RunResult solve_for_N_full_output(int N,
                                  double omega,
                                  double tol_res,
                                  int max_iter,
                                  int snapshot_interval) {
    double dx = 3.0 / N;

    Grid u = make_grid(N, 0.0);
    Grid f = make_grid(N, 0.0);
    fill_rhs(f, N, dx);

    string prefix = "SOR_N" + to_string(N) + "_w" + to_string(omega);

    ofstream history_file("history_" + prefix + ".csv");
    save_history_header(history_file);

    int iter = 0;
    ResidualInfo res_info{0.0, 0.0};

    for (iter = 0; iter < max_iter; iter++) {
        double max_change = sor_step(u, f, N, dx, omega);
        res_info = compute_residual_info(u, f, N, dx);
        double S = compute_energy(u, f, N, dx);

        save_history_row(history_file, iter, max_change, S, res_info);

        if (iter % snapshot_interval == 0) {
            Grid delta = compute_residual_grid(u, f, N, dx);
            string residual_snapshot_name =
                    "snapshot_residual_" + prefix + "_" + to_string(iter) + ".csv";
            save_scalar_field(delta, N, dx, residual_snapshot_name);
        }

        if (res_info.max_abs < tol_res) {
            break;
        }
    }

    Grid delta_final = compute_residual_grid(u, f, N, dx);
    save_scalar_field(delta_final, N, dx, "residual_final_" + prefix + ".csv");

    save_solution_with_error(u, N, dx, "solution_" + prefix + ".csv");

    double S_final = compute_energy(u, f, N, dx);
    double S_exact = -pow(3.0 * M_PI / 2.0, 2);
    ErrorInfo err_info = compute_error_info(u, N, dx);
    res_info = compute_residual_info(u, f, N, dx);

    RunResult result;
    result.N = N;
    result.dx = dx;
    result.omega = omega;
    result.iterations = iter + 1;
    result.S_final = S_final;
    result.S_error = fabs(S_final - S_exact);
    result.max_abs_res = res_info.max_abs;
    result.rms_res = res_info.rms;
    result.max_abs_error = err_info.max_abs;
    result.rms_error = err_info.rms;

    return result;
}

void save_summary_header(ofstream& file) {
    file << "N,dx,iterations,S_final,S_error,max_abs_res,rms_res,max_abs_error,rms_error\n";
}

void save_summary_row(ofstream& file, const RunResult& r) {
    file << r.N << ","
         << r.dx << ","
         << r.iterations << ","
         << r.S_final << ","
         << r.S_error << ","
         << r.max_abs_res << ","
         << r.rms_res << ","
         << r.max_abs_error << ","
         << r.rms_error << "\n";
}

Grid interpolate_to_finer(const Grid& u_coarse, int N_coarse) {
    int N_fine = 2 * N_coarse;
    Grid u_fine = make_grid(N_fine, 0.0);

    for (int i = 0; i <= N_coarse; i++) {
        for (int j = 0; j <= N_coarse; j++) {

            u_fine[2*i][2*j] = u_coarse[i][j];

            if (i < N_coarse) {
                u_fine[2*i+1][2*j] =
                        0.5 * (u_coarse[i][j] + u_coarse[i+1][j]);
            }

            if (j < N_coarse) {
                u_fine[2*i][2*j+1] =
                        0.5 * (u_coarse[i][j] + u_coarse[i][j+1]);
            }

            if (i < N_coarse && j < N_coarse) {
                u_fine[2*i+1][2*j+1] =
                        0.25 * (
                                u_coarse[i][j] + u_coarse[i+1][j] +
                                u_coarse[i][j+1] + u_coarse[i+1][j+1]
                        );
            }
        }
    }

    return u_fine;
}

void multilevel_solver(int n_final, double omega, double tol_res) {

    int k0 = n_final - 5;
    int N = pow(2, k0);

    Grid u = make_grid(N, 0.0);
    Grid f = make_grid(N, 0.0);

    int total_iterations = 0;

    ofstream file("multilevel_history_2.csv");
    file << "total_iter,N,S\n";

    for (int level = k0; level <= n_final; level++) {

        N = pow(2, level);
        double dx = 3.0 / N;

        cout << "Level N = " << N << endl;

        if (level > k0) {
            u = interpolate_to_finer(u, N / 2);
        }

        f = make_grid(N, 0.0);
        fill_rhs(f, N, dx);

        int iter = 0;

        while (true) {
            sor_step(u, f, N, dx, omega);

            auto res = compute_residual_info(u, f, N, dx);

            iter++;
            total_iterations++;

            double S = compute_energy(u, f, N, dx);

            file << total_iterations << ","
                 << N << ","
                 << S << "\n";

            if (res.max_abs < tol_res) break;
        }

        cout << "  iterations on this level: " << iter << endl;
    }

    cout << "Total iterations (all levels): " << total_iterations << endl;
}

int main() {
//    const vector<int> Ns = {32, 64, 128};
//
//    const double tol_res = 1e-8;
//    const int max_iter = 50000;
//    const int snapshot_interval = 200;
//
////    ofstream summary_file("refinement_summary.csv");
////    summary_file << fixed << setprecision(12);
////    save_summary_header(summary_file);
////
////    for (int N : Ns) {
////        cout << "Running N = " << N << " ..." << endl;
////        RunResult result = solve_for_N(N, tol_res, max_iter, snapshot_interval);
////        save_summary_row(summary_file, result);
////
////        cout << "  iterations   = " << result.iterations << "\n";
////        cout << "  S_final      = " << result.S_final << "\n";
////        cout << "  S_error      = " << result.S_error << "\n";
////        cout << "  max_abs_res  = " << result.max_abs_res << "\n";
////        cout << "  max_abs_err  = " << result.max_abs_error << "\n";
////        cout << endl;
////    }
////
////    cout << "All computations finished." << endl;
//
//    ofstream summary("sor_summary.csv");
//    summary << "N,omega_opt,iterations,S_final,S_error,max_abs_res,rms_res,max_abs_error,rms_error\n";
//
//    for (int N : Ns) {
//        cout << "Searching optimal omega for N = " << N << "...\n";
//
//        string omega_file = "omega_search_N" + to_string(N) + ".csv";
//        double omega_opt = find_optimal_omega(N, tol_res, max_iter, omega_file);
//
//        cout << "Optimal omega for N = " << N << " is " << omega_opt << "\n";
//
//        RunResult r = solve_for_N_full_output(N, omega_opt, tol_res, max_iter, snapshot_interval);
//
//        summary << r.N << ","
//                << omega_opt << ","
//                << r.iterations << ","
//                << r.S_final << ","
//                << r.S_error << ","
//                << r.max_abs_res << ","
//                << r.rms_res << ","
//                << r.max_abs_error << ","
//                << r.rms_error << "\n";
//    }
    int n_final = 9512;
    double tol_res = 1e-8;
    double omega = 1.9;

    cout << "Running multilevel grid refinement...\n";
    cout << "Final grid N = " << pow(2, n_final) << endl;

    multilevel_solver(n_final, omega, tol_res);

    cout << "Done.\n";

    return 0;
}