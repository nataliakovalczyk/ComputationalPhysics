#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

#define a_mercury 0.397098
#define T_mercury 0.240846
#define M_S 1.998e30
#define m 2.4e23
#define e 0.206
#define k_max 4

const double GM = 4.0 * M_PI * M_PI;

const double a[k_max] = {0.675694, -0.175604, -0.175604, 0.675694};
const double b[k_max] = {1.35121, -1.70241,1.35121, 0.0};

// Perihelion
const double r_min = a_mercury * (1.0 - e);
const double v_max = std::sqrt( GM * ((1.0 + e) / (a_mercury * (1.0 - e)))
                                * (1.0 + m / M_S) );

// Aphelion
const double r_max = a_mercury * (1.0 + e);
const double v_min = std::sqrt( GM * ((1.0 - e) / (a_mercury * (1.0 + e)))
                                * (1.0 + m / M_S) );

void simplectic_sim(double t_max, double dt, double alpha,
                    const std::string &filename) {

    double x = r_max;
    double y = 0.0;
    double v_x = 0.0;
    double v_y = v_min;

    int n = static_cast<int>(t_max / dt);
    double time = 0.0;

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open file " << filename << "\n";
        return;
    }

    for (int i = 0; i < n; i++) {
        double x_old = x;
        double y_old = y;
        double v_x_old = v_x;
        double v_y_old = v_y;

        double x_new = 0.0;
        double y_new = 0.0;
        double v_x_new = 0.0;
        double v_y_new = 0.0;

        for (int k = 0; k < k_max; k++) {
            x_new = x_old + a[k] * v_x_old * dt;
            y_new = y_old + a[k] * v_y_old * dt;
            double r_new = std::hypot(x_new, y_new);

            const double r2 = r_new * r_new;
            const double r3 = r2 * r_new;
            const double omega_new = (-4.0 * M_PI * M_PI / r3) * (1.0 + alpha / r2);

            v_x_new = v_x_old + b[k]* omega_new * x_new * dt;
            v_y_new = v_y_old + b[k]* omega_new * y_new * dt;

            x_old = x_new;
            y_old = y_new;
            v_x_old = v_x_new;
            v_y_old = v_y_new;
        }
        x = x_new;
        y = y_new;
        v_x = v_x_new;
        v_y = v_y_new;
        time += dt;

        file << x << "," << y << "\n";
    }
    std::cout << "Simulation complete. Output saved to: " << filename << "\n";
    file.close();
}

void find_extrema(const std::string &trajectory_file,const std::string &extrema_file,double dt){

    std::ifstream in(trajectory_file);

    std::vector<double> x, y;
    std::string line;

    while (std::getline(in, line)) {
        std::size_t comma = line.find(',');
        double X = std::stod(line.substr(0, comma));
        double Y = std::stod(line.substr(comma + 1));
        x.push_back(X);
        y.push_back(Y);
    }
    in.close();

    const std::size_t n = x.size();

    std::ofstream out(extrema_file);

    for (std::size_t i = 1; i + 1 < n; ++i) {
        const double r_prev = std::hypot(x[i-1], y[i-1]);
        const double r_now  = std::hypot(x[i],   y[i]);
        const double r_next = std::hypot(x[i+1], y[i+1]);

        const double t = static_cast<double>(i) * dt;

        if (r_now < r_prev && r_now < r_next) {
            out << "perihelion," << t << "," << x[i] << "," << y[i] << "," << r_now << "\n";
        } else if (r_now > r_prev && r_now > r_next) {
            out << "aphelion,"  << t << "," << x[i] << "," << y[i] << "," << r_now << "\n";
        }
    }
    out.close();

    std::cout << "Extrema saved to: " << extrema_file << "\n";
}

static inline double wrap_delta(double dtheta) {
    const double two_pi = 2.0 * M_PI;
    dtheta = std::fmod(dtheta + M_PI, two_pi);
    if (dtheta < 0.0) dtheta += two_pi;
    return dtheta - M_PI;
}

void sweep_alpha_omega(double t_max, double dt, const std::string &filename) {

    const int n_steps = static_cast<int>(t_max / dt);

    std::ofstream file(filename);
    file.setf(std::ios::scientific);
    file << std::setprecision(16);

    for (int j = 0; j <= 6; ++j) {
        const double alpha = 0.001 / std::pow(2.0, j);

        double x = r_max;
        double y = 0.0;
        double vx = 0.0;
        double vy = v_min;

        double time = 0.0;

        double r_prev = std::hypot(x, y);
        double th_prev = std::atan2(y, x);
        double t_prev = time;

        auto step = [&](double &x_, double &y_, double &vx_, double &vy_) {
            double x_old = x_, y_old = y_, vx_old = vx_, vy_old = vy_;
            double x_new = x_old, y_new = y_old, vx_new = vx_old, vy_new = vy_old;

            for (int k = 0; k < k_max; ++k) {
                x_new = x_old + a[k] * vx_old * dt;
                y_new = y_old + a[k] * vy_old * dt;
                const double r_new = std::hypot(x_new, y_new);

                const double r2 = r_new * r_new;
                const double r3 = r2 * r_new;
                const double acc_scale = (-4.0 * M_PI * M_PI / r3) * (1.0 + alpha / r2);

                vx_new = vx_old + b[k] * acc_scale * x_new * dt;
                vy_new = vy_old + b[k] * acc_scale * y_new * dt;

                x_old = x_new;
                y_old = y_new;
                vx_old = vx_new;
                vy_old = vy_new;
            }
            x_ = x_new; y_ = y_new; vx_ = vx_new; vy_ = vy_new;
        };

        step(x, y, vx, vy);
        time += dt;

        double r_now = std::hypot(x, y);
        double th_now = std::atan2(y, x);
        double t_now = time;

        int found = 0;
        double th1 = 0.0, th2 = 0.0;
        double t1 = 0.0, t2 = 0.0;

        for (int i = 2; i < n_steps; ++i) {
            step(x, y, vx, vy);
            time += dt;

            const double r_next = std::hypot(x, y);
            const double th_next = std::atan2(y, x);
            const double t_next = time;

            if (r_now < r_prev && r_now < r_next) {
                if (found == 0) {
                    th1 = th_now;
                    t1 = t_now;
                    found = 1;
                } else if (found == 1) {
                    th2 = th_now;
                    t2 = t_now;
                    found = 2;
                    break;
                }
            }

            r_prev = r_now; th_prev = th_now; t_prev = t_now;
            r_now = r_next; th_now = th_next; t_now = t_next;
        }

        if (found == 2 && t2 > t1) {
            const double dtheta = wrap_delta(th2 - th1);
            const double omega = dtheta / (t2 - t1);
            file << alpha << ", " << omega << "\n";
        }
    }

    std::cout << "Sweep complete. Output saved to: " << filename << "\n";
    file.close();
}

int main() {
    // first test simulation - no relativistic effects
    std::cout << "Running test simulation...\n";
    simplectic_sim(0.95 * T_mercury, 1.0e-4,  0.0, "test_no_relativistic.csv");

    // test of stability
    std::cout << "Running test simulation...\n";
    simplectic_sim(100 * T_mercury, 1.0e-4,  0.0, "test_of_stability.csv");

    // precession test
    std::cout << "Running precession test...\n";
    simplectic_sim(4.0 * T_mercury, 1.0e-4, 0.01, "precession_alpha001.csv");

    // find extrema
    std::cout << "Finding extrema...\n";
    find_extrema("precession_alpha001.csv","extrema_precession.csv",1.0e-4);

    // alpha and omega sweep
    std::cout <<"Sweeping for alpha and omega...\n";
    sweep_alpha_omega(3.0, 1e-5, "alpha_omega.csv");

    return 0;
}