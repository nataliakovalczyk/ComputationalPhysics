#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
// #include <lapacke.h>
#define T_0 293
#define g 9.81
#define Alpha 2.5 // constant for air

double collision_correction(double Delta_t, double x, double y, double v_0, double Theta_0, double D, double a, double m, const std::string& filename) {
    x = 0.0;
    y = 0.0;
    double t = 0.0;

    std::ofstream CollisionData(filename);
    if (!CollisionData) {
        std::cerr << "Failed to open " << filename << "\n";
        return -1.0;
    }

    double v_x = v_0 * std::cos(Theta_0);
    double v_y = v_0 * std::sin(Theta_0);
    bool RUN = true;
    double x_max = 0.0;

    do {
        double x_old = x;
        double y_old = y;
        double t_old = t;

        double v = std::sqrt(v_x*v_x + v_y*v_y);
        double F_x = -D * v * v_x * std::pow(1 - (a*y)/T_0,Alpha);
        double F_y = -m * g - D * v * v_y * std::pow(1 - (a*y)/T_0,Alpha);
        x += Delta_t * v_x;
        y += Delta_t * v_y;
        v_x += (Delta_t * F_x) / m;
        v_y += (Delta_t * F_y) / m;
        t += Delta_t;

        if (x > x_max) x_max = x;

        if (y < 0.0) {
            double r = y_old / (y_old - y);
            x = x_old + (x - x_old) * r;
            y = y_old + (y - y_old) * r;
            t = t - (1 - r) * Delta_t;
            if (x > x_max) x_max = x;
            RUN = false;
        }

        CollisionData << t << " " << x << " " << y << std::endl;
    } while (RUN);

    CollisionData.close();
    return x_max; // needed for task 3
}

int main() {
    double x = 0.0;
    double y = 0.0;
    double v_0 = 100.0;
    double Theta_0 = 45.0 * M_PI / 180.0; // 45 degrees in radians
    double D = 0.0;
    double a = 0.0;
    double m = 1.0;

    // derived analytical formulae for the range xmax and the time of flight tmax
    double R_analytic = v_0*v_0*std::sin(2*Theta_0)/g;
    double t_of_flight = 2.0 * v_0 * std::sin(Theta_0) / g;

    // simulating the trajectory without the drag force
    int n_values[] = {10, 20, 50, 100, 200, 500};
    std::ofstream error_file("global_error.txt");
    std::cout << "Analytical range: " << R_analytic << " m\n";
    std::cout << "Analytical time of flight: " << t_of_flight << " s\n";

    for (int i = 0; i < 6; ++i) {
        int n = n_values[i];
        double Delta_t = t_of_flight / n;
        std::string filename = "trajectory_n" + std::to_string(n) + ".txt";
        double x_num_max = collision_correction(Delta_t, x, y, v_0, Theta_0, D, a, m, filename);
        double Eglob = R_analytic - x_num_max;
        error_file << Delta_t << " " << Eglob << std::endl;
        std::cout << "n = " << n << ", Delta_t = " << Delta_t << ", x_num_max = " << x_num_max << ", Eglob = " << Eglob << std::endl;
    }
    error_file.close();

    // enabling the drag force but turn oﬀ the altitude amendment assuming a=0.0
    double drag_D_values[] = {0.0, 1e-4, 2e-4, 5e-4, 1e-3};
    for (int i = 0; i < 5; ++i) {
        double D_drag = drag_D_values[i];
        std::string filename = "drag_D" + std::to_string(D_drag) + ".txt";
        double x_num_max = collision_correction(0.07, x, y, v_0, Theta_0, D_drag, a, m, filename);
        std::cout << "D = " << D_drag << ", x_num_max = " << x_num_max << std::endl;
    }
    std::cout << "Data written to drag_D*.txt \n";

    // range of projectile
    double scan_D_values[] = {0.0, 1e-3, 2e-3};
    double v_0_scan = 100.0;
    double m_scan = 1.0;
    for (int d = 0; d < 3; ++d) {
        double D_scan = scan_D_values[d];
        std::string filename = std::string("range_D") + (D_scan < 1e-6 ? "0.000" : (D_scan < 2e-3 ? "0.001" : "0.002")) + ".txt";
        std::ofstream range_file(filename);
        for (int deg = 15; deg <= 65; ++deg) {
            double theta_rad = deg * M_PI / 180.0;
            double range = collision_correction(0.07, x, y, v_0_scan, theta_rad, D_scan, a, m_scan, "temp.txt");
            range_file << deg << " " << range << std::endl;
        }
        range_file.close();
        std::cout << "Range data for D=" << D_scan << " written to " << filename << std::endl;
    }

    // drag force with altitude correction
    double m_art = 20.0;
    double v0_art = 700.0;
    double D_art = 1e-3;
    double a_values[] = {0.0, 6.5e-3};
    int alpha_degrees[] = {35, 45};
    for (int a_idx = 0; a_idx < 2; ++a_idx) {
        double a_art = a_values[a_idx];
        for (int ang_idx = 0; ang_idx < 2; ++ang_idx) {
            int alpha_deg = alpha_degrees[ang_idx];
            double theta_rad = alpha_deg * M_PI / 180.0;
            std::string filename = std::string("artillery_a") + (a_art == 0.0 ? "0" : "6.5e-3") + "_alpha" + std::to_string(alpha_deg) + ".txt";
            collision_correction(0.07, x, y, v0_art, theta_rad, D_art, a_art, m_art, filename);
            std::cout << "Artillery trajectory for a=" << a_art << ", alpha=" << alpha_deg << " written to " << filename << std::endl;
        }
    }

    return 0;
}