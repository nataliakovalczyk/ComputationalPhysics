
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <complex>

using Complex = std::complex<double>;

int index(int i, int j, int N)
{
    return i * N + j;
}

void normalize(std::vector<double>& psi, int N, double dx, double dy)
{
    double sum = 0.0;

    for (int id = 0; id < N * N; id++)
        sum += psi[id] * psi[id] * dx * dy;

    double norm = std::sqrt(sum);

    if (norm == 0.0)
    {
        std::cerr << "Error: zero norm in normalize().\n";
        return;
    }

    for (int id = 0; id < N * N; id++)
        psi[id] /= norm;
}

void initializeStateGuess(
        std::vector<double>& psi,
        int stateIndex,
        int N,
        double dx,
        double dy,
        double x_min,
        double y_min
)
{
    for (int i = 0; i < N; i++)
    {
        double x = x_min + i * dx;

        for (int j = 0; j < N; j++)
        {
            double y = y_min + j * dy;
            int id = index(i, j, N);

            if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
            {
                psi[id] = 0.0;
                continue;
            }

            double gaussian = std::exp(-0.5 * (x * x + y * y));

            if (stateIndex == 0)
                psi[id] = gaussian;
            else if (stateIndex == 1)
                psi[id] = x * gaussian;
            else if (stateIndex == 2)
                psi[id] = y * gaussian;
            else
                psi[id] = gaussian;
        }
    }

    normalize(psi, N, dx, dy);
}

void applyHamiltonian(
        const std::vector<double>& psi,
        std::vector<double>& Hpsi,
        const std::vector<double>& V,
        int N,
        double dx,
        double dy
)
{
    std::fill(Hpsi.begin(), Hpsi.end(), 0.0);

    double dx2 = dx * dx;
    double dy2 = dy * dy;

    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            int id = index(i, j, N);

            double laplacian =
                    (psi[index(i + 1, j, N)]
                     + psi[index(i - 1, j, N)]
                     - 2.0 * psi[id]) / dx2
                    + (psi[index(i, j + 1, N)]
                       + psi[index(i, j - 1, N)]
                       - 2.0 * psi[id]) / dy2;

            Hpsi[id] = -0.5 * laplacian + V[id] * psi[id];
        }
    }
}

void applyHamiltonianComplex(
        const std::vector<Complex>& psi,
        std::vector<Complex>& Hpsi,
        const std::vector<double>& V,
        int N,
        double dx,
        double dy
) {
    std::fill(Hpsi.begin(), Hpsi.end(), Complex(0.0, 0.0));

    double dx2 = dx * dx;
    double dy2 = dy * dy;

    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            int id = index(i, j, N);

            Complex laplacian =
                    (psi[index(i + 1, j, N)]
                     + psi[index(i - 1, j, N)]
                     - 2.0 * psi[id]) / dx2
                    + (psi[index(i, j + 1, N)]
                       + psi[index(i, j - 1, N)]
                       - 2.0 * psi[id]) / dy2;

            Hpsi[id] = -0.5 * laplacian + V[id] * psi[id];
        }
    }
}

void initializePotential(
        std::vector<double>& V,
        int N,
        double dx,
        double dy,
        double x_min,
        double y_min,
        double omega_x,
        double omega_y
)
{
    for (int i = 0; i < N; i++)
    {
        double x = x_min + i * dx;

        for (int j = 0; j < N; j++)
        {
            double y = y_min + j * dy;

            V[index(i, j, N)] =
                    0.5 * (omega_x * omega_x * x * x
                           + omega_y * omega_y * y * y);
        }
    }
}

double computeEnergy(
        const std::vector<double>& psi,
        const std::vector<double>& Hpsi,
        int N,
        double dx,
        double dy
)
{
    double E = 0.0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int id = index(i, j, N);
            E += psi[id] * Hpsi[id] * dx * dy;
        }
    }

    return E;
}

double overlap(
        const std::vector<double>& psi1,
        const std::vector<double>& psi2,
        int N,
        double dx,
        double dy
)
{
    double result = 0.0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int id = index(i, j, N);
            result += psi1[id] * psi2[id] * dx * dy;
        }
    }

    return result;
}

void orthogonalize(
        std::vector<double>& psi,
        const std::vector<std::vector<double>>& previousStates,
        int N,
        double dx,
        double dy
)
{
    for (const auto& state : previousStates)
    {
        double c = overlap(state, psi, N, dx, dy);

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int id = index(i, j, N);
                psi[id] -= c * state[id];
            }
        }
    }

    normalize(psi, N, dx, dy);
}

void saveWavefunction(
        const std::vector<double>& psi,
        int N,
        double dx,
        double dy,
        double x_min,
        double y_min,
        const std::string& filename
)
{
    std::ofstream file(filename);

    for (int i = 0; i < N; i++)
    {
        double x = x_min + i * dx;

        for (int j = 0; j < N; j++)
        {
            double y = y_min + j * dy;
            int id = index(i, j, N);

            file << x << " " << y << " " << psi[id] << "\n";
        }

        file << "\n";
    }
}

void crankNicolsonStep(
        std::vector<Complex>& psi,
        const std::vector<double>& V,
        int N,
        double dx,
        double dy,
        double dt,
        int iterations
)
{
    std::vector<Complex> H_old(N * N);
    std::vector<Complex> H_iter(N * N);
    std::vector<Complex> psi_old = psi;
    std::vector<Complex> psi_new = psi;
    std::vector<Complex> psi_next = psi;

    applyHamiltonianComplex(psi_old, H_old, V, N, dx, dy);

    Complex factor(0.0, -dt / 2.0);

    for (int nu = 0; nu < iterations; nu++)
    {
        applyHamiltonianComplex(psi_new, H_iter, V, N, dx, dy);

        psi_next = psi_old;

        for (int i = 1; i < N - 1; i++)
        {
            for (int j = 1; j < N - 1; j++)
            {
                int id = index(i, j, N);

                psi_next[id] =
                        psi_old[id]
                        + factor * (H_old[id] + H_iter[id]);
            }
        }

        for (int i = 0; i < N; i++)
        {
            psi_next[index(i, 0, N)] = Complex(0.0, 0.0);
            psi_next[index(i, N - 1, N)] = Complex(0.0, 0.0);
            psi_next[index(0, i, N)] = Complex(0.0, 0.0);
            psi_next[index(N - 1, i, N)] = Complex(0.0, 0.0);
        }

        psi_new = psi_next;
    }

    psi = psi_new;
}

double normComplex(const std::vector<Complex>& psi, int N, double dx, double dy)
{
    double sum = 0.0;

    for (int id = 0; id < N * N; id++)
        sum += std::norm(psi[id]) * dx * dy;

    return sum;
}

void normalizeComplex(std::vector<Complex>& psi, int N, double dx, double dy)
{
    double n = std::sqrt(normComplex(psi, N, dx, dy));

    if (n == 0.0)
    {
        std::cerr << "Error: zero norm in normalizeComplex().\n";
        return;
    }

    for (int id = 0; id < N * N; id++)
        psi[id] /= n;
}

double averageX(
        const std::vector<Complex>& psi,
        int N,
        double dx,
        double dy,
        double x_min
)
{
    double result = 0.0;
    double norm = normComplex(psi, N, dx, dy);

    for (int i = 0; i < N; i++)
    {
        double x = x_min + i * dx;

        for (int j = 0; j < N; j++)
        {
            int id = index(i, j, N);
            result += x * std::norm(psi[id]) * dx * dy;
        }
    }

    return result / norm;
}

double averageY(
        const std::vector<Complex>& psi,
        int N,
        double dx,
        double dy,
        double y_min
)
{
    double result = 0.0;
    double norm = normComplex(psi, N, dx, dy);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double y = y_min + j * dy;
            int id = index(i, j, N);
            result += y * std::norm(psi[id]) * dx * dy;
        }
    }

    return result / norm;
}

double computeEnergyComplex(
        const std::vector<Complex>& psi,
        const std::vector<Complex>& Hpsi,
        int N,
        double dx,
        double dy
)
{
    Complex E = 0.0;
    double norm = normComplex(psi, N, dx, dy);

    for (int id = 0; id < N * N; id++)
        E += std::conj(psi[id]) * Hpsi[id] * dx * dy;

    return E.real() / norm;
}



std::vector<double> findState(
        int stateIndex,
        int N,
        double alpha,
        int maxIter,
        double tolerance,
        const std::vector<double>& V,
        const std::vector<std::vector<double>>& previousStates,
        double dx,
        double dy,
        double x_min,
        double y_min
)
{
    std::vector<double> psi(N * N);
    std::vector<double> Hpsi(N * N);

    initializeStateGuess(psi, stateIndex, N, dx, dy, x_min, y_min);

    if (!previousStates.empty())
    {
        orthogonalize(psi, previousStates, N, dx, dy);
    }
    else
    {
        normalize(psi, N, dx, dy);
    }

    std::string energyFilename =
            "energy_state_" + std::to_string(stateIndex) + ".dat";

    std::ofstream energyFile(energyFilename);

    double previousEnergy = 1e100;
    double energy = 0.0;

    for (int iter = 0; iter < maxIter; iter++)
    {
        applyHamiltonian(psi, Hpsi, V, N, dx, dy);

        for (int i = 1; i < N - 1; i++)
        {
            for (int j = 1; j < N - 1; j++)
            {
                int id = index(i, j, N);
                psi[id] -= alpha * Hpsi[id];
            }
        }

        // Boundary condition: psi = 0 on the edges
        for (int i = 0; i < N; i++)
        {
            psi[index(i, 0, N)] = 0.0;
            psi[index(i, N - 1, N)] = 0.0;
            psi[index(0, i, N)] = 0.0;
            psi[index(N - 1, i, N)] = 0.0;
        }

        if (!previousStates.empty())
        {
            orthogonalize(psi, previousStates, N, dx, dy);
        }
        else
        {
            normalize(psi, N, dx, dy);
        }

        applyHamiltonian(psi, Hpsi, V, N, dx, dy);
        energy = computeEnergy(psi, Hpsi, N, dx, dy);

        energyFile << iter << " " << energy << "\n";

        if (iter % 1000 == 0)
        {
            std::cout
                    << "State " << stateIndex
                    << ", iter " << iter
                    << ", E = " << energy
                    << std::endl;
        }

        if (!std::isfinite(energy))
        {
            std::cout
                    << "State " << stateIndex
                    << " became unstable."
                    << std::endl;
            break;
        }

        if (std::abs(energy - previousEnergy) < tolerance)
        {
            std::cout
                    << "State " << stateIndex
                    << " converged after " << iter
                    << " iterations, E = " << energy
                    << std::endl;
            break;
        }

        previousEnergy = energy;
    }

    std::string psiFilename =
            "psi_state_" + std::to_string(stateIndex) + ".dat";

    saveWavefunction(psi, N, dx, dy, x_min, y_min, psiFilename);

    return psi;
}

void runTimeEvolution(
        std::vector<Complex> psi,
        const std::vector<double>& V,
        int N,
        double dx,
        double dy,
        double dt,
        int cnIterations,
        double tMax,
        double x_min,
        double y_min,
        const std::string& filename
)
{
    normalizeComplex(psi, N, dx, dy);

    std::vector<Complex> Hpsi(N * N);

    std::ofstream file(filename);
    file << "t,norm,energy,x_avg,y_avg\n";

    int steps = static_cast<int>(tMax / dt);

    for (int step = 0; step <= steps; step++)
    {
        double t = step * dt;

        applyHamiltonianComplex(psi, Hpsi, V, N, dx, dy);

        double norm = normComplex(psi, N, dx, dy);
        double energy = computeEnergyComplex(psi, Hpsi, N, dx, dy);
        double xavg = averageX(psi, N, dx, dy, x_min);
        double yavg = averageY(psi, N, dx, dy, y_min);

        file << t << ","
             << norm << ","
             << energy << ","
             << xavg << ","
             << yavg << "\n";

        crankNicolsonStep(psi, V, N, dx, dy, dt, cnIterations);
    }

    std::cout << "Saved: " << filename << std::endl;
}

int main()
{
    int N = 65;

    double omega_x = 1.0;
    double omega_y = 1.001;

    double L = 8.0;
    double x_min = -L / 2.0;
    double y_min = -L / 2.0;

    double dx = L / (N - 1);
    double dy = dx;

    double dt = 0.001;
    int cnIterations = 5;
    double tMax = 4.0 * M_PI;

    std::vector<double> V(N * N);
    initializePotential(V, N, dx, dy, x_min, y_min, omega_x, omega_y);

    std::vector<Complex> psi(N * N);
    std::vector<Complex> Hpsi(N * N);

    double alpha_c = dx * dx / 2.0;
    double alpha = 0.9 * alpha_c;

    int maxIter = 50000;
    double tolerance = 1e-10;

    std::vector<std::vector<double>> states;

    for (int k = 0; k < 3; k++)
    {
        std::cout << "Finding state " << k << std::endl;

        std::vector<double> psi_k =
                findState(
                        k,
                        N,
                        alpha,
                        maxIter,
                        tolerance,
                        V,
                        states,
                        dx,
                        dy,
                        x_min,
                        y_min
                );

        states.push_back(psi_k);
    }

    std::vector<Complex> psi_initial(N * N);
    std::vector<Complex> psi_reverse(N * N);
    std::vector<Complex> psi_x(N * N);
    std::vector<Complex> psi_y(N * N);
    std::vector<Complex> psi_eigen0(N * N);
    std::vector<Complex> psi_eigen1(N * N);
    std::vector<Complex> psi_eigen2(N * N);

    for (int id = 0; id < N * N; id++)
    {
        // Original rotating packet
        psi_initial[id] =
                (states[0][id]
                 + states[1][id]
                 + Complex(0.0, 1.0) * states[2][id]) / std::sqrt(3.0);

        // Opposite rotation
        psi_reverse[id] =
                (states[0][id]
                 + states[1][id]
                 - Complex(0.0, 1.0) * states[2][id]) / std::sqrt(3.0);

        // Pure x oscillation
        psi_x[id] =
                (states[0][id]
                 + states[1][id]) / std::sqrt(2.0);

        // Pure y oscillation
        psi_y[id] =
                (states[0][id]
                 + states[2][id]) / std::sqrt(2.0);

        // Eigenstate initial conditions
        psi_eigen0[id] = states[0][id];
        psi_eigen1[id] = states[1][id];
        psi_eigen2[id] = states[2][id];
    }

    runTimeEvolution(
            psi_initial, V, N, dx, dy, dt, cnIterations, tMax,
            x_min, y_min,
            "time_observables_original.csv"
    );

    runTimeEvolution(
            psi_reverse, V, N, dx, dy, dt, cnIterations, tMax,
            x_min, y_min,
            "time_observables_reverse.csv"
    );

    runTimeEvolution(
            psi_x, V, N, dx, dy, dt, cnIterations, tMax,
            x_min, y_min,
            "time_observables_x_oscillation.csv"
    );

    runTimeEvolution(
            psi_y, V, N, dx, dy, dt, cnIterations, tMax,
            x_min, y_min,
            "time_observables_y_oscillation.csv"
    );

    runTimeEvolution(
            psi_eigen0, V, N, dx, dy, dt, cnIterations, tMax,
            x_min, y_min,
            "time_observables_eigenstate_0.csv"
    );

    runTimeEvolution(
            psi_eigen1, V, N, dx, dy, dt, cnIterations, tMax,
            x_min, y_min,
            "time_observables_eigenstate_1.csv"
    );

    runTimeEvolution(
            psi_eigen2, V, N, dx, dy, dt, cnIterations, tMax,
            x_min, y_min,
            "time_observables_eigenstate_2.csv"
    );

    std::cout << "All time evolutions finished.\n";

    return 0;
}