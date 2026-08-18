#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

int index(int i, int j, int N)
{
    return i * N + j;
}

void normalize(std::vector<double>& psi, int N, double dx, double dy)
{
    double sum = 0.0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double value = psi[index(i, j, N)];
            sum += value * value * dx * dy;
        }
    }

    double norm = std::sqrt(sum);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            psi[index(i, j, N)] /= norm;
        }
    }
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

void imaginaryTimeStep(
        std::vector<double>& psi,
        std::vector<double>& Hpsi,
        const std::vector<double>& V,
        int N,
        double dx,
        double dy,
        double alpha
)
{
    applyHamiltonian(psi, Hpsi, V, N, dx, dy);

    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            int id = index(i, j, N);
            psi[id] = psi[id] - alpha * Hpsi[id];
        }
    }

    for (int i = 0; i < N; i++)
    {
        psi[index(i, 0, N)] = 0.0;
        psi[index(i, N - 1, N)] = 0.0;
        psi[index(0, i, N)] = 0.0;
        psi[index(N - 1, i, N)] = 0.0;
    }

    normalize(psi, N, dx, dy);
}

void initializePotential(
        std::vector<double>& V,
        int N,
        double dx,
        double dy
)
{
    double omega_x = 1.0;
    double omega_y = 2.0;

    double x_min = -2.0;
    double y_min = -2.0;

    for (int i = 0; i < N; i++)
    {
        double x = x_min + i * dx;

        for (int j = 0; j < N; j++)
        {
            double y = y_min + j * dy;

            V[index(i, j, N)] =
                    0.5 * (
                            omega_x * omega_x * x * x
                            + omega_y * omega_y * y * y
                    );
        }
    }
}

void initializeRandomPsi(
        std::vector<double>& psi,
        int N
)
{
    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int id = index(i, j, N);

            if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
            {
                psi[id] = 0.0;
            }
            else
            {
                psi[id] = dist(gen);
            }
        }
    }
}

void runGroundState(
        int N,
        double alphaFactor,
        int maxIter,
        double tolerance
)
{
    double L = 4.0;
    double dx = L / (N - 1);
    double dy = dx;

    double alpha_c = dx * dx / 2.0;
    double alpha = alphaFactor * alpha_c;

    std::vector<double> psi(N * N);
    std::vector<double> Hpsi(N * N);
    std::vector<double> V(N * N);

    initializePotential(V, N, dx, dy);

    initializeRandomPsi(psi, N);

    normalize(psi, N, dx, dy);

    std::string filename =
            "energy_alpha_" + std::to_string(alphaFactor) + ".dat";

    std::ofstream file(filename);

    double previousEnergy = 1e100;

    for (int iter = 0; iter < maxIter; iter++)
    {
        imaginaryTimeStep(psi, Hpsi, V, N, dx, dy, alpha);

        applyHamiltonian(psi, Hpsi, V, N, dx, dy);
        double energy = computeEnergy(psi, Hpsi, N, dx, dy);

        file << iter << " " << energy << "\n";

        if (std::abs(energy - previousEnergy) < tolerance)
        {
            std::cout
                    << "alpha factor = " << alphaFactor
                    << ", converged after " << iter
                    << " iterations, E = " << energy
                    << std::endl;

            break;
        }

        if (!std::isfinite(energy))
        {
            std::cout
                    << "alpha factor = " << alphaFactor
                    << " became unstable."
                    << std::endl;

            break;
        }

        previousEnergy = energy;
    }
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
        const std::string& filename
)
{
    std::ofstream file(filename);

    double x_min = -2.0;
    double y_min = -2.0;

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

std::vector<double> findState(
        int stateIndex,
        int N,
        double alpha,
        int maxIter,
        double tolerance,
        const std::vector<double>& V,
        const std::vector<std::vector<double>>& previousStates
)
{
    double L = 4.0;
    double dx = L / (N - 1);
    double dy = dx;

    std::vector<double> psi(N * N);
    std::vector<double> Hpsi(N * N);

    initializeRandomPsi(psi, N);

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

    saveWavefunction(psi, N, dx, dy, psiFilename);

    return psi;
}

int main()
{
    int N = 200;

    double L = 4.0;
    double dx = L / (N - 1);
    double dy = dx;

    double alpha_c = dx * dx / 2.0;
    double alpha = 0.9 * alpha_c;

    int maxIter = 50000;
    double tolerance = 1e-10;

    std::vector<double> alphaFactors = {
            0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.05
    };

    for (double factor : alphaFactors)
    {
        runGroundState(N, factor, maxIter, tolerance);
    }

    std::vector<double> V(N * N);
    initializePotential(V, N, dx, dy);

    std::vector<std::vector<double>> states;

    int numberOfStates = 4;

    for (int k = 0; k < numberOfStates; k++)
    {
        std::cout << "\nFinding state " << k << std::endl;

        std::vector<double> psi_k =
                findState(
                        k,
                        N,
                        alpha,
                        maxIter,
                        tolerance,
                        V,
                        states
                );

        states.push_back(psi_k);

        std::cout << "Stored state " << k << std::endl;
    }

    std::cout << "\nExact energies:" << std::endl;
    std::cout << "E(0,0) = 1.5" << std::endl;
    std::cout << "E(1,0) = 2.5" << std::endl;
    std::cout << "E(2,0) = 3.5" << std::endl;
    std::cout << "E(0,1) = 3.5" << std::endl;

    return 0;
}