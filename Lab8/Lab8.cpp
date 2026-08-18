#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>

using namespace std;

const double epsilon = 1.0;
const double sigma   = 1.0;
const double mass    = 1.0;
const double dt = 0.002;

class Particle {
public:
    double x, y;
    double vx, vy;
    double vxp, vyp;
    double fx, fy;
    double ax, ay;

    Particle(double x_=0, double y_=0, double vx_=0, double vy_=0)
            : x(x_), y(y_), vx(vx_), vy(vy_), vxp(0.0), vyp(0.0),
              fx(0.0), fy(0.0), ax(0.0), ay(0.0) {}

    void resetForce() { fx = 0.0; fy = 0.0; }
};

inline double wrap(double x, double L) {
    // maps x into [0, L)
    x = fmod(x, L);
    if (x < 0) x += L;
    return x;
}

inline double minImage(double dx, double L) {
    if (dx >  0.5 * L) dx -= L;
    if (dx <= -0.5 * L) dx += L;
    return dx;
}

void applyPeriodicBoundary(vector<Particle>& particles, double L) {
    for (auto& p : particles) {
        p.x = wrap(p.x, L);
        p.y = wrap(p.y, L);
    }
}

struct ForceEnergy {
    double epot = 0.0;
};

inline double lj_U(double r2) {
    double inv_r2 = 1.0 / r2;
    double sr2 = inv_r2;
    double sr6 = sr2 * sr2 * sr2;
    double sr12 = sr6 * sr6;
    return 4.0 * (sr12 - sr6);
}

inline double lj_dUdr(double r) {
    double inv_r = 1.0 / r;
    double inv_r2 = inv_r * inv_r;
    double inv_r6 = inv_r2 * inv_r2 * inv_r2;
    double inv_r7 = inv_r6 * inv_r;
    double inv_r13 = inv_r7 * inv_r6;
    return 4.0 * (-12.0 * inv_r13 + 6.0 * inv_r7);
}


void computeForces(vector<Particle>& particles, double L, double rcut, ForceEnergy& fe) {
    const double rcut2 = rcut * rcut;

    const double Uc     = lj_U(rcut2);
    const double dUdr_c = lj_dUdr(rcut);

    for (auto& p : particles) p.resetForce();
    fe.epot = 0.0;

    const int N = (int)particles.size();

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {

            double dx = minImage(particles[j].x - particles[i].x, L);
            double dy = minImage(particles[j].y - particles[i].y, L);

            double r2 = dx*dx + dy*dy;
            if (r2 >= rcut2) continue;
            if (r2 < 1e-12) continue;

            double r = sqrt(r2);

            fe.epot += (lj_U(r2) - Uc - (r - rcut) * dUdr_c);

            double dUdr = lj_dUdr(r);
            double dUdr_shift = dUdr - dUdr_c;

            double coef = dUdr_shift / r;

            double fx = coef * dx;
            double fy = coef * dy;

            particles[i].fx += fx;
            particles[i].fy += fy;
            particles[j].fx -= fx;
            particles[j].fy -= fy;
        }
    }

    for (auto& p : particles) {
        p.ax = p.fx / mass;
        p.ay = p.fy / mass;
    }
}



double kineticEnergy(const vector<Particle>& particles) {
    double ek = 0.0;
    for (const auto& p : particles) {
        ek += 0.5 * mass * (p.vx*p.vx + p.vy*p.vy);
    }
    return ek;
}

double instantaneousTemperature(const vector<Particle>& particles) {
    const int N = (int)particles.size();
    double ek = kineticEnergy(particles);
    return ek / (double)N;
}

void velocityVerletStep(vector<Particle>& particles, double L, double rcut, ForceEnergy& fe) {
    for (auto& p : particles) {
        p.vxp = p.vx + 0.5 * dt * p.ax;
        p.vyp = p.vy + 0.5 * dt * p.ay;

        p.x += dt * p.vxp;
        p.y += dt * p.vyp;
    }

    applyPeriodicBoundary(particles, L);

    computeForces(particles, L, rcut, fe);

    for (auto& p : particles) {
        p.vx = p.vxp + 0.5 * dt * p.ax;
        p.vy = p.vyp + 0.5 * dt * p.ay;
    }
}

void savePositions(const vector<Particle>& particles, const string& filename) {
    ofstream out(filename);
    for (const auto& p : particles) out << p.x << " " << p.y << "\n";
}

void savePositionsVelocities(const vector<Particle>& particles,
                             const string& filename) {
    ofstream out(filename);
    for (const auto& p : particles) {
        out << p.x << " " << p.y << " "
            << p.vx << " " << p.vy << "\n";
    }
}


void appendEnergies(ofstream& out, int step, double ek, double ep, double T) {
    out << step << " " << ek << " " << ep << " " << (ek + ep) << " " << T << "\n";
}


void initializePositions(vector<Particle>& particles, int N, double L) {
    int k = (int)ceil(sqrt((double)N));
    double Delta = L / (double)k;

    for (int i = 0; i < N; ++i) {
        int ix = i % k;
        int iy = i / k;
        particles[i].x = 0.5 * Delta + ix * Delta;
        particles[i].y = 0.5 * Delta + iy * Delta;
    }
}


void initializeVelocities(vector<Particle>& particles, double T0) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, sqrt(T0/mass));

    double vx_sum = 0.0, vy_sum = 0.0;

    for(auto& p : particles){
        p.vx = dist(gen);
        p.vy = dist(gen);
        vx_sum += p.vx;
        vy_sum += p.vy;
    }

    vx_sum /= particles.size();
    vy_sum /= particles.size();

    for(auto& p : particles){
        p.vx -= vx_sum;
        p.vy -= vy_sum;
    }
}

void rescaleVelocitiesToTemperature(vector<Particle>& particles, double T_target) {
    double T_now = instantaneousTemperature(particles);
    double scale = sqrt(T_target / T_now);
    for (auto& p : particles) {
        p.vx *= scale;
        p.vy *= scale;
    }
}

struct StepResult {
    double ekin = 0.0;
    double epot = 0.0;
    double etot = 0.0;
    double Tinst = 0.0;
};

StepResult noseHooverVerletStep(
        vector<Particle>& particles,
        double L,
        double rcut,
        double T_target,
        double Q,
        double& xi,
        int kmax,
        ForceEnergy& fe
) {
    const int N = (int)particles.size();
    const int g = 2*N - 2;

    for (auto& p : particles) {
        p.vxp = p.vx + 0.5*dt*(p.ax - xi*p.vx);
        p.vyp = p.vy + 0.5*dt*(p.ay - xi*p.vy);
    }

    for (auto& p : particles) {
        p.x += dt * p.vxp;
        p.y += dt * p.vyp;
    }
    applyPeriodicBoundary(particles, L);

    computeForces(particles, L, rcut, fe);

    for (auto& p : particles) {
        p.vx = p.vxp;
        p.vy = p.vyp;
    }

    StepResult sr;

    for (int k = 0; k < kmax; ++k) {
        sr.ekin  = kineticEnergy(particles);
        sr.Tinst = (2.0 * sr.ekin) / (double)g;

        xi += (dt / Q) * (2.0*sr.ekin - g*T_target);

        const double denom = 1.0 + 0.5*dt*xi;
        for (auto& p : particles) {
            p.vx = (p.vxp + 0.5*dt*p.ax) / denom;
            p.vy = (p.vyp + 0.5*dt*p.ay) / denom;
        }
    }

    sr.epot = fe.epot;
    return sr;
}

inline double maxwell2D_speed_pdf(double v, double T) {
    if (v < 0.0) return 0.0;
    return (v / T) * exp(-(v*v) / (2.0*T));
}

void accumulateVelocityHistogram(
        const vector<Particle>& particles,
        vector<double>& vhist,
        double dv,
        double normFactor
) {
    const int nv = (int)vhist.size();
    for (const auto& p : particles) {
        double v = sqrt(p.vx*p.vx + p.vy*p.vy);
        int k = (int)floor(v / dv);
        if (k >= 0 && k < nv) vhist[k] += normFactor;
    }
}

void saveVelocityHistogram(
        const string& filename,
        const vector<double>& vhist,
        double dv,
        double T_target
) {
    ofstream out(filename);
    out << "# v_center vhist f_exact(T=" << T_target << ")\n";
    out << std::setprecision(12);

    for (int k = 0; k < (int)vhist.size(); ++k) {
        double v_center = (k + 0.5) * dv;
        double f_exact  = maxwell2D_speed_pdf(v_center, T_target);
        out << v_center << " " << vhist[k] << " " << f_exact << "\n";
    }
}

double T_target_it(int it, double Tmax, double Tmin, int Nit, int nT) {
    int Kit = Nit / nT;
    double dT = (Tmax - Tmin) / nT;

    int level = it / Kit;
    if (level > nT) level = nT;

    return Tmax - level * dT;
}


int main() {
    const double L = 25.0;
    const int N = 201;
    const double rcut = 2.7;
    const double T0 = 300.0 / 119.0;
    const int Nit  = 10000;
    const int Nit2 = 50000;

    vector<Particle> particles(N);
    ForceEnergy fe;

    /* =========================
       TASK 1: INITIAL CONDITIONS
       ========================= */

    cout << "task 1 running\n";
    initializePositions(particles, N, L);
    savePositions(particles, "pos_init.dat");

    initializeVelocities(particles, T0);
    savePositionsVelocities(particles, "pos_vel_init.dat");

    computeForces(particles, L, rcut, fe);

    double T_after_drift = instantaneousTemperature(particles);

    rescaleVelocitiesToTemperature(particles, T0);
    double T_after_rescale = instantaneousTemperature(particles);

    cout << "T_target (T0): " << T0 << "\n";
    cout << "T_after_drift: " << T_after_drift << "\n";
    cout << "T_after_rescale: " << T_after_rescale << "\n";

    /* =========================
       TASK 2(a): ENERGY CONSERVATION (NVE)
       ========================= */

    cout << "task 2a running\n";
    ofstream eout("energies_2a.dat");
    eout << "# step time Ekin Epot Etot T\n";

    for (int it = 0; it <= Nit; ++it) {
        if (it % 10 == 0) {
            double ek = kineticEnergy(particles);
            double ep = fe.epot;
            double T  = ek / (double)N;
            double time = it * dt;

            eout << it << " "
                 << time << " "
                 << ek << " "
                 << ep << " "
                 << (ek + ep) << " "
                 << T << "\n";
        }

        velocityVerletStep(particles, L, rcut, fe);
    }
    eout.close();

    /* =========================
       TASK 2(b): TRAJECTORY (FRESH RUN!)
       ========================= */

    cout << "task 2b running\n";
    // IMPORTANT: reinitialize system
    initializePositions(particles, N, L);
    initializeVelocities(particles, T0);
    computeForces(particles, L, rcut, fe);

    const int l = N / 2;

    ofstream tout("traj.dat");
    tout << "# step time x y\n";

    for (int it = 0; it <= Nit2; ++it) {
        if (it % 10 == 0) {
            double time = it * dt;
            tout << it << " "
                 << time << " "
                 << particles[l].x << " "
                 << particles[l].y << "\n";
        }

        velocityVerletStep(particles, L, rcut, fe);
    }
    tout.close();

    /* =========================
       TASK 3
       ========================= */

    cout << "task 3 running\n";
    vector<double> Qvals = {1.0, 0.1, 0.01};

    for (double Q : Qvals) {
        vector<Particle> particles(N);
        ForceEnergy fe;

        // init
        initializePositions(particles, N, L);
        initializeVelocities(particles, T0);


        computeForces(particles, L, rcut, fe);

        double xi = 0.0;

        string fname = "task3_Q_" + to_string(Q) + ".dat";
        ofstream out(fname);
        out << "# step time Ekin Epot Etot T\n";

        for (int it = 0; it <= Nit; ++it) {
            StepResult sr = noseHooverVerletStep(
                    particles, L, rcut, T0, Q, xi, 5, fe
            );

            if (it % 10 == 0) {
                double time = it * dt;
                out << it << " " << time << " "
                    << sr.ekin << " " << sr.epot << " "
                    << (sr.ekin + sr.epot) << " "
                    << sr.Tinst << "\n";
            }
        }
        out.close();

    }
    /* =========================
       TASK 4
       ========================= */

    cout << "task 4 running\n";
    const double Q  = 10.0;
    const int kmax = 5;

    const int nv = 120;
    const double vc = sqrt(2.0 * T0);
    const double vmax = 4.0 * vc;
    const double dv = vmax / (double)nv;

    const int Nit_list[3] = {1000, 10000, 100000};

    for (int caseIdx = 0; caseIdx < 3; ++caseIdx) {
        int Nit = Nit_list[caseIdx];

        vector<Particle> particles(N);
        ForceEnergy fe;

        initializePositions(particles, N, L);
        initializeVelocities(particles, T0);
        computeForces(particles, L, rcut, fe);

        vector<double> vhist(nv, 0.0);

        const double normFactor = 1.0 / ( (double)Nit * (double)N * dv );

        double xi = 0.0;

        const int Neq = (int)(0.1 * Nit);
        for (int it = 0; it < Nit; ++it) {

            StepResult sr = noseHooverVerletStep(particles, L, rcut, T0, Q, xi, kmax, fe);

            if (it >= Neq) {
                const double normEq = 1.0 / ( (double)(Nit - Neq) * (double)N * dv );
                accumulateVelocityHistogram(particles, vhist, dv, normEq);
            }
        }

        string fname = "task4_Nit_" + to_string(Nit) + ".dat";
        saveVelocityHistogram(fname, vhist, dv, T0);

        cout << "Saved " << fname << " (Q=" << Q << ", Nit=" << Nit << ")\n";
    }

    cout << "task 5 running\n";

    const double Q5 = 1.0;
    const int kmax5 = 5;
    const int nT = 500;

    const double Tmax = 300.0 / 119.0;
    const double Tmin = 5.0   / 119.0;

    auto T_target_it = [&](int it, int NitRun) {
        int Kit = NitRun / nT;
        double dT = (Tmax - Tmin) / nT;

        int level = (Kit > 0) ? (it / Kit) : 0;
        if (level > nT) level = nT;

        return Tmax - level * dT;
    };

    auto run_task5 = [&](int NitRun, const string& tag) {

        vector<Particle> particles5(N);
        ForceEnergy fe5;

        initializePositions(particles5, N, L);
        initializeVelocities(particles5, Tmax);     // start at Tmax (not T0)
        computeForces(particles5, L, rcut, fe5);

        double xi5 = 0.0;

        ofstream Tout("task5_T_" + tag + ".dat");
        Tout << "# it time Tinst Tset\n";

        for (int it = 0; it <= NitRun; ++it) {

            double Tset = T_target_it(it, NitRun);

            StepResult sr = noseHooverVerletStep(particles5, L, rcut, Tset, Q5, xi5, kmax5, fe5);

            if (it % 10 == 0) {
                double Tinst = instantaneousTemperature(particles5);
                Tout << it << " " << it * dt << " " << Tinst << " " << Tset << "\n";
            }
        }
        Tout.close();

        ofstream Pout("task5_pos_" + tag + ".dat");
        Pout << "# x y\n";
        for (auto &p : particles5) {
            Pout << p.x << " " << p.y << "\n";
        }
        Pout.close();

        cout << "Task 5 done: " << tag << " (Nit=" << NitRun << ")\n";
    };

    run_task5(10000, "fast");
    run_task5(100000, "slow");


    return 0;
}


