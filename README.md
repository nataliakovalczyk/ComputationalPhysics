<div align="center">
<h1>Computational Physics</h1>

Numerical simulations and analysis from **Computational Physics 1 & 2** coursework.

[Overview](#overview) • [What’s included](#whats-included) • [Getting started](#getting-started) • [Repository layout](#repository-layout)
</div>

## Overview

This repository contains a collection of lab assignments and larger projects covering core computational physics methods:

- Ordinary differential equations (RK4, symplectic schemes)
- Partial differential equations (finite differences, Crank–Nicolson, SOR)
- Finite element methods (FEM) for static and time-dependent problems
- Eigenvalue problems and quantum simulations
- Molecular dynamics (Lennard-Jones + thermostats)

Each module typically includes:
- a **C++ implementation** of the simulation,
- a **Python post-processing script** for plots,
- generated data files, figures, and a report PDF.

## What’s included

### Labs (`Lab1` → `Lab8`)

Hands-on exercises focused on specific methods and models:

- **Lab1**: projectile motion with drag and atmosphere correction
- **Lab2**: damped/driven oscillator and resonance
- **Lab3**: Mercury orbit and perihelion precession
- **Lab4**: stiff radioactive decay system with adaptive stepping
- **Lab5**: 2D electrostatics / Poisson-type relaxation (SOR)
- **Lab6**: 2D heat diffusion with boundary heat exchange
- **Lab7**: 1D wave equation, damping, forcing, modal analysis
- **Lab8**: 2D molecular dynamics and thermalization

### Projects (`Project1` → `Project10`)

Larger numerical studies, including:

- iterative Poisson solvers and multilevel refinement,
- FEM potential/eigenvalue formulations (bilinear and biparabolic bases),
- 2D quantum state search and time evolution,
- driven membrane dynamics (FEM and finite differences),
- advection-diffusion and variable-coefficient diffusion in 2D.

## Tech stack

- **C++** for simulation kernels
- **Python** for data analysis and plotting
- **Eigen** in FEM/eigenvalue-heavy projects
- **NumPy / Pandas / Matplotlib** for processing and visualization

## Getting started

### Prerequisites

- C++ compiler with C++17 support (or newer)
- Python 3.9+
- Python packages:

```bash
pip install numpy pandas matplotlib
```

> [!TIP]
> Most modules are self-contained. You can run one folder at a time without setting up a monorepo-style build.

### Run a C++ simulation

Example from a module folder:

```bash
cd Lab2
g++ -O2 -std=c++17 Lab2.cpp -o Lab2
./Lab2
```

For modules using Eigen (for example `Project3`, `Project6`, `Project7`, `Project9`), ensure Eigen headers are available to your compiler.

### Generate plots with Python

```bash
cd Lab2
python3 Lab2.py
```

## Repository layout

```text
ComputationalPhysics/
├── Lab1 ... Lab8/          # Focused coursework exercises
├── Project1 ... Project10/ # Larger numerical projects
└── README.md
```

Inside each lab/project directory you will generally find:

- `*.cpp` simulation source
- `*.py` plotting/analysis script
- `*.csv`, `*.dat`, `*.txt` outputs
- `*.png` figures
- `*_report.pdf` report deliverables

## Scope at a glance

This repository is organized **by assignment/topic** rather than as a reusable software library.  
It is best used as:

- a reference for numerical method implementations,
- a companion to computational physics coursework,
- a collection of reproducible simulation examples.
