import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = '/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/'

n_values = [10, 20, 50, 100, 200, 500]
colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple']

plt.figure(figsize=(8, 6))
for i, n in enumerate(n_values):
    file_path = os.path.join(base_dir, f'trajectory_n{n}.txt')
    data = np.loadtxt(file_path)
    plt.plot(data[:,1], data[:,2], label=f'n={n}', color=colors[i])
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Projectile Trajectories for Different n')
plt.legend()
plt.grid()
plt.savefig(os.path.join(base_dir, 'trajectories.png'))
plt.show()

error_file = os.path.join(base_dir, 'global_error.txt')
error_data = np.loadtxt(error_file)
plt.figure(figsize=(8, 6))
plt.loglog(error_data[:,0], np.abs(error_data[:,1]), 'o-')
plt.xlabel('Δt [s]')
plt.ylabel('Global Error |Eglob| [m]')
plt.title('Global Error vs. Time Step (Δt)')
plt.grid(which='both')
plt.savefig(os.path.join(base_dir, 'global_error.png'))
plt.show()

file_paths = [
    'drag_D0.000000.txt',
    'drag_D0.000100.txt',
    'drag_D0.000200.txt',
    'drag_D0.000500.txt',
    'drag_D0.001000.txt'
]
labels = ['D=0', 'D=1e-4', 'D=2e-4', 'D=5e-4', 'D=1e-3']
colors = ['red', 'orange', 'gold', 'green', 'blue']

plt.figure(figsize=(8, 6))
for i, file_path in enumerate(file_paths):
    data = np.loadtxt(base_dir + file_path)
    plt.plot(data[:,1], data[:,2], label=labels[i], color=colors[i])

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Projectile Trajectories with Drag')
plt.legend()
plt.grid()
plt.savefig('/Users/nataliakowalczyk/CLionProjects/ComputationalPhysics/cmake-build-debug/drag_trajectories.png')
plt.show()

files = [
    ('range_D0.000.txt', 'D = 0'),
    ('range_D0.001.txt', 'D = 1e-3'),
    ('range_D0.002.txt', 'D = 2e-3')
]

for filename, label in files:
    data = np.loadtxt(base_dir + filename)
    angles = data[:,0]
    ranges = data[:,1]
    plt.figure(figsize=(8, 6))
    plt.plot(angles, ranges, marker='o')
    plt.xlabel('Firing angle θ [deg]')
    plt.ylabel('Range $x_{max}$ [m]')
    plt.title(f'Range vs. Firing Angle ({label})')
    plt.grid()
    plt.savefig(f"{base_dir}range_vs_angle_{label}.png")
    plt.show()

files = [
    ('artillery_a0_alpha35.txt', 'a=0, α=35°'),
    ('artillery_a0_alpha45.txt', 'a=0, α=45°'),
    ('artillery_a6.5e-3_alpha35.txt', 'a=6.5e-3, α=35°'),
    ('artillery_a6.5e-3_alpha45.txt', 'a=6.5e-3, α=45°')
]
colors = ['red', 'orange', 'gold', 'green']

plt.figure(figsize=(10, 7))
for i, (filename, label) in enumerate(files):
    data = np.loadtxt(base_dir + filename)
    plt.plot(data[:,1], data[:,2], label=label, color=colors[i])

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Artillery Trajectories: Altitude Correction and Firing Angle')
plt.legend()
plt.grid()
plt.savefig(base_dir + 'artillery_trajectories_comparison.png')
plt.show()