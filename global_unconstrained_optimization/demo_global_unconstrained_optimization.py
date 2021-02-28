"""
demo_global_unconstrained_optimization.py 

Minimize a function of two variables:
Minimize f(X), where X = [x0, x1].T

Naive random walk algorithm
    General algorithm for any f(X) function.
    X0 converges to X* for any X0.
    
Simulated annealing algorithm
    General algorithm for any f(X) function.
    X0 converges to X* for any X0.
    
Particle swarm optimization algorithm
    General algorithm for any f(X) function.
    X0 converges to X* for any X0.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from naive_random_search_algorithm import naive_random_search_algorithm
from simulated_annealing_algorithm import simulated_annealing_algorithm
from particle_swam_optimization_algorithm import particle_swam_optimization_algorithm
from print_report import print_report
from plot_progress_y import plot_progress_y
from plot_progress_x import plot_progress_x
import time

# Minimize the following objective function:
# X = [x0, x1].T;
# f(X) = 3 * (1 - x0)^2 * exp(-x0^2 - (x1 + 1)^2) - 10 * (x0/5 - x1^3 - x1^5) * exp(-x0^2 - x1^2) - exp(-(x0 + 1)^2 - x1^2)/3;
func = lambda X: 3 * (1 - X[0])**2 * np.exp(-X[0]**2 - (X[1] + 1)**2) - 10 * (X[0] / 5 - X[0]**3 - X[1]**5) \
    * np.exp(-X[0]**2 - X[1]**2) - np.exp(-(X[0] + 1)**2 - X[1]) / 3;
# Objective function for particle swarm optimization algorithm (particles along row dimension, axis = 0)
func_ps = lambda X: 3 * (1 - X[:, [0]])**2 * np.exp(-X[:, [0]]**2 - (X[:, [1]] + 1)**2) - 10 \
    * (X[:, [0]] / 5 - X[:, [0]]**3 - X[:, [1]]**5) * np.exp(-X[:, [0]]**2 - X[:, [1]]**2) - np.exp(-(X[:, [0]] + 1)**2 - X[:, [1]]) / 3;

fig = plt.figure();
X_data = np.arange(-3, 3, 0.1);
Y_data = np.arange(-3, 3, 0.1);
Z_data = np.empty(shape = (X_data.size,Y_data.size));
for iX in range (X_data.size):
    for iY in range (Y_data.size):
        Z_data[iX][iY] = func(np.array([X_data[iX], Y_data[iY]]));
X_data, Y_data = np.meshgrid(X_data, Y_data);

# Plot the surface
ax = fig.add_subplot(1, 2, 1, projection ='3d')
surf = ax.plot_surface(X_data, Y_data, Z_data, cmap = cm.coolwarm,
                       linewidth = 0, antialiased = False)
ax.set_xlabel('X')
ax.set_xlim(-3, 3)
ax.set_ylabel('Y')
ax.set_ylim(-3, 3)
plt.title('Objective function: surface plot')

# Plot the contour
ax = fig.add_subplot(1, 2, 2)
cset = ax.contour(X_data, Y_data, Z_data, 50, cmap=cm.coolwarm)
ax.set_xlabel('X')
ax.set_xlim(-3, 3)
ax.set_ylabel('Y')
ax.set_ylim(-3, 3)
plt.title('Objective function: contour plot')
plt.show()

# Naive random walk algorithm
print('***********************************************************************');
print('Naive random walk algorithm');
N_iter_max = 1000;
tolerance_x = 10e-8;
tolerance_y = 10e-8;
X_lower = np.array([[-2], [-2]]); # X lower bound   
X_upper = np.array([[2], [2]]); # X upper bound
alpha = 0.25; # step size
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 
           'x_upper' : X_upper, 'alpha' : alpha};
X0 = np.array([[0], [1.5]]); # X0 = X_lower + (X_upper - X_lower) * np.random.rand(X_lower.size, 1); 
start = time.time();
X, report = naive_random_search_algorithm(X0, func, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Naive random walk algorithm';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Simulated annealing algorithm
print('***********************************************************************');
print('Simulated annealing algorithm');
N_iter_max = 1000;
tolerance_x = 10e-8;
tolerance_y = 10e-8;
X_lower = np.array([[-2], [-2]]); # X lower bound   
X_upper = np.array([[2], [2]]); # X upper bound   
alpha = 0.25; # step size
gamma = 1.5; # controls temperature decay, gamma > 0
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 
           'x_upper' : X_upper, 'alpha' : alpha, 'gamma' : gamma};
X0 = np.array([[0], [1.5]]); # X0 = X_lower + (X_upper - X_lower) * np.random.rand(X_lower.size, 1); 
start = time.time();
X, report = simulated_annealing_algorithm(X0, func, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Simulated annealing algorithm';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Particle swarm optimization algorithm
print('***********************************************************************');
print('Particle swarm optimization algorithm');
N_iter_max = 1000;
tolerance_x = 10e-8;
tolerance_y = 10e-8;
X_lower = np.array([[-2], [-2]]); # X lower bound   
X_upper = np.array([[2], [2]]); # X upper bound 
d_lower = -1; # direction (aka velocity) lower bound
d_upper = 1; # direction (aka velocity) upper bound
N_ps = 10000; # number of particles
w = 0.9; # inertial constant, w < 1
c1 = 1.5; # cognitive/independent component, c1 ~ 2
c2 = 1.5; # social component, c2 ~ 2
alpha = 1; # step size
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 
           'x_upper' : X_upper, 'alpha' : alpha, 'd_lower' : d_lower, 'd_upper' : d_upper, 'N_ps' : N_ps, 'w' : w, 'c1' : c1, 'c2' : c2};
start = time.time();
X, report = particle_swam_optimization_algorithm(func_ps, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Particle swarm optimization algorithm';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');