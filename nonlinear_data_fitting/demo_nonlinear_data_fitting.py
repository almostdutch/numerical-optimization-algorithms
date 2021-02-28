"""
demo_nonlinear_data_fitting.py

Fit a model A * sin(W * t + phi) to the data f(X, ti) = yi to find A, W, and phi
m = number of data points

Solve a system of non-linear equations f(X, ti) - yi = 0:
    x1 * sin(x2 * t + x3) - y = 0, where X = [x1 = A, x2 = W, x3 = phi].T, t = [t1, t2, ..., tm].T and y = [y1, y2, ..., ym].T

Minimize the following objective function: (f(X, t) - y).T @ (f(X, t) - y)

Levenberg - Marquardt algorithm
    General algorithm for any f(X) function. Requires residuals and jacobian.
    X0 converges to X* for any X0.
    
Naive random walk algorithm
    General algorithm for any f(X) function. Requires only residuals.
    X0 converges to X* for any X0.
    
Simulated annealing algorithm
    General algorithm for any f(X) function. Requires only residuals.
    X0 converges to X* for any X0.
    
Particle swarm optimization algorithm
    General algorithm for any f(X) function. Requires only residuals.
    X0 converges to X* for any X0.
"""

import numpy as np
import matplotlib.pyplot as plt
from levenberg_marquardt_algorithm import levenberg_marquardt_algorithm
from naive_random_search_algorithm import naive_random_search_algorithm
from simulated_annealing_algorithm import simulated_annealing_algorithm
from particle_swam_optimization_algorithm import particle_swam_optimization_algorithm
from print_report import print_report
from plot_progress_y import plot_progress_y
import time

X = np.array([[0.75], [2.3], [-3]]);
t = np.arange(0, 20, 0.1);
# Nonlinear model to fit
func = lambda X : X[0] * np.sin(X[1] * t + X[2]); 
# Plot the curve
fig = plt.figure();
y = func(X);
plt.plot(t, y)
plt.show()

func_residual = lambda X : (X[0] * np.sin(X[1] * t + X[2]) - y).reshape((t.size, 1));
func_jacobian = lambda X : np.array([[-np.sin(X[1] * t + X[2])], [-t * X[0] * np.cos(X[1] * t + X[2])], [-X[0] * np.cos(X[1] * t + X[2])]]).reshape((t.size, X.size));
# Objective function for naive random walk and simulated annealing algorithms
func_error = lambda X : np.linalg.norm((X[0] * np.sin(X[1] * t + X[2]) - y).reshape((t.size, 1)), axis = 0) ** 2;
# Objective function for particle swarm optimization algorithm (particles along row dimension, axis = 0)
func_error_ps = lambda X : np.linalg.norm(X[:, [0]] * np.sin(X[:, [1]] * t + X[:, [2]]) - y, axis = 1).reshape((-1, 1)) ** 2;

# Levenberg-Marquardt algorithm
print('***********************************************************************');
print('Levenberg - Marquardt algorithm');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
X_lower = np.array([[0], [0], [-5]]); # X lower bound   
X_upper = np.array([[2], [5], [0]]); # X upper bound
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 
           'x_upper' : X_upper};
X0 = np.array([[0.1], [1], [-2]]);
start = time.time();
X, report = levenberg_marquardt_algorithm(X0, func_residual, func_jacobian, options);
end = time.time();
print_report(func_error, report);
# Plot path to X* for Y
algorithm_name = 'Levenberg - Marquardt algorithm';
plot_progress_y(algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Naive random walk algorithm
print('***********************************************************************');
print('Naive random walk algorithm');
N_iter_max = 1000;
tolerance_x = 10e-8;
tolerance_y = 10e-8;
X_lower = np.array([[0], [0], [-5]]); # X lower bound   
X_upper = np.array([[2], [5], [0]]); # X upper bound
alpha = 0.5; # step size
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 
           'x_upper' : X_upper, 'alpha' : alpha};
X0 = np.array([[0.1], [1], [-2]]); # X0 = X_lower + (X_upper - X_lower) * np.random.rand(X_lower.size, 1); 
start = time.time();
X, report = naive_random_search_algorithm(X0, func_error, options);
end = time.time();
print_report(func_error, report);
# Plot path to X* for Y
algorithm_name = 'Naive random walk algorithm';
plot_progress_y(algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Simulated annealing algorithm
print('***********************************************************************');
print('Simulated annealing algorithm');
N_iter_max = 1000;
tolerance_x = 10e-8;
tolerance_y = 10e-8;
X_lower = np.array([[0], [0], [-5]]); # X lower bound   
X_upper = np.array([[2], [5], [0]]); # X upper bound
alpha = 1.0; # step size
gamma = 1.5; # controls temperature decay, gamma > 0
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 
           'x_upper' : X_upper, 'alpha' : alpha, 'gamma' : gamma};
X0 = np.array([[0.1], [1], [-2]]); # X0 = X_lower + (X_upper - X_lower) * np.random.rand(X_lower.size, 1); 
start = time.time();
X, report = simulated_annealing_algorithm(X0, func_error, options);
end = time.time();
print_report(func_error, report);
# Plot path to X* for Y
algorithm_name = 'Simulated annealing algorithm';
plot_progress_y(algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Particle swarm optimization algorithm
print('***********************************************************************');
print('Particle swarm optimization algorithm');
N_iter_max = 1000;
tolerance_x = 10e-8;
tolerance_y = 10e-8;
X_lower = np.array([[0], [0], [-5]]); # X lower bound   
X_upper = np.array([[2], [5], [0]]); # X upper bound
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
X, report = particle_swam_optimization_algorithm(func_error_ps, options);
end = time.time();
print_report(func_error, report);
# Plot path to X* for Y
algorithm_name = 'Particle swarm optimization algorithm';
plot_progress_y(algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');