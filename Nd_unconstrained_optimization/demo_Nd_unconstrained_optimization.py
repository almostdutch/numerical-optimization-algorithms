"""
demo_Nd_unconstrained_optimization.py.py

Minimize a (quadratic) function of two variables:
Minimize f(X), where X = [x0, x1].T

Gradient algorithm with variable step size (steepest descent). 
    Only for quadratic f(X) function. Requires gradient and hessian. 
    Guaranteed to converge in n steps for n parameters, i.e. for X = [x1, ..., xn].T. 
    X0 converges to X* for any X0.
    In parameter space, moving along orthogonal directions with variable step size.
    
Gradient algorithm with fixed step size. 
    Only for quadratic f(X) function. Requires gradient and hessian. 
    X0 converges to X* for any X0.
    In parameter space, moving along orthogonal directions with fixed step size.

Gradient algorithm with variable step size based on line search
    General algorithm for any f(X) function. Requires only gradient.
    X0 converges to X* for any X0.
    In parameter space, moving along orthogonal directions with variable step size determined by line search.
    Step size is determined by line search to ensure the descent property f(Xn+1) < f(Xn)

Conjugate gradient algorithm with variable step size (steepest descent). 
    Only for quadratic f(X) function. Requires gradient and hessian.
    Guaranteed to converge in n steps for n parameters, i.e. for X = [x1, ..., xn].T. 
    X0 converges to X* for any X0.
    In parameter space, moving along mutually conjugate directions with variable step size.    
    
Conjugate gradient algorithm with variable step size based on line search
    General algorithm for any f(X) function. Requires only gradient.
    X0 converges to X* for any X0.
    In parameter space, moving along mutually conjugate directions with variable step size.
    Step size is determined by line search to ensure the descent property f(Xn+1) < f(Xn).
    For non-quadratic f(X) conjugate direction vectors will tend to deteriorate and need to be 
        reset to the negative gradient every n steps.
        
Newton's algorithm. 
    General algorithm for any f(X) function. Requires gradient and hessian.
    Can converge in as few as 1 step (for quadratic f(X)).
    X0 should be close to X*.
    In parameter space, moving to the MIN of parabola in 1 step.
    
Newton's algorithm with variable step size based on line search
    General algorithm for any f(X) function. Requires gradient and hessian.
    Can converge in as few as 1 step (for quadratic f(X)).
    X0 converges to X* for any X0.
    In parameter space, moving to the MIN of parabola in 1 step. Might be affected by linesearch. 
    Step size is determined by line search to ensure the descent property f(Xn+1) < f(Xn).

Quasi-Newton algorithm with variable step size (steepest descent).
    Only for quadratic f(X) function. Requires gradient and hessian.
    Guaranteed to converge in n steps for n parameters, i.e. for X = [x1, ..., xn].T. 
    X0 converges to X* for any X0.
    In parameter space, moving along mutually conjugate directions with variable step size.
    For non-quadratic f(X) conjugate direction vectors will tend to deteriorate and need to be 
        reset to the negative gradient every n steps.
        
Quasi-Newton algorithm with variable step size based on line search
    General algorithm for any f(X) function. Requires only gradient.
    X0 converges to X* for any X0.
    In parameter space, moving along mutually conjugate directions with variable step size.
    Step size is determined by line search to ensure the descent property f(Xn+1) < f(Xn).
    For non-quadratic f(X) conjugate direction vectors will tend to deteriorate and need to be 
        reset to the negative gradient every n steps.
        
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
from gradient_algorithm import gradient_algorithm_var_alpha, gradient_algorithm_fixed_alpha, gradient_algorithm_linesearch
from conjugate_gradient_algorithm import conjugate_gradient_algorithm, conjugate_gradient_algorithm_linesearch
from newton_algorithm import newton_algorithm, newton_algorithm_linesearch
from quasi_newton_algorithm import quasi_newton_algorithm, quasi_newton_algorithm_linesearch
from naive_random_search_algorithm import naive_random_search_algorithm
from simulated_annealing_algorithm import simulated_annealing_algorithm
from particle_swam_optimization_algorithm import particle_swam_optimization_algorithm
from print_report import print_report
from plot_progress_y import plot_progress_y
from plot_progress_x import plot_progress_x
import time

reg_coeff = 0.01; # regularization coefficient
# Minimize the following objective function:
# X = [x0, x1].T;
# f(X) = 1 / 2 * X.T @ Q @ X + X.T @ B + C + reg_coeff * X.T @ X;
Q = np.array([[8, 2 * np.sqrt(2)], [2 * np.sqrt(2), 10]]);
B = np.array([[3], [6]]);
C = 24;
func = lambda X : 1 / 2 * X.T @ Q @ X + X.T @ B + C + reg_coeff * X.T @ X; 
func_grad = lambda X : Q @ X + B + 2 * reg_coeff * X;
func_hessian = lambda X : Q + 2 * reg_coeff;
# Objective function for particle swarm optimization algorithm (particles along row dimension, axis = 0)
func_ps = lambda X: 3 * X[:, [0]] + 6 * X[:, [1]] + X[:, [0]] * (4 * X[:, [0]] + np.sqrt(2) * X[:, [1]]) + X[:, [1]] \
    * (5 * X[:, [1]] + np.sqrt(2) * X[:, [0]]) + 24 + reg_coeff * (X[:, [0]] * X[:, [0]] + X[:, [1]] * X[:, [1]]); 
 
fig = plt.figure();
X_data = np.arange(-100, 100, 5);
Y_data = np.arange(-100, 100, 5);
Z_data = np.zeros(shape = (X_data.size,Y_data.size));
for iX in range (X_data.size):
    for iY in range (Y_data.size):
        Z_data[iX][iY] = func(np.array([X_data[iX], Y_data[iY]]));
X_data, Y_data = np.meshgrid(X_data, Y_data);

# Plot the surface
ax = fig.add_subplot(1, 2, 1, projection ='3d')
surf = ax.plot_surface(X_data, Y_data, Z_data, cmap = cm.coolwarm,
                       linewidth = 0, antialiased = False)
ax.set_xlabel('X')
ax.set_xlim(-100, 100)
ax.set_ylabel('Y')
ax.set_ylim(-100, 100)
plt.title('Objective function: surface plot')

# Plot the contour
ax = fig.add_subplot(1, 2, 2)
cset = ax.contour(X_data, Y_data, Z_data, 50, cmap=cm.coolwarm)
ax.set_xlabel('X')
ax.set_xlim(-100, 100)
ax.set_ylabel('Y')
ax.set_ylim(-100, 100)
plt.title('Objective function: contour plot')
plt.show()

# Gradient algorithm with variable step size (steepest descent)
print('***********************************************************************');
print('Gradient algorithm with variable step size (steepest descent)');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.array([[-100], [-100]]);   
start = time.time();
X, report = gradient_algorithm_var_alpha(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Gradient algorithm with variable alpha';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Gradient algorithm with fixed step size
print('***********************************************************************');
print('Gradient algorithm with fixed step size');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.array([[-100], [-100]]);   
start = time.time();
X, report = gradient_algorithm_fixed_alpha(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Gradient algorithm with fixed alpha';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Gradient algorithm with variable step size based on line search
print('***********************************************************************');
print('Gradient algorithm with variable step size based on line search');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.array([[-100], [-100]]);   
start = time.time();
X, report = gradient_algorithm_linesearch(X0, func, func_grad, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Gradient algorithm with line search';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Conjugate gradient algorithm with variable step size (steepest descent)
print('***********************************************************************');
print('Conjugate gradient algorithm with variable step size (steepest descent)');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.array([[-100], [-100]]);   
start = time.time();
X, report = conjugate_gradient_algorithm(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Conjugate gradient algorithm with variable alpha';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Conjugate gradient algorithm with variable step size based on line search
print('***********************************************************************');
print('Conjugate gradient algorithm with variable step size based on line search');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.array([[-100], [-100]]);   
start = time.time();
X, report = conjugate_gradient_algorithm_linesearch(X0, func, func_grad, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Conjugate gradient algorithm with line search';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Newton's algorithm
print('***********************************************************************');
print('Newton\'s algorithm');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.array([[-100], [-100]]);   
start = time.time();
X, report = newton_algorithm(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Newton\'s algorithm';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Newton's algorithm with variable step size based on line search
print('***********************************************************************');
print('Newton\'s algorithm with variable step size based on line search');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.array([[-100], [-100]]);   
start = time.time();
X, report = newton_algorithm_linesearch(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Newton\'s algorithm with line search';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Quasi-Newton algorithm with variable step size (steepest descent)
print('***********************************************************************');
print('Quasi-Newton algorithm with variable step size (steepest descent)');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.array([[-100], [-100]]);   
start = time.time();
X, report = quasi_newton_algorithm(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Quasi-Newton algorithm with variable alpha';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Quasi-Newton algorithm with variable step size based on line search
print('***********************************************************************');
print('Quasi-Newton algorithm with variable step size based on line search');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.array([[-100], [-100]]);   
start = time.time();
X, report = quasi_newton_algorithm_linesearch(X0, func, func_grad, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Quasi-Newton algorithm with line search';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Naive random walk algorithm
print('***********************************************************************');
print('Naive random walk algorithm');
N_iter_max = 1000;
tolerance_x = 10e-8;
tolerance_y = 10e-8;
X_lower = np.array([[-150], [-150]]); # X lower bound   
X_upper = np.array([[150], [150]]); # X upper bound
alpha = 5; # step size
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 
           'x_upper' : X_upper, 'alpha' : alpha};
X0 = np.array([[-20], [-20]]); # X0 = X_lower + (X_upper - X_lower) * np.random.rand(X_lower.size, 1); 
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
X_lower = np.array([[-150], [-150]]); # X lower bound   
X_upper = np.array([[150], [150]]); # X upper bound 
alpha = 5; # step size
gamma = 1.5; # controls temperature decay, gamma > 0
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 
           'x_upper' : X_upper, 'alpha' : alpha, 'gamma' : gamma};
X0 = np.array([[-20], [-20]]); # X0 = X_lower + (X_upper - X_lower) * np.random.rand(X_lower.size, 1); 
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
X_lower = np.array([[-150], [-150]]); # X lower bound   
X_upper = np.array([[150], [150]]); # X upper bound
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

