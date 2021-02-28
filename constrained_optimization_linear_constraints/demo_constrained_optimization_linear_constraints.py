"""
demo_constrained_optimization_linear_constraints.py

Minimize a (quadratic) function of two variables:
Minimize f(X), where X = [x0, x1].T;
X* is MIN;

m = number of equality constraints;
n = number of parameters;

Objective function:
    Minimize a (quadratic) function of two variables: f(X);
    X = [x1, ..., xn].T is vector of variables;

Box constraints:
    X_lower = [x1_lower, ..., xn_lower].T; lower bounds for X;
    X_upper = [x1_upper, ..., xn_upper].T; upper bounds for X;
    
Linear equality constraints:
    Subject to linear equality constraints: h(X);
    h(X) = [h1(X), ..., hn(X)].T is vector of constraints;
    h(X) = A @ X - B = 0;
    A @ X = B;
    m < n;

Gradient of linear equality constraints:    
Dh(X) = [Dh1(X), ..., Dhn(X)] = A.T;

Solution:
X* of f(X) lies in the N(A), where N(A) is the nullspace of A;
Projector onto N(A):
P = In - A.T @ inv(A @ A.T) @ A.T, where In is the identity matrix; 

On each iteration, X should be projected onto the constrained parameter space:
    (1) onto the constrained parameter space given by the equality constraints, i.e. onto the null space of A, 
    because X* lies in N(A);
    (2) onto constrained parameter space given by the box constraints, i.e. onto the box constraints of X;
    
Projected gradient algorithm with variable step size (steepest descent). 
    Only for quadratic f(X) function. Requires gradient and hessian. 
    X0 converges to X* for any X0 (subject to the constraints).
    In parameter space, moving along orthogonal directions with variable step size.
    
Projected gradient algorithm with variable step size based on line search
    General algorithm for any f(X) function. Requires only gradient.
    X0 converges to X* for any X0 (subject to the constraints).
    In parameter space, moving along orthogonal directions with variable step size determined by line search.
    Step size is determined by line search to ensure the descent property f(Xn+1) < f(Xn)

Projected conjugate gradient algorithm with variable step size (steepest descent). 
    Only for quadratic f(X) function. Requires gradient and hessian.
    Guaranteed to converge in n steps for n parameters, i.e. for X = [x1, ..., xn].T. 
    X0 converges to X* for any X0 (subject to the constraints).
    In parameter space, moving along mutually conjugate directions with variable step size.    
    
Projected conjugate gradient algorithm with variable step size based on line search
    General algorithm for any f(X) function. Requires only gradient.
    X0 converges to X* for any X0 (subject to the constraints).
    In parameter space, moving along mutually conjugate directions with variable step size.
    Step size is determined by line search to ensure the descent property f(Xn+1) < f(Xn).
    For non-quadratic f(X) conjugate direction vectors will tend to deteriorate and need to be 
        reset to the negative gradient every n steps.
        
Projected Newton's algorithm. 
    General algorithm for any f(X) function. Requires gradient and hessian.
    Can converge in as few as 1 step (for quadratic f(X)).
    X0 should be close to X*. X0 is subject to the constraints.
    In parameter space, moving to the MIN of parabola in 1 step.
    
Projected Quasi-Newton algorithm with variable step size (steepest descent).
    Only for quadratic f(X) function. Requires gradient and hessian.
    Guaranteed to converge in n steps for n parameters, i.e. for X = [x1, ..., xn].T. 
    X0 converges to X* for any X0 (subject to the constraints).
    In parameter space, moving along mutually conjugate directions with variable step size.
    For non-quadratic f(X) conjugate direction vectors will tend to deteriorate and need to be 
        reset to the negative gradient every n steps.
        
Projected Quasi-Newton algorithm with variable step size based on line search
    General algorithm for any f(X) function. Requires only gradient.
    X0 converges to X* for any X0 (subject to the constraints).
    In parameter space, moving along mutually conjugate directions with variable step size.
    Step size is determined by line search to ensure the descent property f(Xn+1) < f(Xn).
    For non-quadratic f(X) conjugate direction vectors will tend to deteriorate and need to be 
        reset to the negative gradient every n steps.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from projected_gradient_algorithm import projected_gradient_algorithm_var_alpha, projected_gradient_algorithm_linesearch
from projected_conjugate_gradient_algorithm import projected_conjugate_gradient_algorithm, projected_conjugate_gradient_algorithm_linesearch
from projected_newton_algorithm import projected_newton_algorithm
from projected_quasi_newton_algorithm import projected_quasi_newton_algorithm, projected_quasi_newton_algorithm_linesearch
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
# func_grad = lambda X : Q @ X + B + 2 * reg_coeff * X; # original gradient
func_hessian = lambda X : Q + 2 * reg_coeff;

# Box constraints:
X_lower = np.array([[-20], [-20]]); # X lower bound   
X_upper = np.array([[20], [20]]); # X upper bound 

# Linear equality constraints:
# h(X) = A @ X - Ba = 0;
# h(X) = 2x1 + 5x2 - 10 = 0;
# X = [x1, x2].T;
m = 1;
n = 2;
A = np.array([[2, 5]]);
Ba = np.array([[10]]); 
P = np.eye(n) - A.T @ np.linalg.inv(A @ A.T) @ A;
func_grad = lambda X : P @ (Q @ X + B + 2 * reg_coeff * X); # gradient projected onto N(A)
# Initial guess (subject to the equality constraints):
X0 = np.array([[50], [-18]]);   

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
cset = ax.contour(X_data, Y_data, Z_data, 50, cmap = cm.coolwarm);
ax.set_xlabel('X')
ax.set_xlim(-100, 100)
ax.set_ylabel('Y')
ax.set_ylim(-100, 100)
plt.title('Objective function: contour plot');
plt.show();

# Projected gradient algorithm with variable step size (steepest descent)
print('***********************************************************************');
print('Projected gradient algorithm with variable step size (steepest descent)');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 'x_upper' : X_upper};
start = time.time();
X, report = projected_gradient_algorithm_var_alpha(X0, func, func_grad, func_hessian, P, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Projected gradient algorithm with variable alpha';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Projected gradient algorithm with variable step size based on line search
print('***********************************************************************');
print('Projected gradient algorithm with variable step size based on line search');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 'x_upper' : X_upper}; 
start = time.time();
X, report = projected_gradient_algorithm_linesearch(X0, func, func_grad, P, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Projected gradient algorithm with line search';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Projected conjugate gradient algorithm with variable step size (steepest descent)
print('***********************************************************************');
print('Projected conjugate gradient algorithm with variable step size (steepest descent)');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 'x_upper' : X_upper}; 
start = time.time();
X, report = projected_conjugate_gradient_algorithm(X0, func, func_grad, func_hessian, P, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Projected conjugate gradient algorithm with variable alpha';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Projected conjugate gradient algorithm with variable step size based on line search
print('***********************************************************************');
print('Projected conjugate gradient algorithm with variable step size based on line search');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 'x_upper' : X_upper}; 
start = time.time();
X, report = projected_conjugate_gradient_algorithm_linesearch(X0, func, func_grad, P, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Projected conjugate gradient algorithm with line search';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Projected Newton's algorithm
print('***********************************************************************');
print('Projected Newton\'s algorithm');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 'x_upper' : X_upper}; 
start = time.time();
X, report = projected_newton_algorithm(X0, func, func_grad, func_hessian, P, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Projected Newton\'s algorithm';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Projected Quasi-Newton algorithm with variable step size (steepest descent)
print('***********************************************************************');
print('Projected Quasi-Newton algorithm with variable step size (steepest descent)');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 'x_upper' : X_upper}; 
start = time.time();
X, report = projected_quasi_newton_algorithm(X0, func, func_grad, func_hessian, P, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Projected Quasi-Newton algorithm with variable alpha';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Projected Quasi-Newton algorithm with variable step size based on line search
print('***********************************************************************');
print('Projected Quasi-Newton algorithm with variable step size based on line search');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 'x_upper' : X_upper};    
start = time.time();
X, report = projected_quasi_newton_algorithm_linesearch(X0, func, func_grad, P, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'Projected Quasi-Newton algorithm with line search';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');
