"""
demo_system_of_linear_equations.py 

Solving a system of linear equations in a least squares way:
A @ X = B (matrix form)

Normal equation with Tikhonov regularization

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
from gradient_algorithm import gradient_algorithm_var_alpha, gradient_algorithm_fixed_alpha, gradient_algorithm_linesearch
from conjugate_gradient_algorithm import conjugate_gradient_algorithm, conjugate_gradient_algorithm_linesearch
from newton_algorithm import newton_algorithm, newton_algorithm_linesearch
from quasi_newton_algorithm import quasi_newton_algorithm, quasi_newton_algorithm_linesearch
from naive_random_search_algorithm import naive_random_search_algorithm
from simulated_annealing_algorithm import simulated_annealing_algorithm
from particle_swam_optimization_algorithm import particle_swam_optimization_algorithm
from print_report import print_report
import time

reg_coef = 0.01; # regularization coefficient
# A @ X = B (matrix form)
# Minimize the following objective function:
# X = [x0, ..., xn].T;
# f(X) = norm(A * X - B)^2 =  X.T * (A.T @ A) @ X - 2 * (A.T @ B).T @ X + B.T @ B + reg_coef * X.T @ X; 
N_eqs = 5000; # number of equations
N_pars = 10; # number of parameters.
A = np.random.rand(N_eqs, N_pars);
B = np.random.rand(N_eqs, 1);
Q = A.T @ A;
func = lambda X : X.T @ Q @ X - 2 * (A.T @ B).T @ X + B.T @ B + reg_coef * X.T @ X; 
func_grad = lambda X : 2 * Q @ X - 2 * (A.T @ B) + 2 * reg_coef * X;
func_hessian = lambda X : 2 * Q + 2 * reg_coef;
# Objective function for naive random walk and simulated annealing algorithms
func_error = lambda X : np.linalg.norm((A @ X - B + reg_coef * X.T @ X), axis = 0) ** 2;
# Objective function for particle swarm optimization algorithm (particles along row dimension, axis = 0)
def func_error_ps(X):
    a = np.linalg.norm((X @ A.T - B.T + reg_coef * np.diag(X @ X.T).reshape(-1, 1)), axis = 1) ** 2;
    return a.reshape(-1,1);

# Normal equation with Tikhonov regularization
print('***********************************************************************');
print('Normal equation with Tikhonov regularization');
start = time.time();
X = np.linalg.solve(Q + reg_coef * np.eye(N_pars), A.T @ B);
end = time.time();
print("Norm(X): %0.3f" % (np.linalg.norm(X)));
print('f(X): %0.3f' % (func(X)));
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Gradient algorithm with variable step size (steepest descent)
print('***********************************************************************');
print('Gradient algorithm with variable step size (steepest descent)');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.zeros((N_pars, 1));
start = time.time();
X, report = gradient_algorithm_var_alpha(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Gradient algorithm with fixed step size
print('***********************************************************************');
print('Gradient algorithm with fixed step size');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.zeros((N_pars, 1));   
start = time.time();
X, report = gradient_algorithm_fixed_alpha(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Gradient algorithm with variable step size based on line search
print('***********************************************************************');
print('Gradient algorithm with variable step size based on line search');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.zeros((N_pars, 1));   
start = time.time();
X, report = gradient_algorithm_linesearch(X0, func, func_grad, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Conjugate gradient algorithm with variable step size (steepest descent)
print('***********************************************************************');
print('Conjugate gradient algorithm with variable step size (steepest descent)');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.zeros((N_pars, 1));   
start = time.time();
X, report = conjugate_gradient_algorithm(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Conjugate gradient algorithm with variable step size based on line search
print('***********************************************************************');
print('Conjugate gradient algorithm with variable step size based on line search');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.zeros((N_pars, 1));   
start = time.time();
X, report = conjugate_gradient_algorithm_linesearch(X0, func, func_grad, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Newton's algorithm
print('***********************************************************************');
print('Newton\'s algorithm');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.zeros((N_pars, 1));   
start = time.time();
X, report = newton_algorithm(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Newton's algorithm with variable step size based on line search
print('***********************************************************************');
print('Newton\'s algorithm with variable step size based on line search');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.zeros((N_pars, 1));   
start = time.time();
X, report = newton_algorithm_linesearch(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Quasi-Newton algorithm with variable step size (steepest descent)
print('***********************************************************************');
print('Quasi-Newton algorithm with variable step size (steepest descent)');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.zeros((N_pars, 1));   
start = time.time();
X, report = quasi_newton_algorithm(X0, func, func_grad, func_hessian, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Quasi-Newton algorithm with variable step size based on line search
print('***********************************************************************');
print('Quasi-Newton algorithm with variable step size based on line search');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = np.zeros((N_pars, 1));   
start = time.time();
X, report = quasi_newton_algorithm_linesearch(X0, func, func_grad, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Naive random walk algorithm
print('***********************************************************************');
print('Naive random walk algorithm');
N_iter_max = 10000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
X_lower = -1 * np.ones((N_pars, 1)); # X lower bound   
X_upper = 1 * np.ones((N_pars, 1)); # X upper bound 
alpha = 1.0; # step size
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 
           'x_upper' : X_upper, 'alpha' : alpha};
X0 = X_lower + (X_upper - X_lower) * np.random.rand(X_lower.size, 1); 
start = time.time();
X, report = naive_random_search_algorithm(X0, func_error, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Simulated annealing algorithm
print('***********************************************************************');
print('Simulated annealing algorithm');
N_iter_max = 10000;
tolerance_x = 10e-6;
tolerance_y = 10e-8;
X_lower = -1 * np.ones((N_pars, 1)); # X lower bound   
X_upper = 1 * np.ones((N_pars, 1)); # X upper bound  
alpha = 1; # step size
gamma = 1.0; # controls temperature decay, gamma > 0
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 
           'x_upper' : X_upper, 'alpha' : alpha, 'gamma' : gamma};
X0 = X_lower + (X_upper - X_lower) * np.random.rand(X_lower.size, 1); 
start = time.time();
X, report = simulated_annealing_algorithm(X0, func_error, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Particle swarm optimization algorithm
print('***********************************************************************');
print('Particle swarm optimization algorithm');
N_iter_max = 1000;
tolerance_x = 10e-8;
tolerance_y = 10e-8;
X_lower = -1 * np.ones((N_pars, 1)); # X lower bound   
X_upper = 1 * np.ones((N_pars, 1)); # X upper bound  
d_lower = -0.25; # direction (aka velocity) lower bound
d_upper = 0.25; # direction (aka velocity) upper bound
N_ps = 1000; # number of particles
w = 1.0; # inertial constant, w < 1
c1 = 1.0; # cognitive/independent component, c1 ~ 2
c2 = 0; # social component, c2 ~ 2
alpha = 1.0; # step size
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 
           'x_upper' : X_upper, 'alpha' : alpha, 'd_lower' : d_lower, 'd_upper' : d_upper, 'N_ps' : N_ps, 'w' : w, 'c1' : c1, 'c2' : c2};
start = time.time();
X, report = particle_swam_optimization_algorithm(func_error_ps, options);
end = time.time();
print_report(func, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');