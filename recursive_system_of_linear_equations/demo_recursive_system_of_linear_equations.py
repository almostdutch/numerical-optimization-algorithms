"""
demo_recursive_system_of_linear_equations.py 

Solving a system of linear equations in a least squares way:
A @ X = B (matrix form)

As more data arrives (A1 @ X = B1), the least squares solution (X*) should be updated by adding 
a correction factor:
    X* = X*_old + correction;
"""

import numpy as np
from rls_algorithm import rls_algorithm
import time

reg_coef = 0.01; # regularization coefficient
# A @ X = B (matrix form)
# Minimize the following objective function:
# X = [x0, ..., xn].T;
# f(X) = norm(A @ X - B)^2 =  X.T * (A.T @ A) @ X - 2 * (A.T @ B).T @ X + B.T @ B + reg_coef * X.T @ X; 
# As more data arrives in the form of A1 and B1, the least squares solution (X*) should be updated
#   by adding a correction factor: X* = X*_old + correction;
N_eqs = 5000; # number of equations
N_pars = 100; # number of parameters.
A = np.random.rand(N_eqs, N_pars);
B = np.random.rand(N_eqs, 1);
Q = A.T @ A;
func = lambda X : X.T @ Q @ X - 2 * (A.T @ B).T @ X + B.T @ B + reg_coef * X.T @ X; 
func_grad = lambda X : 2 * Q @ X - 2 * (A.T @ B) + 2 * reg_coef * X;
func_hessian = lambda X : 2 * Q + 2 * reg_coef;
# New data added:
N_eqs = 500; # number of equations
A1 = np.random.rand(N_eqs, N_pars);
B1 = np.random.rand(N_eqs, 1);

# Normal equation with Tikhonov regularization
print('***********************************************************************');
print('Normal equation with Tikhonov regularization');
start = time.time();
X = np.linalg.solve(Q + reg_coef * np.eye(N_pars), A.T @ B);
end = time.time();
print("Norm(X): %0.3f" % (np.linalg.norm(X)));
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Recursive algorithm for updating the minimizer as more data arrives
print('***********************************************************************');
print('Recursive algorithm for updating the minimizer as more data arrives');
start = time.time();
P = np.linalg.inv(Q + reg_coef * np.eye(N_pars));
X, P = rls_algorithm(X, P, A1, B1);
end = time.time();
print("Norm(X): %0.3f" % (np.linalg.norm(X)));
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');
