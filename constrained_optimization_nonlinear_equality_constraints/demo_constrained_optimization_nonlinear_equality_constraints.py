"""
demo_constrained_optimization_nonlinear_equality_constraints.py

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
    
(Non)linear equality constraints:
    Subject to (non)linear equality constraints: h(X);
    h(X) = [h1(X), ..., hn(X)].T is vector of constraints;
    m < n;

First order Lagrange algorithm

Lagrangian function:
l(X, Lambda) = f(X) + Lambda.T @ h(X);
Lambda = [lambda1, ..., lambdan].T is vector of langrange multipliers;

Update equation for X:
    X = X_old + alpha * d;
    alpha is step size (<<1);
    d is direction:
        d = - (Df(X_old) + Dh(X_old).T @ Lambda);
        
Update equation for Lambda:
    Lambda = Lambda_old + beta * d;
    beta is step size (<<1);
    d is direction:
        d = h(X_old);

If alpha and beta are small enough, the algorithm will converge to a fixed point, which satisfies the Lagrange condition.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from first_order_lagrangian_algorithm import first_order_lagrangian_algorithm
from print_report import print_report
from plot_progress_y import plot_progress_y
from plot_progress_x import plot_progress_x
import time

reg_coeff =10; # regularization coefficient
# Minimize the following objective function:
# X = [x0, x1].T;
# f(X) = 1 / 2 * X.T @ Q @ X + X.T @ B + C + reg_coeff * X.T @ X;
Q = np.array([[8, 2 * np.sqrt(2)], [2 * np.sqrt(2), 10]]);
B = np.array([[3], [6]]);
C = 24;
func = lambda X : 1 / 2 * X.T @ Q @ X + X.T @ B + C + reg_coeff * X.T @ X; 
Dy = lambda X : Q @ X + B + 2 * reg_coeff * X;

# Box constraints:
X_lower = np.array([[-20], [-20]]); # X lower bound   
X_upper = np.array([[20], [20]]); # X upper bound 

# (Non)linear equality constraints:
# h(X) = x1 * x2 - 20 = 0
h = lambda X : X[0] * X[1] - 20; 
Dh = lambda X : np.array([X[1], X[0]]);
# Initial guess (subject to the equality constraints):
X0 = np.array([[-50], [-0.4]]);   
Lambda0 = np.array([[0]]);

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

# First order Lagrangian algorithm
print('***********************************************************************');
print('First order Lagrangian algorithm');
N_iter_max = 10000;
tolerance_x = 10e-6;
tolerance_y = 10e-6;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'x_lower' : X_lower, 'x_upper' : X_upper};
start = time.time();
X, report = first_order_lagrangian_algorithm(X0, Lambda0, func, Dy, Dh, h, options);
end = time.time();
print_report(func, report);
# Plot path to X* for Y
algorithm_name = 'First order Lagrangian algorithm';
plot_progress_y(algorithm_name, report);
# Plot path to X* for X
plot_progress_x(X_data, Y_data, Z_data, algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');