"""
demo_linear_programming.py 

Linear objective function and linear constraints

Minimize f(X) = C.T @ X, where
C.T = [c1, c2, ..., cn] is vector of cost coefficients 
X = [x1, x2, ..., xn].T is vector of variables

Subject to equality constraints: A @ X = b, where X >= 0;
Nonequality constraints can be converted to equality constraints 
    by introducing slack variables:
x1 + x2 <= whatever;
x1 + x2 + x3 = whatever;
    OR
x1 + x2 >= whatever;
x1 + x2 - x3 = whatever;

C = vector of cost coefficients
indx_cols = list of indices of basic (linear independent) columns of A
b = (basic feasible) solution
m = number of equations
n = number of variables

Simplex algorithm
    
Revised simplex algorithm

Affine scaling algorithm

Simplex algorithm with Gomory cuts (int solutions)
"""

import numpy as np
from simplex_algorithm import simplex_algorithm
from revised_simplex_algorithm import revised_simplex_algorithm
from affine_scaling_algorithm import affine_scaling_algorithm
from simplex_algorithm_int import simplex_algorithm_int
import time

"""
# Problem # 1
# Maximize f(X) = 6x1 + 4x2 + 7x3 + 5x4
# or Minimize f(X) = -6x1 - 4x2 - 7x3 - 5x4
# Constraints:
#     x1 + 2x2 + x3 + 2x4 <= 20
#     6x1 + 5x2 + 3x3 + 2x4 <= 100
#     3x1 + 4x2 + 9x3 + 12x4 <= 75
    
A = np.array([[1, 2, 1, 2, 1, 0, 0], [6, 5, 3, 2, 0, 1, 0], [3, 4, 9, 12, 0, 0, 1]]);
b = np.array([[20], [100], [75]]);
C = np.array([[-6], [-4], [-7], [-5], [0], [0], [0]]);
indx_cols = [4, 5, 6];
"""
A = np.array([[1, 2, 1, 2, 1, 0, 0], [6, 5, 3, 2, 0, 1, 0], [3, 4, 9, 12, 0, 0, 1]]);
b = np.array([[20], [100], [75]]);
C = np.array([[-6], [-4], [-7], [-5], [0], [0], [0]]);
indx_cols = [0, 1, 2];


"""
# Problem # 2 (cyclic unless bland's rule used)
# Minimize f(X) = -3x4/4 + 20x5 - x6/2 + 6x7 
# Constraits:
#     x1 + x4/4 - 8x5 - x6 + 9x7 = 0
#     x2 + x4/2 - 12x5 - x6/2 + 3x7 = 0 
#     x3 + x6 = 1

A = np.array([[1, 0, 0, 1/4, -8, -1, 9], [0, 1, 0, 1/2, -12, -1/2, 3], [0, 0, 1, 0, 0, 1, 0]]);
b = np.array([[0], [0], [1]]);
C = np.array([[0], [0], [0], [-3/4], [20], [-1/2], [6]]);
indx_cols = [0, 1, 2];
"""

# Simplex algorithm
print('***********************************************************************');
print('Simplex algorithm');
b_print = False;
b_blands_rule = False;
options = {'b_print' : b_print, 'b_blands_rule' : b_blands_rule};
start = time.time();
X, cost = simplex_algorithm(A, b, C, indx_cols, options);
print('X:');
print(X);
end = time.time();
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Revised simplex algorithm
print('***********************************************************************');
print('Revised simplex algorithm');
b_print = False;
b_blands_rule = False;
options = {'b_print' : b_print, 'b_blands_rule' : b_blands_rule};
start = time.time();
X, cost = revised_simplex_algorithm(A, b, C, indx_cols, options);
print('X:');
print(X);
end = time.time();
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');

# Affine scaling algorithm
print('***********************************************************************');
print('Affine scaling algorithm');
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-6;
alpha = 0.99; # step size (alpha < 1)
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max, 'alpha' : alpha};
start = time.time();
X, report = affine_scaling_algorithm(A, b, C, options);
print('X:');
print(X);
end = time.time();
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n'); 
    
# Simplex algorithm with Gomory cuts
print('***********************************************************************');
print('Simplex algorithm with Gomory cuts (int solutions)');
b_print = False;
b_blands_rule = False;
options = {'b_print' : b_print, 'b_blands_rule' : b_blands_rule};
start = time.time();
X, cost = simplex_algorithm_int(A, b, C, indx_cols, options);
print('X:');
print(X);
end = time.time();
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n');