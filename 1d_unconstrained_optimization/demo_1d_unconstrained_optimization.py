"""
demo_1d_unconstrained_optimization.py.py

Minimize a function of one variable:
Minimize f(X)

Golden section algorithm
    Only special requirements.
    
Fibonacci algorithm
    Only special requirements.
    
Newton algorithm
    Requires 1st and 2nd derivatives.
    
Secant algorithm
    Requires only 1st derivative (2nd derivative estimated using 1st derivative).
"""

import numpy as np
import matplotlib.pyplot as plt
from bracketing import bracketing
from golden_section_algorithm import golden_section_algorithm, golden_section_algorithm_calc_N_iter
from fibonacci_algorithm import fibonacci_algorithm, fibonacci_algorithm_calc_N_iter
from newton_algorithm import newton_algorithm
from secant_algorithm import secant_algorithm
from print_report import print_report
from plot_progress_y import plot_progress_y
import time

reg_coeff = 0.01; # regularization coefficient
# Minimize the following objective function:
# f(X) = X^4 - 14 * X^3 + 60 * X^2 - 70 * X + reg_coeff * X^2;
func = lambda X : np.power(X, 4) - 14 * np.power(X, 3) + 60 * np.power(X, 2) -70 * X + reg_coeff * np.power(X, 2); 
func_1st_der = lambda X : 4 * np.power(X, 3) - 3 * 14 * np.power(X, 2) + 2 * 60 * np.power(X, 1) -70 + 2 * reg_coeff * X;
func_2nd_der = lambda X : 3 * 4 * np.power(X, 2) - 2 * 3 * 14 * np.power(X, 1) + 2 * 60 + 2 * reg_coeff;
 
# Plot the curve
fig = plt.figure();
X = np.arange(-5, 12, 0.5);
Y = np.empty(shape = X.size);
for ii in range(X.size):
    Y[ii] = func(X[ii]);

plt.plot(X, Y)
plt.show()

# Bracketing
print('***********************************************************************')
print('Bracketing')
X0 = -20;
start = time.time();
interval0 = bracketing(X0, func, func_1st_der);
end = time.time();
print("Minimizer lies in interval [%0.3f %0.3f]" % (interval0[0], interval0[1]));
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n')

# Golden section algorithm
print('***********************************************************************')
print('Golden section algorithm')
uncertainty_range_desired = 0.01;
start = time.time();
N_iter = golden_section_algorithm_calc_N_iter(interval0, uncertainty_range_desired);
interval = golden_section_algorithm(func, interval0, N_iter);
end = time.time();
print("Desired uncertainty range: %0.2f" % (uncertainty_range_desired));
print("Algorithm converged in %d iteration" % (N_iter));
print("Initial uncertainty interval: [%0.3f %0.3f]" % (interval0[0], interval0[1]));
print("Reduced uncertainty interval: [%0.3f %0.3f]" % (interval[0], interval[1]));
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n')

# Fibonacci algorithm
print('***********************************************************************')
print('Fibonacci algorithm')
uncertainty_range_desired = 0.01;
start = time.time();
N_iter, F = fibonacci_algorithm_calc_N_iter(interval0, uncertainty_range_desired);
interval = fibonacci_algorithm(func, interval0, N_iter, F);
end = time.time();
print("Desired uncertainty range: %0.2f" % (uncertainty_range_desired));
print("Algorithm converged in %d iteration" % (N_iter));
print("Initial uncertainty interval: [%0.3f %0.3f]" % (interval0[0], interval0[1]));
print("Reduced uncertainty interval: [%0.3f %0.3f]" % (interval[0], interval[1]));
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n')

# Newton's algorithm
print('***********************************************************************')
print('Newton\'s algorithm')
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-6;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = 0;
start = time.time();
X, report = newton_algorithm(X0, func, func_1st_der, func_2nd_der, options);
end = time.time();
print_report(report);
# Plot path to X* for Y
algorithm_name = 'Newton\'s algorithm';
plot_progress_y(algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n')

# Secant algorithm
print('***********************************************************************')
print('Secant algorithm')
N_iter_max = 1000;
tolerance_x = 10e-6;
tolerance_y = 10e-6;
options = {'tolerance_x' : tolerance_x, 'tolerance_y' : tolerance_y, 'N_iter_max' : N_iter_max};
X0 = 0;
start = time.time();
X, report = secant_algorithm(X0, func, func_1st_der, options);
end = time.time();
print_report(report);
# Plot path to X* for Y
algorithm_name = 'Secant algorithm';
plot_progress_y(algorithm_name, report);
print('Elapsed time [s]: %0.5f' % (end - start));
print('***********************************************************************\n')
