"""
secant_algorithm.py

Returns the minimizer of the function
X0 - initial guess
func - anonimous function
func_1st_der - anonimous function 1st derivative
N_iter_max - max number of iterations
"""

import numpy as np

def secant_algorithm(X0, func, func_1st_der, options):
  
    epsilon = 10e-6
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    progress_x = np.zeros((1, N_iter_max + 1));
    progress_y = np.zeros((1, N_iter_max + 1));
    progress_x[:, [0]] = X0;
    progress_y[0, [0]] = func(X0);
    X_old = X0;
 
    for iter_no in range(1, N_iter_max + 1):
        grad = func_1st_der(X_old);
        
        if (np.linalg.norm(grad) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;
            
        func_approx_2nd_der = (grad - func_1st_der(X_old - epsilon)) / (epsilon);
        X = X_old - grad / func_approx_2nd_der;
        progress_x[:, [iter_no]] = X;
        progress_y[0, [iter_no]] = func(X);
        
        if (np.abs(X - X_old) < tolerance_x * np.abs(X_old)):
            print('Tolerance in X is reached in %d iterations, exit..' % (iter_no));
            break;
            
        if (np.abs(progress_y[0, [iter_no]] - progress_y[0, [iter_no - 1]]) < tolerance_y * np.abs(progress_y[0, [iter_no - 1]])):
            print('Tolerance in Y is reached in %d iterations, exit..' % (iter_no));
            break;

        X_old = X;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);
