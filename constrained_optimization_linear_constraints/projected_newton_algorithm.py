"""
projected_newton_algorithm.py

Returns the minimizer of the function
X0 - initial guess
func - anonimous function
func_grad - anonimous function gradient
func_hessian - anonimous function hessian
P - projector onto nullspace of A, A @ X = B;
"""

import numpy as np

def projected_newton_algorithm(X0, func, func_grad, func_hessian, P, options):
  
    epsilon = 10e-6
    reg_coeff = 1;
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    X_lower = options['x_lower'];
    X_upper = options['x_upper']; 
    progress_x = np.zeros((X0.size, N_iter_max + 1));
    progress_y = np.zeros((1, N_iter_max + 1));
    progress_x[:, [0]] = X0;
    progress_y[0, [0]] = func(X0);
    X_old = X0;
    
    for iter_no in range(1, N_iter_max + 1):
        grad =  func_grad(X_old);
        
        if (np.linalg.norm(grad) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;                
            
        # Projection onto the N(A)    
        d = P @ np.linalg.solve(func_hessian(X_old) + reg_coeff * np.eye(X_old.size, X_old.size), -grad); # directional vector, Levenberg-Marquardt modification
        X = X_old + d;
        
        # Projection onto the box constraints of X:
        indx_limits = (X < X_lower);
        X[indx_limits] = X_lower[indx_limits];
        indx_limits = (X > X_upper);
        X[indx_limits] = X_upper[indx_limits]; 
        
        progress_x[:, [iter_no]] = X;
        progress_y[0, [iter_no]] = func(X);
        
        if (np.linalg.norm(X - X_old) < tolerance_x * np.linalg.norm(X_old)):
            print('Tolerance in X is reached in %d iterations, exit..' % (iter_no));
            break;
            
        if (np.abs(progress_y[0, [iter_no]] - progress_y[0, [iter_no - 1]]) < tolerance_y * np.abs(progress_y[0, [iter_no - 1]])):
            print('Tolerance in Y is reached in %d iterations, exit..' % (iter_no));
            break;
            
        X_old = X;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);

