"""
levenberg_marquardt_algorithm.py

Returns the minimizer of the function
X0 - initial guess
func_residual- anonimous function residual
func_jacobian- anonimous function jacobian
"""

import numpy as np

def levenberg_marquardt_algorithm(X0, func_residual, func_jacobian, options):
  
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
    progress_y[0, [0]] = np.sum(func_residual(X0));
    X_old = X0;
    
    for iter_no in range(1, N_iter_max + 1):
        jacobian = func_jacobian(X_old);
        residuals = func_residual(X_old); 
            
        #d = np.linalg.solve(jacobian, residuals); # Newton's algorithm
        #d = np.linalg.solve(jacobian.T @ jacobian, jacobian.T @ residuals); # Newton's algorithm
        d = np.linalg.solve(jacobian.T @ jacobian + reg_coeff * np.eye(X_old.size, X_old.size), jacobian.T @ residuals); # with Levenberg-Marquardt modification
        X = X_old + d.reshape(X_old.shape);
        
        # Projection onto constrained parameter space
        indx_limits = (X < X_lower);
        X[indx_limits] = X_lower[indx_limits];
        indx_limits = (X > X_upper);
        X[indx_limits] = X_upper[indx_limits];
        
        progress_x[:, [iter_no]] = X;
        progress_y[0, [iter_no]] = np.sum(func_residual(X));
        
        if (np.linalg.norm(X - X_old) < tolerance_x * np.linalg.norm(X_old)):
            print('Tolerance in X is reached in %d iterations, exit..' % (iter_no));
            break;
            
        if (np.abs(progress_y[0, [iter_no]] - progress_y[0, [iter_no - 1]]) < tolerance_y * np.abs(progress_y[0, [iter_no - 1]])):
            print('Tolerance in Y is reached in %d iterations, exit..' % (iter_no));
            break;
            
        X_old = X;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);
