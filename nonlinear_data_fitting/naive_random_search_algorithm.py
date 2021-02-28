"""
naive_random_search_algorithm.py

Returns the minimizer of the function
X0 - initial guess
func - anonimous function
"""

import numpy as np
np.random.seed(10);

def naive_random_search_algorithm(X0, func, options):

    report = {};
    N_iter_max = options['N_iter_max']; 
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    X_lower = options['x_lower'];
    X_upper = options['x_upper'];   
    alpha = options['alpha']; # step size     
    progress_x = np.zeros((X0.size, N_iter_max + 1)); # vector of positions
    progress_y = np.zeros((1, N_iter_max + 1));
    progress_x[:, [0]] = X0;
    progress_y[0, [0]] = func(X0);
    X_best = progress_x[:, [0]];
    Y_best = progress_y[0, [0]];
    
    X_old = X_best;
    for iter_no in range(1, N_iter_max + 1):
        X = X_old + alpha * (2 * np.random.rand(X0.size,1) - 1);
        alpha *= 0.99; # to speed up convergence
        
        # Projection onto constrained parameter space
        indx_limits = (X < X_lower);
        X[indx_limits] = X_lower[indx_limits];
        indx_limits = (X > X_upper);
        X[indx_limits] = X_upper[indx_limits]; 
        
        progress_x[:, [iter_no]] = X;
        progress_y[0, [iter_no]] = func(X);        
        
        if (np.linalg.norm(progress_x[:, [iter_no]] - X_best) < tolerance_x * np.linalg.norm(X_best)):
            print('Tolerance in X is reached in %d iterations, exit..' % (iter_no));
            break;
            
        if (np.abs(progress_y[0, [iter_no]] - Y_best) < tolerance_y * np.abs(Y_best)):
            print('Tolerance in Y is reached in %d iterations, exit..' % (iter_no));
            break;
            
        if (progress_y[0, [iter_no]] < Y_best):
            X_best = progress_x[:, [iter_no]];
            Y_best = progress_y[0, [iter_no]];
        
        X_old = X_best;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);
