"""
newton_algorithm.py

Returns the minimizer of the function
X0 - initial guess
func - anonimous function
func_grad - anonimous function gradient
func_hessian - anonimous function hessian
"""

import numpy as np

def alpha_linesearch_secant_algorithm(X, func_grad, d):
    
    epsilon = 10e-3;
    N_iter_max = 100;
    alpha_curr = 0;
    alpha = 0.5;
    obj_func_dir_derivative_at_X0 = func_grad(X).T @ d; # directional derivative
    obj_func_dir_derivative = obj_func_dir_derivative_at_X0; # directional derivative
    
    for iter_no in range(1, N_iter_max + 1):
        alpha_old = alpha_curr;
        alpha_curr = alpha;
        obj_func_dir_derivative_old = obj_func_dir_derivative; # directional derivative
        obj_func_dir_derivative = func_grad(X + alpha_curr * d).T @ d; # directional derivative
        
        if (obj_func_dir_derivative < epsilon):
            break;                
            
        alpha = (obj_func_dir_derivative * alpha_old - obj_func_dir_derivative_old * alpha_curr) / (obj_func_dir_derivative - obj_func_dir_derivative_old);
        
        if (np.abs(obj_func_dir_derivative) < epsilon * np.abs(obj_func_dir_derivative_at_X0)): 
            break;
            
        if (iter_no == N_iter_max):
            print('Terminating line search with number of iterations: %d' % (iter_no));                 

    return alpha;

def newton_algorithm(X0, func, func_grad, func_hessian, options):
  
    epsilon = 10e-6
    reg_coeff = 1;
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    progress_x = np.zeros((X0.size, N_iter_max + 1));
    progress_y = np.zeros((1, N_iter_max + 1));
    progress_x[:, [0]] = X0;
    progress_y[0, [0]] = func(X0);
    X_old = X0;
    
    for iter_no in range(1, N_iter_max + 1):
        grad = func_grad(X_old);
        
        if (np.linalg.norm(grad) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;                 
            
        d = np.linalg.solve(func_hessian(X_old) + reg_coeff * np.eye(X_old.size, X_old.size), -grad); # directional vector, Levenberg-Marquardt modification
        X = X_old + d;
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

def newton_algorithm_linesearch(X0, func, func_grad, func_hessian, options):
  
    epsilon = 10e-6
    reg_coeff = 1;
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    progress_x = np.zeros((X0.size, N_iter_max + 1));
    progress_y = np.zeros((1, N_iter_max + 1));
    progress_x[:, [0]] = X0;
    progress_y[0, [0]] = func(X0);
    X_old = X0;
    
    for iter_no in range(1, N_iter_max + 1):
        grad = func_grad(X_old); 
        
        if (np.linalg.norm(grad) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;                 
            
        d = np.linalg.solve(func_hessian(X_old) + reg_coeff * np.eye(X_old.size, X_old.size), -grad); # directional vector, Levenberg-Marquardt modification
        alpha = alpha_linesearch_secant_algorithm(X_old, func_grad, d); # step size
        X = X_old + alpha * d; 
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
