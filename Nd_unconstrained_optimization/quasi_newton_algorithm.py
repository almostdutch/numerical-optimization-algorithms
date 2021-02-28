"""
quasi_newton_algorithm.py

Returns the minimizer of the function
X0 - initial guess
func - anonimous function
func_grad - anonimous function gradient
N_iter_max - max number of iterations
"""

import numpy as np

def srs(h_old, delta_X, delta_grad):
    
    delta_g = delta_grad;
    h_old_times_delta_g = h_old @ delta_g;
    top = np.outer(delta_X - h_old_times_delta_g, (delta_X - h_old_times_delta_g).T);
    bottom = delta_g.T @ (delta_X - h_old_times_delta_g);
    delta_h = top / bottom;
    
    return delta_h;

def dfp(h_old, delta_X, delta_grad):
    
    delta_g = delta_grad;
    h_old_times_delta_g = h_old @ delta_g;
    a_top = np.outer(delta_X, delta_X.T);
    a_bottom = delta_X.T @ delta_g;
    b_top = np.outer(h_old_times_delta_g, h_old_times_delta_g.T);
    b_bottom = delta_g.T @ h_old_times_delta_g;
    delta_h = a_top / a_bottom - b_top / b_bottom;
    
    return delta_h;

def bfgs(h_old, delta_X, delta_grad):
    
    delta_g = delta_grad;
    h_old_times_delta_g = h_old @ delta_g;
    delta_gT_times_delt_X = delta_g.T @ delta_X;
    a_top = delta_g.T @ h_old_times_delta_g;
    a_bottom = delta_gT_times_delt_X;
    b_top = np.outer(delta_X, delta_X.T);
    b_bottom = delta_gT_times_delt_X
    c_top = np.outer(delta_X,  h_old_times_delta_g.T) + np.outer(h_old_times_delta_g, delta_X.T);
    c_bottom = delta_gT_times_delt_X;
    delta_h = 1 + (a_top / a_bottom) * (b_top / b_bottom) - (c_top / c_bottom); 

    return delta_h;

def alpha_linesearch_secant_algorithm(X, func_grad, d):
    
    epsilon = 10e-3;
    N_iter_max = 100;
    alpha_curr = 0; # step size
    alpha = 0.5; # step size
    obj_func_dir_derivative_at_X0 = func_grad(X).T @ d; # directional derivative
    obj_func_dir_derivative = obj_func_dir_derivative_at_X0; # directional derivative
    
    for iter_no in range(1, N_iter_max + 1):
        alpha_old = alpha_curr; # step size
        alpha_curr = alpha; # step size
        obj_func_dir_derivative_old = obj_func_dir_derivative; # directional derivative
        obj_func_dir_derivative = func_grad(X + alpha_curr * d).T @ d; # directional derivative
        
        if (obj_func_dir_derivative < epsilon):
            break;                
            
        alpha = (obj_func_dir_derivative * alpha_old - obj_func_dir_derivative_old * alpha_curr) / (obj_func_dir_derivative - obj_func_dir_derivative_old);
        
        if (np.abs(obj_func_dir_derivative) < epsilon * np.abs(obj_func_dir_derivative_at_X0)): 
            break;

    return alpha;

def quasi_newton_algorithm(X0, func, func_grad, func_hessian, options):

    epsilon = 10e-6
    reset_dir_every_n_iter = X0.size;
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    progress_x = np.zeros((X0.size, N_iter_max + 1));
    progress_y = np.zeros((1, N_iter_max + 1));
    progress_x[:, [0]] = X0;
    progress_y[0, [0]] = func(X0);
    X_old = X0;
    h_old = np.eye(X0.size, X0.size); # approximation of inv(hessian)
    
    for iter_no in range(1, N_iter_max + 1):
        grad_old = func_grad(X_old);
        
        if (np.linalg.norm(grad_old) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;   
            
        if (iter_no == 1 or iter_no % (reset_dir_every_n_iter) == 0):
            d = -grad_old; # resetting directional vector
        else:
            d = -h_old @ grad_old; # conjugate directional vector
        
        alpha = -(grad_old.T @ d) / (d.T @ func_hessian(X_old) @ d); # step size, this formula valid only for quadratic function
        X = X_old + alpha * d;        
        progress_x[:, [iter_no]] = X;
        progress_y[0, [iter_no]] = func(X);
        
        if (np.linalg.norm(X - X_old) < tolerance_x * np.linalg.norm(X_old)):
            print('Tolerance in X is reached in %d iterations, exit..' % (iter_no));
            break;
            
        if (np.abs(progress_y[0, [iter_no]] - progress_y[0, [iter_no - 1]]) < tolerance_y * np.abs(progress_y[0, [iter_no - 1]])):
            print('Tolerance in Y is reached in %d iterations, exit..' % (iter_no));
            break;
                    
        grad = func_grad(X);
        delta_X = X - X_old;
        delta_grad = grad - grad_old;
        delta_h = bfgs(h_old, delta_X, delta_grad);
        h = h_old + delta_h;      
        X_old = X;
        h_old = h;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);

def quasi_newton_algorithm_linesearch(X0, func, func_grad, options):
  
    epsilon = 10e-6
    reset_dir_every_n_iter = X0.size;
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    progress_x = np.zeros((X0.size, N_iter_max + 1));
    progress_y = np.zeros((1, N_iter_max + 1));
    progress_x[:, [0]] = X0;
    progress_y[0, [0]] = func(X0);
    X_old = X0;
    h_old = np.eye(X0.size, X0.size); # approximation of inv(hessian)
    
    for iter_no in range(1, N_iter_max + 1):
        grad_old = func_grad(X_old);
        
        if (np.linalg.norm(grad_old) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;   
            
        if (iter_no == 1 or iter_no % (reset_dir_every_n_iter) == 0):
            d = -grad_old; # resetting directional vector
        else:
            d = -h_old @ grad_old; # conjugate directional vector
        
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
        
        grad = func_grad(X);
        delta_X = X - X_old;
        delta_grad = grad - grad_old;
        delta_h = bfgs(h_old, delta_X, delta_grad);
        h = h_old + delta_h;      
        X_old = X;
        h_old = h;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);
