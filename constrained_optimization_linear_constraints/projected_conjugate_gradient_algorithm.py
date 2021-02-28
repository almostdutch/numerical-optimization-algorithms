"""
projected_conjugate_gradient_algorithm.py

Returns the minimizer of the function
X0 - initial guess
func - anonimous function
func_grad - anonimous function gradient
func_hessian - anonimous function hessian
P - projector onto nullspace of A, A @ X = B;
"""

import numpy as np

def hestenes_stiefel(grad_old, grad, d):
    beta = (grad.T @ (grad - grad_old)) / (d.T @ (grad - grad_old));
    return beta;
    
def polak_ribiere(grad_old, grad, d):
    beta = (grad.T @ (grad - grad_old)) / (grad_old.T @ grad_old);
    return beta;

def fletcher_reeves(grad_old, grad, d):
    beta = (grad.T @ grad) / (grad_old.T @ grad_old);
    return beta;
    
def powel(grad_old, grad, d):
    beta = (grad.T @ (grad - grad_old)) / (grad_old.T @ grad_old);
    beta = np.max([0, beta]); 
    return beta;    

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

def projected_conjugate_gradient_algorithm(X0, func, func_grad, func_hessian, P, options):
  
    epsilon = 10e-6
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
        grad = func_grad(X_old);
        
        if (np.linalg.norm(grad) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;   
            
        if (iter_no == 1):
            d = -grad; # directional vector
        else:
            # coefficient for calculating conjugate directional vector, this formula valid only for quadratic function
            beta = (grad.T @ func_hessian(X_old) @ d) / (d.T @ func_hessian(X_old) @ d); 
            d = -grad + beta * d; # directional vector
        
        # Projection onto N(A)
        d = P @ d;        
        alpha = -(grad.T @ d) / (d.T @ func_hessian(X_old) @ d); # step size, this formula valid only for quadratic function
        X = X_old + alpha * d;   
        
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
    
def projected_conjugate_gradient_algorithm_linesearch(X0, func, func_grad, P, options):
  
    epsilon = 10e-6
    reset_dir_every_n_iter = X0.size;
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
    grad_old = 0;
    
    for iter_no in range(1, N_iter_max + 1):
        grad = func_grad(X_old);
        
        if (np.linalg.norm(grad) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;   
            
        if (iter_no == 1 or iter_no % (reset_dir_every_n_iter) == 0):
            d = -grad; # # resetting directional vector
        else:
            beta = fletcher_reeves(grad_old, grad, d); # coefficient for calculating conjugate directional vector
            d = -grad + beta * d; # directional vector
        
        # Projection onto N(A)
        d = P @ d;
        alpha = alpha_linesearch_secant_algorithm(X_old, func_grad, d); # step size
        X = X_old + alpha * d; 
        
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
        grad_old = grad;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);

