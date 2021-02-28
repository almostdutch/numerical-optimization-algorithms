"""
affine_scaling_algorithm.py

Returns the minimizer of the function
"""

import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)});
 
def affine_scaling_algorithm(A, b, C, options):
           
    m, n = A.shape;
    u = np.random.rand(n, 1);
    v = b - A @ u;
    condition = (v != np.zeros((m, 1)));
    
    # Searching for strictly interior feasible solution
    if (condition.all()):
        C_new = np.concatenate((np.zeros((n, 1)), np.array([[1]])), axis = 0);
        A_new = np.concatenate((A, v), axis = 1);
        b_new = b;
        feasible_solution = np.concatenate((u, np.array([[1]])), axis = 0);
        u, report = affine_scaling_optimization(A_new, b_new, C_new, feasible_solution, options);     
        u = np.delete(u, n).reshape((n, 1));

    # Searching for optimum solution
    X, report = affine_scaling_optimization(A, b, C, u, options);
    
    return (X, report);
    
    
def affine_scaling_optimization(A, b, C, X0, options):
    
    report = {};
    m, n = A.shape;
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    alpha = options['alpha']; # step size
    progress_x = np.zeros((N_iter_max + 1, X0.size));
    progress_y = np.zeros((N_iter_max + 1, 1));
    progress_x[0] = X0.T;
    progress_y[0] = C.T @ X0;
    X_old = X0;
    
    for iter_no in range(1, N_iter_max + 1):      
        D = np.diag(X_old.flatten());
        A_bar = A @ D;
        # Orthogonal projector onto nullspace of A_bar
        P_bar = np.eye(n) - A_bar.T @ np.linalg.inv(A_bar @ A_bar.T) @ A_bar; 
        d = - D @ P_bar @ D @ C; # directional vector (negative gradiant, -C, projected onto N(A))
        
        if ((d != np.zeros((n ,1))).any()):
            mask = (d < 0);
            r = np.min(-X_old[mask] / d[mask]);
        else:
            print('d = 0, exit ..');
            break;
        
        X = X_old + alpha * r * d;
        progress_x[iter_no] = X.T;
        progress_y[iter_no] = C.T @ X;
        
        if (np.linalg.norm(X - X_old) < tolerance_x * np.linalg.norm(X_old)):
            print('Tolerance in X is reached in %d iterations, exit..' % (iter_no));
            break;
            
        if (np.abs(progress_y[iter_no] - progress_y[iter_no - 1]) < tolerance_y * np.abs(progress_y[iter_no - 1])):
            print('Tolerance in Y is reached in %d iterations, exit..' % (iter_no));
            break;
            
        X_old = X;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);