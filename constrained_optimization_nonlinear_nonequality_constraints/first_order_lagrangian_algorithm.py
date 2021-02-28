"""
first_order_lagrangian_algorithm.py

Returns the minimizer of the function
X0 - initial guess for X;
Lambda0 - initial guess for Lambda;
func - anonimous function (objective function)
Dy - anonimous function gradient (objective function)
Dh - anonimous function gradient (equality constraints function)
Dg - anonimous function gradient (nonequality constraints function)
h - anonimous function (equality constraints function)
g - anonimous function (nonequality constraints function)
"""

import numpy as np

def first_order_lagrangian_algorithm(X0, Lambda0, Mu0, func, Dy, Dh, Dg, h, g, options):

    epsilon = 10e-3;
    alpha = 0.01; # step size for X
    beta = 0.01; # step size for Lambda
    gamma = 0.01 # step size for Mu
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
    m = Lambda0.shape[0];
    n = X0.shape[0];
    X_old = X0;
    Lambda_old = Lambda0;
    Mu_old = Mu0;
    W_old = np.concatenate((X0, Lambda0, Mu0), axis=0);
        
    for iter_no in range(1, N_iter_max + 1):
               
        dX = -(Dy(X_old) + Dh(X_old) @ Lambda_old + Dg(X_old) @ Mu_old); # direction for X
        dLambda = h(X_old); # direction for Lambda
        dMu = g(X_old); # direction for Mu

        X = X_old + alpha * dX;
        Lambda = Lambda_old + beta * dLambda;
        Mu = Mu_old + gamma * dMu;
        indx_limits = (Mu < 0);
        Mu[indx_limits] = 0;
        
        # # # Projection onto the box constraints of X (causes numerical instability):
        # indx_limits = (X < X_lower);
        # X[indx_limits] = X_lower[indx_limits];
        # indx_limits = (X > X_upper);
        # X[indx_limits] = X_upper[indx_limits]; 
                
        W = np.concatenate((X, Lambda, Mu), axis=0);
        progress_x[:, [iter_no]] = X;
        progress_y[0, [iter_no]] = func(X);

        if (np.linalg.norm(h(X)) < epsilon and np.linalg.norm(g(X)) < epsilon):
            print('Tolerance in h(X) and g(X) is reached in %d iterations, exit..' % (iter_no));
            break;
        
        # if (np.linalg.norm(W - W_old) < tolerance_x * np.linalg.norm(W_old)):
        #     print('Tolerance in X is reached in %d iterations, exit..' % (iter_no));
        #     break;
            
        # if (np.abs(progress_y[0, [iter_no]] - progress_y[0, [iter_no - 1]]) < tolerance_y * np.abs(progress_y[0, [iter_no - 1]])):
        #     print('Tolerance in Y is reached in %d iterations, exit..' % (iter_no));
        #     break;

        X_old = X;
        Lambda_old = Lambda;
        Mu_old = Mu;
        W_old = W;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);

