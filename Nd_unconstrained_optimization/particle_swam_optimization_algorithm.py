"""
particle_swam_optimization_algorithm.py

Returns the minimizer of the function
func_error_ps - anonimous function (vectorized for multiple particles)
"""

import numpy as np
import numpy.matlib
np.random.seed();

def particle_swam_optimization_algorithm(func_error_ps, options):

    report = {};
    N_ps = options['N_ps']; # number of particles
    N_iter_max = options['N_iter_max']; 
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    X_lower = options['x_lower'].T;
    ps_X_lower = np.matlib.repmat(X_lower, N_ps, 1);
    X_upper = options['x_upper'].T;   
    ps_X_upper = np.matlib.repmat(X_upper, N_ps, 1);
    d_lower = options['d_lower'];
    d_upper = options['d_upper'];      
    X0 = X_lower + (X_upper - X_lower) * np.random.rand(N_ps, X_lower.size); # initial positions of all particles
    d = d_lower + (d_upper - d_lower) * np.random.rand(N_ps, X_lower.size); # initial directions (aka velocities) of all particles
    alpha = options['alpha']; # step size     
    w = options['w'];
    c1 = options['c1'];
    c2 = options['c2'];
    progress_x_ps = np.zeros((N_iter_max + 1, N_ps, X_lower.size)); # vector of positions of all particles
    progress_y_ps = np.zeros((N_iter_max + 1, N_ps, 1));
    progress_x = np.zeros((N_iter_max + 1, X_lower.size)); # vector of global best positions
    progress_y = np.zeros((N_iter_max + 1, 1));
    ps_X_best = np.zeros((N_ps, X_lower.size)); # local best positions of all particles
    ps_Y_best = np.zeros((N_ps, 1));
    X_best = np.zeros((1, X_lower.size)); # global best position of all particles
    
    progress_x_ps[0] = X0;
    progress_y_ps[0] = func_error_ps(X0);
    ps_X_best = progress_x_ps[0];
    ps_Y_best = progress_y_ps[0];  
    
    indx_min = np.argmin(ps_Y_best, axis = 0);
    X_best = ps_X_best[indx_min];
    progress_x[0] = ps_X_best[indx_min];
    progress_y[0] = ps_Y_best[indx_min];
    
    X_old = ps_X_best;
    for iter_no in range(1, N_iter_max + 1):
        d =  w * d + c1 * np.random.rand(N_ps, 1) * (ps_X_best - X_old) + c2 * np.random.rand(N_ps, 1) * (X_best - X_old);
        
        # Projection onto constrained parameter space
        d[d < d_lower] = d_lower;
        d[d > d_upper] = d_upper;
               
        alpha *= 0.99; # to speed up convergence
        X = X_old + alpha * d;
              
        # Projection onto constrained parameter space
        indx_limits = (X < ps_X_lower);
        X[indx_limits] = ps_X_lower[indx_limits];
        indx_limits = (X > ps_X_upper);
        X[indx_limits] = ps_X_upper[indx_limits];  
        
        progress_x_ps[iter_no] = X;
        progress_y_ps[iter_no] = func_error_ps(X);
            
        indx_update = (progress_y_ps[iter_no] < ps_Y_best).ravel();
        ps_X_best[indx_update] = progress_x_ps[iter_no][indx_update]; # updating local best positions
        ps_Y_best[indx_update] = progress_y_ps[iter_no][indx_update]; 
                
        indx_min = np.argmin(ps_Y_best, axis = 0);
        X_best = ps_X_best[indx_min]; # updating global best position
        progress_x[iter_no] = ps_X_best[indx_min];
        progress_y[iter_no] = ps_Y_best[indx_min];
        
        if (np.linalg.norm(progress_x[iter_no].reshape(X_lower.size, 1) - progress_x[iter_no - 1].reshape(X_lower.size, 1), axis = 0) \
            < tolerance_x * np.linalg.norm(progress_x[iter_no].reshape(X_lower.size, 1), axis = 0)):
            print('Tolerance in X is reached in %d iterations, exit..' % (iter_no));
            break;
            
        if (np.abs(progress_y[iter_no] - progress_y[iter_no - 1]) < tolerance_y * np.abs(progress_y[iter_no - 1])):
            print('Tolerance in Y is reached in %d iterations, exit..' % (iter_no));
            break;
            
        X_old = X_best;
                 
    X_best = X_best.T;
    progress_x = progress_x.T;
    progress_y = progress_y.T;
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X_best' : X_best, 'progress_x_ps' : progress_x_ps, 
              'progress_y_ps' : progress_y_ps, 'progress_x' : progress_x, 'progress_y' : progress_y};

    return (X_best, report);


