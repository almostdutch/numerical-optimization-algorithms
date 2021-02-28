"""
fibonacci_algorithm.py

Returns the reduced uncertainty interval containing the minimizer of the function
func - anonimous function
interval0 - initial uncertainty interval
N_iter - number of iterations
"""

import numpy as np

def fibonacci_algorithm_calc_N_iter(interval0, uncertainty_range_desired):
    
    F = np.empty(shape = (10000), dtype = int);
    F[0] = 0;
    F[1] = 1;
    F[2] = 1;
    
    N = 1;
    epsilon = 10e-6;
    while F[N + 1] < (1 + 2 * epsilon) * (interval0[1] - interval0[0]) / uncertainty_range_desired:
        F[N + 2] = F[N + 1] + F[N];
        N += 1;
    
    N_iter = N - 1; 
    return (N_iter, F);
        
    
def fibonacci_algorithm(func, interval0, N_iter, F):
    
    epsilon = 10e-6;    
    left_limit = interval0[0];
    right_limit = interval0[1];
    
    smaller = 'a';
    a = left_limit + (F[N_iter] / F[N_iter + 1]) * (right_limit - left_limit);
    f_at_a = func(a);
        
    for iter_no in range(N_iter):
        if (iter_no != N_iter):
            rho = 1 - F[N_iter + 1 - iter_no] / F[N_iter + 2 - iter_no];
        else:
            rho = 0.5 - epsilon;
            
        if (smaller == 'a'):
            c = a;
            f_at_c = f_at_a;
            a = left_limit + rho * (right_limit - left_limit);
            f_at_a = func(a);
        else:
            a = c;
            f_at_a = f_at_c;
            c = left_limit + (1 - rho) * (right_limit - left_limit);
            f_at_c = func(c);          
        if (f_at_a < f_at_c):
            right_limit = c;
            smaller = 'a';
        else:
            left_limit = a;
            smaller = 'c';
            
    interval = (left_limit, right_limit);
    return interval;
