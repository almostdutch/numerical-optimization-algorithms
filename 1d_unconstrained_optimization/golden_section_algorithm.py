"""
golden_section_algorithm.py

Returns the reduced uncertainty interval containing the minimizer of the function
func - anonimous function
interval0 - initial uncertainty interval
N_iter - number of iterations
"""

import math
import numpy as np

def golden_section_algorithm_calc_N_iter(interval0, uncertainty_range_desired):
    
    N_iter = math.ceil(math.log(uncertainty_range_desired / (interval0[1] - interval0[0]), 0.618)); 
    
    return N_iter;
        
    
def golden_section_algorithm(func, interval0, N_iter):
    
    rho = (3 - np.sqrt(5)) / 2;
    left_limit = interval0[0];
    right_limit = interval0[1];
    
    smaller = 'a';
    a = left_limit + (1 - rho) * (right_limit - left_limit);
    f_at_a = func(a);
        
    for iter_no in range(N_iter):
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
