"""
bracketing.py

Returns the uncertainty interval containing the minimizer of the function
X0 - initial guess
func - anonimous function
func_1st_der - anonimous function 1st derivative
"""

import numpy as np

def bracketing(X0, func, func_1st_der):
    
    epsilon = 0.5;
    d = - func_1st_der(X0) / np.linalg.norm(func_1st_der(X0)); # directional unit vector
    X = np.empty(shape = (100));
    F_at_X = np.empty(shape = (100));
    X[0] = X0;
    F_at_X[0] = func(X[0]);
    X[1] = X0 + 2 * epsilon * d;
    F_at_X[1] = func(X[1]);
    
    for ii in range(2, 100, 1):
        X[ii] = X0 + 2 * ii * epsilon * d;
        F_at_X[ii] = func(X[ii]);
        if F_at_X[ii - 1] < F_at_X[ii - 2] and F_at_X[ii - 1] < F_at_X[ii] :
            break;

    if (d > 0):
        interval = (X[ii - 2], X[ii]);
    else:
        interval = (X[ii], X[ii - 2]);    
    return interval;
