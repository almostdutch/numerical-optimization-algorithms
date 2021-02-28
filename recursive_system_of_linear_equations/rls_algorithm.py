"""
rls_algorithm.py

Returns the updated minimizer of the objective function
X_old - initial minimizer (A * X = B)
P_old - inv(A.T @ A)
A1 - new data
B1 - new data
"""

import numpy as np

def rls_algorithm(X_old, P_old, A1, B1):
  
    if (B1.size == 1):
        I = 1;
        a = A1 @ P_old;
        aT = a.T;
        P = P_old - (aT @ a) / (1 + a @ A1.T);
        # P = P_old - (P_old @ A1.T @ A1 @ P_old) / (1 + A1 @ P_old @ A1.T);
        correction = P @ A1.T @ (B1 - A1 @ X_old);
    else:
        I = np.eye(A1.shape[0]);
        a = A1 @ P_old;
        aT = a.T;     
        P = P_old - aT @ np.linalg.inv(I + A1 @ aT) @ a;
        # P = P_old - (P_old @ A1.T) @ np.linalg.inv(I + A1 @ P_old @ A1.T) @ A1 @ P_old;
        correction = P @ A1.T @ (B1 - A1 @ X_old);
        
    X = X_old + correction; # updated minimizer

    return (X, P);
