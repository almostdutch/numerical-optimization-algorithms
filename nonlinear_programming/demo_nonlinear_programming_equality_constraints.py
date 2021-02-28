"""
demo_nonlinear_programming_equality_constraints.py 

(Non)linear objective function and (non)linear equality constraints

m = number of equality constraints;
n = number of parameters;

Objective function:
    Minimize f(X), where
    X = [x1, ..., xn].T is vector of variables;
    X* is extreme point (MIN or MAX);
        
Equality constrains:
    Subject to equality constraints: h(X);
    h(X) = [h1(X), ..., hn(X)].T is vector of constraints;
    m < n;
    
Lagrange algorithm

Lagrangian function:
    l(X, Lambda) = f(X) + lambda.T @ h(X);
    Lambda = [lambda1, ..., lambdan].T is vector of langrange multipliers;
    Lambda* is extreme point;

1st Condition:
    Gradient of lagrangian function: 
        Dl(X*, Lambda*) = Df(X*) + Dh(X*).T @ Lambda = 0.T;
    
2nd Condition:
    Normal space formed by all normal vectors of f(X) and h(X), i.e. Df(X) and Dh(X), at X*: 
        N(Dh(X*).T) is null space of Dh(X*).T; 
    Tangent space formed by all tangent (and shared!) vectors of f(X() and h(X) at X*:
        T(X*) is tangent space; 
    Hessian of l(X, lambda):
        Hl(X, lambda) = Hf(X) + lambda1 * Hh1(X) + ... + lambdan * Hhn(X);
    
    Tangent and Normal spaces are orthogonal:
        T(X*) = N(Dh(X*).T); 
    
    Solution lies in the tangent space:
        T(X*).T @ Hl(X*, lambda*) @ T(X*) > 0, X* is MIN;
"""

import numpy as np

"""
Problem # 1
Minimize f(X) = x1^2 + 2x1x2 + 3x2^2 + 4x1 + 5x2 + 6x3;
Constraints:
        h1(X) = x1 + 2x2 - 3 = 0;
        h2(X) = 4x1 + 5x3 - 6 = 0;

X = [x1, x2, x3].T;
Lambda = [lambda1, lambda2].T;

Df(X) = [2x1 + 2x2 + 4
         2x1 + 6x2 + 5
         6];

Dh(X) = [1, 2, 0
         4, 0, 5];

Solve system of eqs.:
1. Dl(X, Lambda) = [2x1 + 2x2 + 4 + lambda1 * [1, 2, 0].T + lambda2 * [4, 0, 5].T = 0.T;
                 2x1 + 6x2 + 5
                 6];
2. x1 + 2x2 - 3 = 0;
3. 4x1 + 5x3 - 6 = 0;

Hl(X*, Lambda*) = [2, 2, 6
                   2, 6, 0
                   0, 0, 0];

T(X*) = [-5/4, 5/8, 1].T;
"""

A = np.array([[2, 2, 0, 1, 4], [2, 6, 0, 2, 0], [0, 0, 0, 0, 5], 
              [1, 2, 0, 0, 0], [4, 0, 5, 0, 0]]);
B = np.array([[-4], [-5], [-6], [3], [6]]);

solution = np.linalg.solve(A, B);
X_opt = solution[0:3]; # X*
lambda_opt = solution[3:-1]; # lambda*
print('X:');
print(X_opt);



"""
Problem # 2
X = [x1, x2].T;
Q = np.array([[8, 2 * np.sqrt(2)], [2 * np.sqrt(2), 10], [5 * np.sqrt(2), 1]]);
C = np.array([[3], [6], [8]]);
D = 24;
A = np.random.rand(2, 3); 
B = np.random.rand(2, 1);
Minimize f(X) = 1 / 2 * X.T @ Q @ X - C.T @ X + D;
Constraints:
        h(X) = A @ X - B = 0;

Df(X) = X.T @ Q - C;
Dh(X) = A;

Solve system of eqs.:
1. Dl(X, Lambda) = X.T @ Q - C.T + lambda.T @ A = 0.T;
2. A @ X - B = 0;

Hl(X*, lambda*) = Q;
"""

Q = np.array([[8, 2 * np.sqrt(2), 11], [2 * np.sqrt(2), 10, 1], [0.5 * np.sqrt(2), 5, 2]]);
C = np.array([[3], [6], [8]]);
D = 24;
A = np.random.rand(2, 3);
B = np.random.rand(2, 1);

Qinv = np.linalg.inv(Q);
Qinv_times_At = Qinv @ A.T;

lambda_opt = np.linalg.inv(Qinv_times_At.T @ A.T) @ (B - Qinv_times_At.T @ C);
X_opt = Qinv @ C + Qinv_times_At @ lambda_opt;
print('X:');
print(X_opt);

