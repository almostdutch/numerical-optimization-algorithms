"""
demo_nonlinear_programming_nonequality_constraints.py 

(Non)linear objective function and (non)linear nonequality constraints

m = number of equality constraints;
n = number of parameters;

Objective function:
    Minimize f(X), where
    X = [x1, ..., xn].T is vector of variables;
    X* is extreme point (MIN or MAX);
    
(Non)linear equality constrains:
    Subject to equality constraints: h(X);
    h(X) = [h1(X), ..., hn(X)].T = 0.T is vector of constraints;
    m < n;
    
(Non)linear nonequality constrains:
    Subject to nonequality constraints: g(X);
    g(X) = [g1(X), ..., gn(X)].T <= 0.T is vector of constraints;    

Lagrange algorithm

Lagrangian function:
    l(X, Lambda) = f(X) + Lambda.T @ h(X) + Mu.T @ g(X);
    Lambda = [lambda1, ..., lambdan].T is vector of multipliers for equality constraints;
    Mu = [mu1, ..., mun].T is vector of multipliers for nonequality constraints;
    Lambda* and Mu* are extreme point;
    Mu >= 0;

1st Condition:
    Gradient of lagrangian function: 
        Dl(X*, Lambda*) = Df(X*) + lambda1 * Dh1(X*) + ... + lambdan * Dhn(X*) = 0.T;
        Mu.T @ g(X*) = 0.T;
    
2nd Condition:
    Normal space formed by all normal vectors of f(X) and h(X), i.e. Df(X) and Dh(X), at X*: 
        N(Dh(X*).T) is null space of Dh(X*).T; 
    Tangent space formed by all tangent (and shared!) vectors of f(X() and h(X) at X*: 
        T(X*) is tangent space; 
    Hessian of l(X, lambda):
        Hl(X, lambda) = Hf(X) + lambda1 * Hh1(X) + ... + lambdan * Hhn(X);
    
    Tangent and Normal spaces are orthogonal:
        T(X*) = N([Dh(X).T, Dgi(X).T].T for mui > 0);
    
    Solution lies in the tangent space:
        T(X*).T @ Hl(X*, lambda*) @ T(X*) > 0, X* is MIN;
"""

import numpy as np

"""
I tried to come up with some general way of solving these types of optimization problems, but boy is it hard!
Depending on the problem at hand, there are multiple options for mu (mu = 0 and mu > 0) and X*, so hard to formulate a generalized solution.
"""