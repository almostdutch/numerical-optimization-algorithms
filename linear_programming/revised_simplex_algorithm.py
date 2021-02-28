"""
revised_simplex_algorithm.py

Returns the minimizer of the function
"""

import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)});

def pivoting(tabl, p, q):
    # pivoting about (p,q)-th element
    tabl_new = tabl - tabl[p] * (tabl[:, [q]] / tabl[p, [q]]);
    tabl_new[p] = tabl[p] / tabl[p, [q]];
    
    return tabl_new;

def revised_simplex_algorithm(A, b, C, indx_cols, options):

    b_print = options['b_print'];
    b_blands_rule = options['b_blands_rule'];
    epsilon = -10e-6; # workaround for -0 issues in while loop
    
    m, n = A.shape;
    Binv = np.linalg.inv(A[:, indx_cols]); # inverse matrix of basic columns of A
    A = Binv @ A;
    b = Binv @ b;
    cB = C[indx_cols]; # vector of cost coefficients for basic cols of A
    r = C - A.T @ cB; # vector of relative cost coefficients
    cost = cB.T @ b; # cost function
    y0 = b;
    
    if (b_print):
        # Forming initial revised canonical tableau
        tabl = np.concatenate((np.array(indx_cols).reshape((m, 1)), Binv, y0), axis = 1); 
        print(' ');
        print('Initial revised canonical Tableau [indx_cols inv(B) y0]');
        print(tabl);
            
    while (np.sum(r >= epsilon * np.ones((n, 1))) != n):
        
        if (b_blands_rule):
            q = 0;
            while(r[q] >= 0):
                q += 1;
        else:
            q = np.argmin(r); # col #q comes into basis
    
        yq = Binv @ A[:, [q]];
        
        if (np.sum((yq > 0)) == 0):
            print('Problem unbounded, exit..');
            break;
        
        a = yq / y0;
        p = np.argmax(a); # col #p leaves basis
       
        # pivoting about (p,q)-th element
        tabl_aug = np.concatenate((Binv, y0, yq), axis = 1);
        tabl_aug = pivoting(tabl_aug, p, m + 1);
        Binv = tabl_aug[:, 0:m];
        y0 = tabl_aug[:, [m]];
        indx_cols[p] = q;
        cB = C[indx_cols];
        r = C - ((cB.T @ Binv) @ A).T; # vector of relative cost coefficients      
        cost = cB.T @ b; # cost function
        
        if (b_print):
            # Forming updated revised canonical tableau
            tabl = np.concatenate((np.array(indx_cols).reshape((m, 1)), Binv, y0), axis = 1);            
            print(' ');
            print('Pivoting about (%d, %d)-th element' % (p, q));
            print(' ');
            print('Updated revised canonical Tableau [indx_cols inv(B) y0]');
            print(tabl);
    
    X = np.zeros((n, 1));
    X[indx_cols] = y0;
    cost = C.T @ X; # cost function
    
    if (b_print):
        print(' ');
        print('f(X) = %0.3f' % (cost));
        print('Solution:');
        print(X);

    return (X, cost);