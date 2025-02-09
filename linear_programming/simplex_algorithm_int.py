"""
simplex_algorithm_int.py

Returns the int minimizer of the function
"""

import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)});
 
def pivoting(tabl, p, q):
    # pivoting about (p,q)-th element
    tabl_new = tabl - tabl[p] * (tabl[:, [q]] / tabl[p, [q]]);
    tabl_new[p] = tabl[p] / tabl[p, [q]];
    
    return tabl_new;

def simplex_algorithm_int(A, b, C, indx_cols, options):

    b_print = options['b_print'];
    b_blands_rule = options['b_blands_rule'];
    epsilon = -10e-6; # workaround for -0 issues in while loop
    
    if (b_print):
        # Forming initial tableau
        tabl_top = np.concatenate((A, b), axis = 1);
        tabl_bottom = np.concatenate((C.T, np.array([[0]])), axis = 1);
        tabl = np.concatenate((tabl_top, tabl_bottom), axis = 0); 
        print(' ');
        print('Initial Tableau [A b; C.T 0]');
        print(tabl);
        
    m, n = A.shape;
    Binv = np.linalg.inv(A[:, indx_cols]); # inverse matrix of basic columns of A
    A = Binv @ A;
    b = Binv @ b;
    cB = C[indx_cols]; # vector of cost coefficients for basic cols of A
    r = C - A.T @ cB; # vector of relative cost coefficients
    cost = -cB.T @ b; # cost function
    
    # Forming canonical tableau
    tabl_top = np.concatenate((A, b), axis = 1);
    tabl_bottom = np.concatenate((r.T, cost), axis = 1);
    tabl = np.concatenate((tabl_top, tabl_bottom), axis = 0); 
    
    if (b_print):
        print(' ');
        print('Canonical Tableau [A b; r.T cost]');
        print(tabl);
    
    while (np.sum(r >= epsilon * np.ones((n, 1))) != n):

        if (b_blands_rule):
            q = 0;
            while(r[q] >= 0):
                q += 1;
        else:
            q = np.argmin(r); # col #q comes into basis
    
        if (np.sum((tabl[0:m, [q]]) > 0) == 0):
            print('Problem unbounded, exit..');
            break;
        
        
        a = tabl[0:m, [q]] / tabl[0:m, [n]];
        p = np.argmax(a); # col #p leaves basis
    
        # pivoting about (p,q)-th element
        tabl = pivoting(tabl, p, q);
        
        if (b_print):
            print(' ');
            print('Pivoting about (%d, %d)-th element' % (p, q));
            print(' ');
            print('Updated canonical Tableau [A b; r.T cost]');
            print(tabl);
            
        indx_cols[p] = q;
        r = (tabl[[m], 0:n]).T;
    
    if (b_print):
        print(' ');
        print('Canonical Tableau [A b; r.T cost] with optimal basic feasible solution');
        print(tabl);

    # Adding int constrains (Gomory cuts)
    indx_cols_int = indx_cols;
    for ii in range (m):
        mask = np.ones(C.size + 1 + ii, dtype = bool);
        mask[indx_cols_int] = False; # list of indices of non-basic cols of A    
        tabl_int = np.zeros((1, tabl.shape[1]));
        tabl_int[:, mask] = tabl[ii, mask] - np.floor(tabl[ii, mask]);
        insert_col = np.eye(tabl.shape[0] + 1, 1, k = -(m + ii)) * (-1);
        all_rows = np.concatenate((tabl[0:m + ii], tabl_int, tabl[[-1]]), axis = 0);
        tabl_int = np.concatenate((all_rows[:, 0:-1], insert_col, all_rows[:, [-1]]), axis = 1);
        tabl = tabl_int;
        
        if (b_print):
            print(' ');
            print('Canonical Tableau [A b; r.T cost] with Gomory cuts');
            print(tabl);
            
        q = m + ii;
        p = q;
        tabl = pivoting(tabl, p, q);
        indx_cols_int = np.append(indx_cols_int, q); # adding a new basic col
            
        if (b_print):
            print(' ');
            print('Pivoting about (%d, %d)-th element' % (p, q));
            print(' ');
            print('Updated canonical Tableau [A b; r.T cost] with Gomory cuts');
            print(tabl);    
    
    X = np.zeros((n, 1));
    X[indx_cols] = tabl[0:m, [-1]];
    cost = C.T @ X; # cost function
    
    if (b_print):
        print(' ');
        print('f(X) = %0.3f' % (cost));
        print('Solution:');
        print(X);

    return (X, cost);