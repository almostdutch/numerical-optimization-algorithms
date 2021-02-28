"""
print_report.py

Print X* or norm(X*)
"""

import numpy as np

def print_report(func, report):
        
    if (report['iter_no'] == report['N_iter_max']):
        print('Finished with %d iterations' % (report['N_iter_max']));
    
    if 'X' in report:
        if (report['X'].size < 5):
            print('X: ')
            print(report['X']);
            print('f(X): %0.3f' % (func(report['X'])));
        else:
            print('Norm(X): %0.3f' % ( np.linalg.norm(report['X'])));
            print('f(X): %0.3f' % (func(report['X'])));
    else:
        if (report['X_best'].size < 5):
            print('X_best: ')
            print(report['X_best']);
            print('f(X): %0.3f' % (func(report['X_best'])));
        else:
            print('Norm(X_best): %0.3f' % ( np.linalg.norm(report['X_best'])));        
            print('f(X): %0.3f' % (func(report['X_best'])));
            