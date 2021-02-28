"""
print_report.py

Show X*
"""

def print_report(report):
        
    if (report['iter_no'] == report['N_iter_max']):
        print('Finished with %d iterations' % (report['N_iter_max']));
    else:
        print('X0: %0.3f' % (report['X0']));
        print('X: %0.3f' % (report['X']));
 