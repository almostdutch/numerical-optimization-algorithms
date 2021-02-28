"""
plot_progress_x.py

Plot contour plot f(X) and overlay path X, as X progresses to X*
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_progress_x(X, Y, Z, algorithm_name, report):
    
    N_max_paths = 500; # number of iterations to plot
    N_iter = report['iter_no'];
    progress_x = report['progress_x'];   
    
    # Plot the contour
    fig = plt.figure();   
    ax = fig.add_subplot(1, 1, 1)
    cset = ax.contour(X, Y, Z, 50, cmap=cm.coolwarm)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.draw()
    plt.title('Path to X* ' + algorithm_name + '(' + str(N_iter) + ' iterations' + ')')
    
    iter_no = 0;
    while (iter_no < N_iter and iter_no < N_max_paths):
        x1 = progress_x[0, iter_no];
        x2 = progress_x[0, iter_no + 1];
        y1 = progress_x[1, iter_no];
        y2 = progress_x[1, iter_no + 1];
        plt.plot([x1, x2], [y1, y2]);
        plt.draw();
        iter_no += 1;
        #print(progress_x[:, [iter_no]])
        