import numpy as np
import matplotlib.pyplot as plt

def plot_progress_y(algorithm_name, report):
    
    N_iter = report['iter_no'];
    progress_y = report['progress_y'];
    # Plot objective function
    fig = plt.figure();
    ax = fig.add_subplot(1, 1, 1)
    X = np.arange(0, N_iter + 1, 1);
    Y = progress_y;
    plt.plot(X, Y[0, 0:N_iter + 1]);
    ax.set_xlabel('Iteration #');
    ax.set_ylabel('Objective function');
    plt.title('Path to X* ' + algorithm_name + '(' + str(N_iter) + ' iterations' + ')');
    plt.show();
