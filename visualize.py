import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to plot point sets
def plot_point_sets(points_set_A, points_set_B, points_set_C=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points set A
    ax.scatter(points_set_A[:, 0], points_set_A[:, 1], points_set_A[:, 2], c='b', marker='o', label='Set A')

    # Plot points set B
    ax.scatter(points_set_B[:, 0], points_set_B[:, 1], points_set_B[:, 2], c='r', marker='^', label='Set B')

    # Plot points set C if provided
    if points_set_C is not None:
        ax.scatter(points_set_C[:, 0], points_set_C[:, 1], points_set_C[:, 2], c='g', marker='s', label='Set C')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()
