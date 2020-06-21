import argparse

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def plot(u, v, p, figname='tmp.png'):
    nx = 41
    ny = 41

    if len(u.shape) == 1:
        u = u.reshape(ny, nx)
    if len(v.shape) == 1:
        v = v.reshape(ny, nx)
    if len(p.shape) == 1:
        p = p.reshape(ny, nx)

    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    _ = plt.figure(figsize=(11, 7), dpi=100)
    # plotting the pressure field as a contour
    plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    plt.colorbar()
    # plotting the pressure field outlines
    plt.contour(X, Y, p, cmap=cm.viridis)
    # plotting velocity field
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(figname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_path', type=str, default=None,
                        help='path to txt file with calculated results')
    parser.add_argument('--figname', type=str, default=None,
                        help='path of output figure')
    args = parser.parse_args()

    with open(args.txt_path, 'r') as f:
        results = f.readlines()

    result_dict = {}
    for line in results:
        line_list = line.split(' ')
        result_dict[line_list[0]] = np.array([float(val) for val in line_list[1:-1]])


    plot(result_dict['u'], result_dict['v'], result_dict['p'], args.figname)
