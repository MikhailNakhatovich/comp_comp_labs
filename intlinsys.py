import matplotlib.pyplot as plt
import numpy as np

from tolsolvty.tolsolvty import tolsolvty


def solve(a, b_inf, b_sup, *args, verbose=False):
    """
    Function for solving interval linear system `ax = [b_inf, b_sup]`
    :param a: (M, N) array_like
        "coefficient" matrix
    :param b_inf: (M,) array_like
        ordinate or "dependent variable" values
    :param b_sup: (M,) array_like
        ordinate or "dependent variable" values
    :param args:
        see description for optional arguments for `tolsolvty` function
    :param verbose:
        if `True` it needs to draw plots
    :return: tolmax, argmax
        see description for returning values of `tolsolvty` function
    """

    if verbose:
        print('Condition number of the matrix: %f' % np.linalg.cond(a))

    tolmax, argmax, envs, ccode = tolsolvty(a, a, b_inf, b_sup, *args)

    if verbose:
        print('ccode = %d\nT = %f' % (ccode, tolmax))

        plt.figure(figsize=(10, 10))
        plt.plot(argmax, color='b')
        plt.xlabel('index')
        plt.ylabel('value')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.plot(np.dot(a, argmax), color='b')
        plt.plot(b_inf, color='g')
        plt.plot(b_sup, color='r')
        plt.xlabel('index')
        plt.ylabel('value')
        plt.legend(['A * tau', 'inf b', 'sup b'])
        plt.grid(True)
        plt.show()

    return tolmax, argmax
