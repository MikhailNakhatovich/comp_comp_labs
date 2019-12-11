import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cond

from tolsolvty.tolsolvty import tolsolvty


def heur_min_cond(a, r=0.1, n=10):
    """
    Function for estimating of the condition number of the interval matrix
    :param a: (M, N) array_like
        matrix
    :param r: float
        radius of the elements of the matrix `a`
        must be in an interval [0, 1]
    :param n: int
        count of iterations
        must be positive
    :return min_cond: {float, inf}
        estimating of the condition number of the interval matrix `[(1-r)a, (1+r)a]`
    """
    
    a = np.array(a)
    a_inf = a * (1 - r)
    a_sup = a * (1 + r)

    matr1 = np.ones_like(a)
    matr2 = np.ones_like(a)

    min_cond = np.inf

    for i in range(n):
        epm = np.random.randint(0, high=2, size=a.shape[:2])
        ind0 = np.where(epm == 0)
        ind1 = np.where(epm == 1)
        matr1[ind0] = a_inf[ind0]
        matr2[ind0] = a_sup[ind0]
        matr1[ind1] = a_sup[ind1]
        matr2[ind1] = a_inf[ind1]

        min_cond = min(min_cond, cond(matr1, 2), cond(matr2, 2))

    return min_cond


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
        print('Condition number of the matrix: %f' % cond(a))

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
