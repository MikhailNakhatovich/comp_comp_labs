import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


def solve(a, b_inf, b_sup, verbose=False):
    """
    Function for solving interval linear system `ax = [b_inf, b_sup]` like linear programming problem
    :param a: (M, N) array_like
        "coefficient" matrix
    :param b_inf: (M,) array_like
        ordinate or "dependent variable" values
    :param b_sup: (M,) array_like
        ordinate or "dependent variable" values
    :param verbose:
        if `True` it needs to draw plots
    :return: x: {(N,) ndarray, None}
        the independent variable vector which optimizes the linear programming problem
    :return: success: bool
        returns `True` if the algorithm succeeded in finding an optimal solution
    """

    a = np.array(a)
    b_inf = np.array(b_inf)
    b_sup = np.array(b_sup)

    m, n = a.shape[:2]
    ki = b_inf.shape[0]
    ks = b_sup.shape[0]
    if ki == ks:
        k = ks
    else:
        print('The number of components in the vectors of the left and right ends is not the same')
        return None, False
    if k != m:
        print('The dimensions of the system matrix do not match the dimensions of the right side')
        return None, False
    if not np.all(b_inf <= b_sup):
        print('Invalid interval component was set on the right side vector')
        return None, False

    f = np.concatenate((np.zeros(n), np.ones(k)))
    mid = (b_inf + b_sup) / 2
    rad = (b_sup - b_inf) / 2
    diag_rad = np.diag(rad)
    c = np.vstack((np.hstack((a, -diag_rad)), np.hstack((-a, -diag_rad))))
    d = np.concatenate((mid, -mid))

    res = linprog(f, A_ub=c, b_ub=d, options={"disp": True})
    z = res['x']
    success = res['success']
    x, w = z[:n], z[n:]
    print("function(z) = %f" % res['fun'])

    if verbose:
        plt.figure(figsize=(10, 10))
        plt.plot(w, color='b')
        plt.xlabel('index')
        plt.ylabel('value')
        plt.grid(True)
        plt.title("$\omega$")
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.plot(x, color='b')
        plt.xlabel('index')
        plt.ylabel('value')
        plt.grid(True)
        plt.title("x")
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.plot(np.dot(a, x), color='b')
        plt.plot(b_inf, color='g')
        plt.plot(b_sup, color='r')
        plt.xlabel('index')
        plt.ylabel('value')
        plt.legend(['A * x', 'inf b', 'sup b'])
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.plot(np.dot(a, x), color='b')
        plt.plot(mid - w * rad, color='g')
        plt.plot(mid + w * rad, color='r')
        plt.xlabel('index')
        plt.ylabel('value')
        plt.legend(['A * x', 'mid b - $\omega$ * rad b', 'mid b + $\omega$ * rad b'])
        plt.grid(True)
        plt.show()

    return x, success
