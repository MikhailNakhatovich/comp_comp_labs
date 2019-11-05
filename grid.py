import matplotlib.pyplot as plt
import numpy as np


def draw_grid(lines, center, radial_lines):
    """
    Function for drawing grid
    :param lines: ndarray
        grid lines
    :param center: array_like
        two digits - point of the center of the grid
    :param radial_lines: array_like
        radial lines including four support radial lines
        each line represents by ndarray
    """

    plt.figure()
    for line in lines:
        line = np.concatenate((line, [line[0]]))
        plt.plot(line[:, 0], line[:, 1], color='blue')
    for line in radial_lines:
        plt.plot(line[:, 0], line[:, 1], color='red')
    plt.scatter(center[0], center[1], c='green')
    plt.grid(True)
    plt.xlabel('r')
    plt.ylabel('z')
    plt.gca().set_aspect('equal')
    plt.show()


def draw_radius(radius):
    x = np.arange(len(radius))
    plt.figure()
    plt.plot(x, radius)
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('radius')
    plt.show()


def get_radius(line):
    """
    Function for getting curvature radius
    :param line: array_like
        array of points that represent by ndarrays
        closed curve, but line[0] is not equal to line[-1]
    :return radius: ndarray
        array of curvature radii for each point of `line`
    """

    n = len(line)
    radius = np.zeros(n)
    for i in range(n):
        p1 = line[i - 2]
        p2 = line[i - 1]
        p3 = line[i]
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        m2 = (p3[1] - p2[1]) / (p3[0] - p2[0])
        x = (m1 * m2 * (p1[1] - p3[1]) + m2 * (p1[0] + p2[0]) - m1 * (p2[0] + p3[0])) / (2 * (m2 - m1))
        y = ((p1[0] + p2[0]) / 2 - x) / m1 + (p1[1] + p2[1]) / 2
        radius[i] = np.linalg.norm(p1 - np.array([x, y]))
    return radius


def get4indices(line, radius):
    """
    Function for getting four indices of points for four support radial lines
    :param line: ndarray
        array of points that represent by ndarrays
        closed curve, but line[0] is not equal to line[-1]
    :param radius: array_like
        array of curvature radii for each point of `line`
    :return indices: array_like
        four indices in the `line` of four points for four support radial lines
    """

    n = len(radius)
    ind1, ind2 = np.nonzero(line[:, 1] == 0)[0]
    min1 = np.argmin(radius[ind1 + 1:ind2])
    min2 = np.argmin(np.concatenate((radius[ind2 + 1:n], radius[0:ind1])))
    return sorted([min1 + 1, (min2 + ind2 + 1) % n, ind1, ind2])


def get_radial_lines(border, center, indices, radials):
    """
    Function for getting all radial lines
    :param border: array_like
        outer border line that represents by ndarrays
        closed curve, but border[0] is not equal to border[-1]
    :param center:
        two digits - point of the center of the grid
    :param indices: array_like
        four indices in the `line` of four points for four support radial lines
    :param radials: int
        count of radial lines between four support radial lines
        must be non-negative
    :return radial_lines: ndarray
        generated radial lines including four support radial lines
    """

    radial_indices = set()
    indices_len = len(indices)
    n = len(border)
    for i in range(indices_len):
        ind1 = indices[i]
        ind2 = indices[(i + 1) % indices_len]
        if ind2 < ind1:
            ind1 = ind2 - (ind2 + n - ind1)
        for k in np.linspace(ind1, ind2, radials + 2):
            radial_indices.add(int(k))
    return np.array([np.array([center, border[_]]) for _ in radial_indices])


def get_lines(border, radial_lines, count):
    """
    Function for getting all grid lines
    :param border: array_like
        outer border line that represents by ndarrays
        closed curve, but border[0] is not equal to border[-1]
    :param radial_lines: array_like
        radial lines including four support radial lines
    :param count: int
        count of lines to generate
        must be non-negative
    :return lines: ndarray
        generated grid lines including `border`
    """

    if count > 0:
        lines = [[] for i in range(count)]
        alpha = 1 / (count + 1)
        for line in radial_lines:
            c = line[0]
            p = line[1]
            for k in range(1, count + 1):
                lines[k - 1].append(c * alpha * k + p * (1 - alpha * k))
        for i in range(len(lines)):
            lines[i] = np.array(lines[i])
    else:
        lines = []
    lines.append(border)
    return np.array(lines)


def generate_grid(border, center, count=0, radials=0):
    """
    Function for generation of grid
    :param border: ndarray
        outer border line that represents by ndarrays
        closed curve, but border[0] is not equal to border[-1]
    :param center: array_like
        two digits - point of the center of the grid
    :param count: int
        count of lines to generate
        must be non-negative
    :param radials: int
        count of radial lines between four support radial lines
        must be non-negative
    :return lines: ndarray
        generated grid lines including `border`
    :return radial_lines: ndarray
        generated radial lines including four support radial lines
    """

    radials = int(radials)
    count = int(count)
    radius = get_radius(border)
    # draw_radius(radius)
    indices = get4indices(border, radius)
    radial_lines = get_radial_lines(border, center, indices, radials)
    lines = get_lines(border, radial_lines, count)
    return lines, radial_lines
