from math import acos, cos, sin, sqrt, atan2, pi
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from shapely.geometry import Polygon, LineString

from grid import generate_domain_grid, draw_domains
from section import sorted_section_by_nonzero_plane


def get_domains(border, center, count=0, radials=0, verbose=False):
    """
    Function for generation of grid and reordering domains of this grid
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
    :param verbose: boolean
        if `True` it needs to draw a plot
    :return domains: ndarray
        generated grid domains
    """

    domains = generate_domain_grid(border, center, count=count, radials=radials)
    n = (radials + 1) * 4
    for i in range(count + 1):
        up_domains = np.flip(domains[i * n:i * n + n // 2], axis=0)
        bottom_domains = np.flip(domains[i * n + n // 2:(i + 1) * n], axis=0)
        domains[i * n:(i + 1) * n] = np.concatenate((up_domains, bottom_domains))
    if verbose:
        draw_domains(domains, center)
    return domains


def create_line(p, q):
    """
    Function for creating an equation of the line by two points
    :param p: array_like
        first point
    :param q: array_like
        second point
    :return: line: array_like
        three coefficients of the line `ax + by = c`, where c - distance from line to (0, 0)
    """

    a = p[1] - q[1]
    b = q[0] - p[0]
    c = p[0] * q[1] - q[0] * p[1]
    mu = sqrt(a ** 2 + b ** 2)
    a = a / mu
    b = b / mu
    c = c / mu

    if c > 0:
        a = -a
        b = -b
    c = abs(c)

    return a, b, c


def line_value(x, line):
    a, b, c = line
    return (c - a * x) / b


def draw_line(line, l, r, color='r'):
    x = np.linspace(l, r, 100)
    y = line_value(x, line)
    plt.plot(x, y, color=color, linewidth=1)


def draw_detector(points_xy, points_z, aperture_xy):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    plt.grid(True)
    for i in range(16):
        plt.scatter(np.full(16, points_xy[i][0]), points_z, c='b')
    ax.set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    plt.grid(True)
    plt.scatter(points_xy[:, 0], points_xy[:, 1], c='b')
    plt.scatter(aperture_xy[0], aperture_xy[1], c='g')
    ax.set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def draw_top(border, points_xy, aperture_xy):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    plt.grid(True)
    ax.add_artist(plt.Circle((0, 0), border[:, 0].min(), color='b', fill=False))
    ax.add_artist(plt.Circle((0, 0), border[:, 0].max(), color='b', fill=False))
    for j in range(16):
        draw_line(create_line(points_xy[j], aperture_xy), points_xy[j][0], 0.8)
    plt.scatter(aperture_xy[0], aperture_xy[1], c='g')
    ax.set_aspect('equal')
    plt.xlim((-0.8, 0.8))
    plt.ylim((-0.8, 0.8))
    plt.show()


def get_detector():
    ang = acos((708 ** 2 + 720 ** 2 - 31 ** 2) / (2 * 708 * 720))
    spd_start = np.array([0, -0.708])
    spd_end = np.array([0.72 * sin(ang), 0.72 * -cos(ang)])
    spd_vect = (spd_end - spd_start) / norm(spd_end - spd_start)
    min_step = (2.3375 - 0.88) * 1e-03
    max_step = (3.81 - 2.3375 + 0.88) * 1e-03
    pp = spd_start + spd_vect * ((min_step + max_step) * 8 + 0.52 * 1e-03) / 2
    aperture_xy_offset = 0.0395
    aperture_xy = np.array([pp[0] - spd_vect[1] * aperture_xy_offset, pp[1] + spd_vect[0] * aperture_xy_offset])
    spd_z_start = (27.52 - 0.49) / 2 * 1e-03
    spd_z_step = -1.72 * 1e-03
    spd_xy = spd_start + spd_vect * (max_step / 2 + 0.26 * 1e-03)

    step = [[min_step, -min_step], [max_step, -max_step]]
    points_z = np.array([spd_z_start + i * spd_z_step for i in range(16)])
    points_xy = np.full((16, 2), spd_start + step[0])
    for j in range(1, 16):
        points_xy[j] = points_xy[j - 1] + spd_vect * (min_step if j % 2 == 1 else max_step)

    return points_xy, points_z, aperture_xy, spd_xy


def get_intersection_length(points, line_points, verbose=False):
    """
    Function for getting length of intersection of polygon and line
    :param points: array_like
        points that represent the polygon
    :param line_points: array_like
        points that represent the line
    :param verbose: boolean
        if `True` it needs to draw a plot
    :return:
        length of the intersection
    """

    def get_length(line_string, c='k'):
        if verbose:
            plt.scatter(line_string.xy[0], line_string.xy[1], c=c, s=5)
        return line_string.length

    length = 0
    polygon = Polygon(points)
    line = LineString(line_points)
    intersection = polygon.intersection(line)
    if not intersection.is_empty:
        if intersection.geom_type == 'MultiLineString':
            for line in intersection.geoms:
                length += get_length(line, c='y')
        else:
            length += get_length(intersection)
    return length


def generate_chord_matrix(border, center, count=0, radials=0, verbose=False):
    """
    Function for generation matrix of chords' lengths
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
    :param verbose: boolean
        if `True` it needs to draw plots
    :return domains: ndarray
        generated matrix of chords' lengths
    """

    domains = get_domains(border, center, count=count, radials=radials, verbose=verbose)

    points_xy, points_z, aperture_xy, spd_xy = get_detector()
    if verbose:
        draw_detector(points_xy, points_z, aperture_xy)
        draw_top(border, points_xy, aperture_xy)

    matrix = np.zeros((256, (count + 1) * (radials + 1) * 4))

    for j in range(16):
        if verbose:
            plt.figure(figsize=(10, 10))
            plt.grid(True)
            plt.xlabel('y')
            plt.ylabel('z')
            plt.gca().set_aspect('equal')

        line = create_line(points_xy[j], aperture_xy)
        spd_r = -sqrt(norm(spd_xy) ** 2 - line[2] ** 2)
        aperture_xz_offset = norm(spd_xy - aperture_xy)
        lines = [create_line([spd_r, p_z], [aperture_xz_offset + spd_r, 0]) for p_z in points_z]
        line_points = [[[spd_r, p_z], [0.6, line_value(0.6, lines[i])]] for i, p_z in enumerate(points_z)]
        for i, domain in enumerate(domains[:]):
            section = sorted_section_by_nonzero_plane(domain, (1, 0, line[2]))
            if section.shape[0] > 0:
                if section.shape[0] >= 2 * domain.shape[0]:
                    left = section[section[:, 1] <= 0, 1:]
                    right = section[section[:, 1] >= 0, 1:]
                    if verbose:
                        c_left = np.concatenate((left, left[:1]))
                        c_right = np.concatenate((right, right[:1]))
                        plt.plot(c_left[:, 0], c_left[:, 1], color='b', linewidth=0.7)
                        plt.plot(c_right[:, 0], c_right[:, 1], color='r', linewidth=0.7)
                    for k, l_p in enumerate(line_points):
                        matrix[j * 16 + k, i] += get_intersection_length(left, l_p, verbose=verbose)
                    if j < 12:
                        for k, l_p in enumerate(line_points):
                            matrix[j * 16 + k, i] += get_intersection_length(right, l_p, verbose=verbose)
                elif section.shape[0] > 2:
                    points = section[:, 1:]
                    if verbose:
                        c_points = np.concatenate((points, points[:1]))
                        plt.plot(c_points[:, 0], c_points[:, 1], color='g', linewidth=0.7)
                    for k, l_p in enumerate(line_points):
                        matrix[j * 16 + k, i] += get_intersection_length(points, l_p, verbose=verbose)

        if verbose:
            for i, line in enumerate(lines):
                draw_line(line, points_xy[j][1], 0.6, 'm')
            plt.scatter(aperture_xz_offset + spd_r, 0, c='y')
            plt.show()

    return matrix
