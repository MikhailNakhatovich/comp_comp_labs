from math import cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from frechet import distance_frechet
from section import find_section_by_plane, draw_plot, draw_points


def gen_torus_points(f, r):
    n = 360
    pi_2 = 2 * np.pi
    return np.array([(f[0] + r * cos(t * pi_2 / n), f[1] + r * sin(t * pi_2 / n)) for t in range(n)])


def run_example_section1():
    planes = [(1, 0, 0), (1, 0, 15), (1, 0, 30), (1, 0, 31), (1, 0, 35), (1, 0, 36), (1, 0, 40), (1, 0, 45)]
    points = [gen_torus_points((40, 0), 5), gen_torus_points((40, 0), 10)]
    for plane in planes:
        sections = [find_section_by_plane(pts, plane) for pts in points]
        draw_points(sections, azim=0)


def draw_plot_points(plot, points, azim=0.0):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plot = np.concatenate((plot, [plot[0]]))
    ax.plot(plot[:, 0], plot[:, 1], zs=plot[:, 2])
    ax.view_init(azim=azim)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='#ff7f0e', s=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(['Section', 'Lemniscate of Bernoulli'])
    plt.show()


def run_example_section2():
    plane = (1, 0, 5)
    points = gen_torus_points((10, 0), 5)
    section = find_section_by_plane(points, plane)
    square_1 = sorted(section[np.all(np.column_stack((section[:, 1] >= 0, section[:, 2] >= 0)), axis=1)], key=lambda e: e[1])
    square_2 = sorted(section[np.all(np.column_stack((section[:, 1] < 0, section[:, 2] >= 0)), axis=1)], key=lambda e: e[1], reverse=True)
    square_3 = sorted(section[np.all(np.column_stack((section[:, 1] < 0, section[:, 2] < 0)), axis=1)], key=lambda e: e[1])
    square_4 = sorted(section[np.all(np.column_stack((section[:, 1] >= 0, section[:, 2] < 0)), axis=1)], key=lambda e: e[1], reverse=True)
    section = np.concatenate((square_2, square_3, square_1, square_4))
    draw_plot([section], azim=45)

    d = np.pi / 180
    x = []
    y = []
    alpha = np.arange(-np.pi / 2, np.pi / 2 + d, d)
    ps = np.tan(alpha)
    c = np.sqrt(200)
    for p in ps:
        x.append(c * (p + p ** 3) / (1 + p ** 4))
        y.append(c * (p - p ** 3) / (1 + p ** 4))
    lemn = np.column_stack((np.ones(len(x)) * 5, x, y))

    draw_plot_points(section, lemn)
    draw_plot_points(section, lemn, 45)
    d, i, j = distance_frechet(section, lemn)
    print("distance = " + str(d))


def run_example_section3():
    planes = [(1, 0, 0), (1, 0, 5), (1, 0, 7), (1, 0, 10), (1, 0, 15)]
    points = gen_torus_points((10, 0), 5)
    for plane in planes:
        sections = find_section_by_plane(points, plane)
        draw_points([sections], azim=0)
