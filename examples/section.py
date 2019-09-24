from math import cos, sin
import numpy as np

from section import find_section_by_plane, draw_plot, draw_points


def gen_torus_points(f, r):
    n = 360
    pi_2 = 2 * np.pi
    return np.array([(f[0] + r * cos(t * pi_2 / n), f[1] + r * sin(t * pi_2 / n)) for t in range(n)])


def run_example_section1():
    planes = [(1, 0, 0), (1, 0, 15), (1, 0, 30), (1, 0, 31), (1, 0, 35), (1, 0, 36), (1, 0, 40), (1, 0, 45)]
    # planes = [(1, 1, (2 ** 0.5) * p[2]) for p in planes]
    points = [gen_torus_points((40, 0), 5), gen_torus_points((40, 0), 10)]
    a_sections = []
    for plane in planes:
        sections = [find_section_by_plane(pts, plane, True) for pts in points]
        draw_plot(sections, azim=0)
        a_sections.extend(sections)
    draw_plot(a_sections, azim=80)


def run_example_section2():
    planes = [(1, 0, 0), (1, 0, 15), (1, 0, 30), (1, 0, 31), (1, 0, 35), (1, 0, 36), (1, 0, 40), (1, 0, 45)]
    # planes = [(1, 1, (2 ** 0.5) * p[2]) for p in planes]
    points = [gen_torus_points((40, 0), 5), gen_torus_points((40, 0), 10)]
    a_sections = []
    for plane in planes:
        sections = [find_section_by_plane(pts, plane) for pts in points]
        draw_points(sections, azim=0)
        a_sections.extend(sections)
    draw_points(a_sections, azim=85)


def run_example_section3():
    points = [gen_torus_points((40, 0), 5), gen_torus_points((40, 0), 10)]
    a_sections = []
    for i in range(51):
        plane = (1, 0, i)
        sections = [find_section_by_plane(pts, plane, True) for pts in points]
        a_sections.extend(sections)
    draw_plot(a_sections, azim=70)
