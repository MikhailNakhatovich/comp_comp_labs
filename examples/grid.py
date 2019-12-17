from examples.fileextractor import run_example_extract
from grid import generate_grid, draw_grid, generate_domain_grid, draw_domains


def run_example_grid1():
    border, center = run_example_extract('data/g035685.00150')
    lines, radial_lines = generate_grid(border, center, count=3, radials=6)
    draw_grid(lines, center, radial_lines)
    domains = generate_domain_grid(border, center, count=3, radials=6)
    draw_domains(domains, center)
