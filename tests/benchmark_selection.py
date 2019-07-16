from itertools import tee
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

from math import sqrt
def calc_fitness(route):
    total = 0
    for (pointA, pointB) in pairwise(route):
        dx = pointA.x - pointB.x
        dy = pointA.y - pointB.y
        distance = sqrt(dx ** 2 + dy ** 2)
        total += distance
    return 1 / total

from point import Point
import genetic_algorithms as GA
import random
from functools import wraps

N_POINTS        = 50
POPULATION_SIZE = 100
K_SELECT        = 2
TOURNAMENT_SIZE = 10

POINTS      = [Point.random() for _ in range(N_POINTS)]
POPULATION  = [random.sample(POINTS, N_POINTS) for _ in range(POPULATION_SIZE)]
FITNESS     = [calc_fitness(route) for route in POPULATION]


def decorator(func, *args, **kwargs):
    @wraps(func)
    def wrapper():
        return func(*args, **kwargs)
    return wrapper

selection_funcs = [
    decorator(GA.fitness_proportionate_selection, POPULATION, FITNESS, K_SELECT),
    decorator(GA.tournament_selection, POPULATION, FITNESS, K_SELECT, TOURNAMENT_SIZE),
    decorator(GA.stochastic_universal_sampling, POPULATION, FITNESS, K_SELECT)
]

selection_index_funcs = [
    decorator(GA.fitness_proportionate_selection_index, FITNESS, K_SELECT),
    decorator(GA.tournament_selection_index, FITNESS, K_SELECT, TOURNAMENT_SIZE),
    decorator(GA.stochastic_universal_sampling_index, FITNESS, K_SELECT)
]

print('N_POINTS', N_POINTS)
print('POPULATION_SIZE', POPULATION_SIZE)
print('K_SELECT', K_SELECT)
print('TOURNAMENT_SIZE', TOURNAMENT_SIZE)
from timeit2 import cyc_timeit
cyc_timeit(selection_index_funcs)