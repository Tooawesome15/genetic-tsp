import genetic_algorithms as GA
from point import Point

import random
from functools import wraps

N_POINTS = 50
K = 10

POINTS = [Point.random() for _ in range(N_POINTS)]
PARENT_A = random.sample(POINTS, N_POINTS)
PARENT_B = random.sample(POINTS, N_POINTS)

def decorator(func, *args):
    @wraps(func)
    def wrapper():
        return func(PARENT_A, PARENT_B, *args)
    return wrapper

crossover_funcs = [
    decorator(GA.partially_matched_crossover, K),
    #decorator(GA.cycle_crossover, K),
    decorator(GA.order_1_crossover, K),
    decorator(GA.edge_recombination_operator)
]

from timeit2 import cyc_timeit
print('N_POINTS', N_POINTS, '\t','K', K)
cyc_timeit(crossover_funcs)