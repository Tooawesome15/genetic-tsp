from functools import wraps
import random

K = 10
VALUES = [random.random() for _ in range(K)]

def decorator(func):
    @wraps(func)
    def wrapper():
        return func(VALUES)
    return wrapper

from itertools import tee
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def cost(a, b):
    return a * b

@decorator
def a(values):
    total = 0
    for a, b in pairwise(values):
        total = cost(a, b)
    return total

@decorator
def b(values):
    total = 0
    for x in pairwise(values):
        total = cost(*x)
    return total

@decorator
def c(values):
    total = 0
    for i in range(1, len(values)):
        total = cost(values[i-1], values[i])
    return total

from timeit2 import cyc_timeit
cyc_timeit([a,b,c], number=1000000)