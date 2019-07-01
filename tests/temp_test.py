import random
from timeit2 import cyc_timeit

K = 100
ELEMENTS = random.sample(range(10), 10)

def a():
    index = random.choice(range(len(ELEMENTS)))
    return ELEMENTS[index]

def b():
    return random.choices(ELEMENTS)[0]

print(a())
print(b())
cyc_timeit([a,b], number=int(2e6))