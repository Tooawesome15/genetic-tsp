import random

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def random(cls, xmin=0, xmax=100, ymin=0, ymax=100):
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)
        return cls(x, y)