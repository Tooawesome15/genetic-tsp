import random
import functools
import itertools
import collections
import multiprocessing as mp
from multiprocessing import Pool

class TSP:
    def __init__(self, points, population_size, parent_pair_size, pair_distances=None):
        # VARIABLES / CONSTANTS
        self.points           = points
        self.generation       = 0
        self.population_size  = population_size
        self.parent_pair_size = parent_pair_size
        
        # Distance Computation Optimisation
        if pair_distances:  self.pair_distances = pair_distances
        else:               self.pair_distances = self.generate_pair_distances(points)

        # GA Functions
        self.parent_selection         = lambda: None
        self.crossover                = lambda: None
        self.mutation                 = lambda: None
        self.survivor_selection_index = lambda: None

        # Important Variables
        self.population       = []
        self.fitnesses        = []
        self.distances        = []

    ### Optimisation ###
    @classmethod
    def generate_pair_distances(cls, points):
        pair_distances = {}
        for (pointA, pointB) in itertools.combinations(points, 2):
            pair_distance = cls.euclidean_distance(pointA, pointB)
            point_pair = frozenset((pointA, pointB))
            pair_distances[point_pair] = pair_distance
        return pair_distances
    
    ### Calculate Distance & Fitness ###
    @staticmethod
    def euclidean_distance(pointA, pointB):
        dx = pointA.x - pointB.x
        dy = pointA.y - pointB.y
        distance = (dx ** 2 + dy ** 2) ** 0.5
        return distance

    @staticmethod
    def euclidean_squared_distance(pointA, pointB):
        dx = pointA.x - pointB.x
        dy = pointA.y - pointB.y
        distance = dx ** 2 + dy ** 2
        return distance

    def compute_route_distance(self, route):
        distances = 0
        prev_point = None
        for point in route:
            point_pair = frozenset((prev_point, point))
            distances += self.pair_distances.get(point_pair, 0)
            prev_point = point
        return distances

    def evaluate_fitness(self, chromosome):
        distance = self.compute_route_distance(chromosome)
        fitness = self.distance2fitness(distance)
        return fitness

    @staticmethod
    def distance2fitness(distance):
        return 1 / distance

    @staticmethod
    def fitness2distance(fitness):
        return 1 / fitness

    ### Init ###
    def initialise_population(self):
        self.population = []
        self.fitnesses = []
        for _ in range(self.population_size):
            chromosome = self.points[:]
            random.shuffle(chromosome)
            fitness = self.evaluate_fitness(chromosome)
            self.population.append(chromosome)
            self.fitnesses.append(fitness)

    ### Set GA Functions ###
    def set_parent_selection(self, parent_selection_func, k=None, tournament_size=None):
        kwargs2 = {}
        if k:               kwargs2['k'] = k
        if tournament_size: kwargs2['tournament_size'] = tournament_size

        @functools.wraps(parent_selection_func)
        def parent_selection_func_wrapper(*args, **kwargs):
            return parent_selection_func(self.population, self.fitnesses, *args, **kwargs, **kwargs2)
        
        self.parent_selection = parent_selection_func_wrapper

    def set_crossover(self, crossover_func, k=None):
        kwargs2 = {}
        if k: kwargs2['k'] = k

        @functools.wraps(crossover_func)
        def crossover_func_wrapper(parentA, parentB, *args, **kwargs):
            return crossover_func(parentA, parentB, *args, **kwargs, **kwargs2)
        
        self.crossover = crossover_func_wrapper

    def set_mutation(self, mutation_func, p, k=None):
        kwargs2 = {}
        if k: kwargs2['k'] = k

        @functools.wraps(mutation_func)
        def mutation_func_wrapper(child, *args, **kwargs):
            if random.random() < p:
                mutation_func(child, *args, **kwargs, **kwargs2)
        
        self.mutation = mutation_func_wrapper

    def set_mutations(self, mutation_funcs, ps, ks=None):
        if ks is None: ks = [None for _ in range(len(mutation_funcs))]
        if len(mutation_funcs) != len(ps) != len(ks): raise Exception('Lengths must be equal.')
        kwargs2s = []
        for k in ks:
            kwargs2 = {}
            if k: kwargs2['k'] = k
            kwargs2s.append(kwargs2)

        def mutations_func_wrapper(child, *args, **kwargs):
            for mutation_func, p, kwargs2 in zip(mutation_funcs, ps, kwargs2s):
                if random.random() < p:
                    mutation_func(child, *args, **kwargs, **kwargs2)

        self.mutation = mutations_func_wrapper

    def set_survivor_selection(self, survivor_selection_index_func, k=None, tournament_size=None):
        kwargs2 = {}
        if k: kwargs2['k'] = k
        if tournament_size: kwargs2['tournament_size'] = tournament_size

        @functools.wraps(survivor_selection_index_func)
        def survivor_selection_index_func_wrapper(*args, **kwargs):
            return survivor_selection_index_func(self.fitnesses, *args, **kwargs, **kwargs2)
        
        self.survivor_selection_index = survivor_selection_index_func_wrapper

    ### Execution ###
    def evolve(self, k):
        for _ in range(k):
            for _ in range(self.parent_pair_size):                      # Repeat
                parentA, parentB = self.parent_selection(2)             # Select Pair of Parents
                for child in self.crossover(parentA, parentB):          # Crossover Parents
                    self.mutation(child)                                # Mutate Child
                    self.population.append(child)                       # Add to Population
                    self.fitnesses.append(self.evaluate_fitness(child)) # Evaluate Fitness
            # Select Survivors
            indices = self.survivor_selection_index(self.population_size)
            self.population = [self.population[index] for index in indices]
            self.fitnesses = [self.fitnesses[index] for index in indices]
        
        self.generation += k
        self.distances = [self.fitness2distance(fitness) for fitness in self.fitnesses]

    def evolveMP(self, k, p):
        pass