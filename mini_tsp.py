from math import sqrt
import random

from point import Point
import genetic_algorithms as GA
import gui

from itertools import tee
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

# Pythagoras theorem
# c^2 = a^2 + b^2
def calc_distance(pointA, pointB):
    dx = pointA.x - pointB.x
    dy = pointA.y - pointB.y
    distance = sqrt(dx ** 2 + dy ** 2)
    return distance

def calc_route_distance(route):
    total = 0
    for (pointA, pointB) in pairwise(route):
        total += calc_distance(pointA, pointB)
    return total

def calc_fitness(distance):
    return 1 / (distance ** 2)

POPULATION_SIZE = 100
PARENTS_SIZE = 50
N_POINTS = 30
POINTS = [Point.random() for _ in range(N_POINTS)]
screen = gui.GUI()

### Initialise Population
population = []
distances = []
fitnesses = []
for _ in range(POPULATION_SIZE):
    # Append Individual
    individual = POINTS[:]
    random.shuffle(individual)
    population.append(individual)
    # Append Distance
    distance = calc_route_distance(individual)
    distances.append(distance)
    # Append Fitness
    fitness = calc_fitness(distance)
    fitnesses.append(fitness)

generation_no = 0
while True:
    generation_no += 1
    ### Parent Crossover
    children = []
    # Parent Selection
    for _ in range(PARENTS_SIZE):
        parentA, parentB = GA.fitness_proportionate_selection(
                population, fitnesses, 2)
        childs = GA.edge_recombination_operator(parentA, parentB)
        children.extend(childs)
    
    ## Child Mutation
    for child in children:
        if random.random() < 0.1:
            GA.swap_mutation(child, 2)
    
    ### Evaluate Fitness
    children_distances = []
    children_fitnesses = []
    for child in children:
        distance = calc_route_distance(child)
        fitness = calc_fitness(distance)
        children_distances.append(distance)
        children_fitnesses.append(fitness)
    
    # Add to new Population
    population.extend(children)
    distances.extend(children_distances)
    fitnesses.extend(children_fitnesses)

    ### Survivor Selection
    indicies = GA.fitness_proportionate_selection_index(fitnesses, POPULATION_SIZE)
    population = [population[index] for index in indicies]
    distances = [distances[index] for index in indicies]
    fitnesses = [fitnesses[index] for index in indicies]
    
    # Display
    screen.display(population, fitnesses, distances, generation_no)