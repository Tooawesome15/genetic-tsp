from point import Point
from tsp import TSP
from gui import GUI
import genetic_algorithms as GA

SCREEN = GUI()
POINTS = [Point.random() for _ in range(50)]
# Initialise
solver = TSP(POINTS, 100, 50)

solver.distance2fitness = lambda distance: 1 / (distance ** 2)
solver.fitness2distance = lambda fitness: 1 / (fitness ** (1 / 2))

solver.initialise_population()
solver.set_parent_selection(GA.fitness_proportionate_selection)
solver.set_crossover(GA.edge_recombination_operator)
solver.set_mutation(GA.swap_mutation, k=2)
solver.set_survivor_selection(GA.fitness_proportionate_selection_index)

while True:
    solver.evolve(1, 0.1)
    SCREEN.display(solver.population, solver.fitnesses, solver.distances, solver.generation)
