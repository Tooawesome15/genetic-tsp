from point import Point
from tsp import TSP
from gui import GUI
import genetic_algorithms as GA

SCREEN = GUI()
POINTS = [Point.random() for _ in range(100)]
# Initialise
solver = TSP(POINTS, 500, 200)

solver.distance2fitness = lambda distance: 1 / (distance ** 5)
solver.fitness2distance = lambda fitness: 1 / (fitness ** (1 / 5))

solver.initialise_population()
solver.set_parent_selection(GA.stochastic_universal_sampling)
solver.set_crossover(GA.edge_recombination_operator)
solver.set_mutation(GA.cycle_inversion_mutation, 0.2, k=10)
# solver.set_mutations(
#     [GA.cycle_inversion_mutation, GA.swap_mutation],
#     [0.2, 0.3],
#     [10, 2]
# )
solver.set_survivor_selection(GA.stochastic_universal_sampling_index)

while True:
    solver.evolve(10)
    SCREEN.display(solver.population, solver.fitnesses, solver.distances, solver.generation)
