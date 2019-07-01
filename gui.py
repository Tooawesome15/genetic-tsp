from math import sqrt, ceil
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def cvt_route4plot(route):
        x = []
        y = []
        for point in route:
            x.append(point.x)
            y.append(point.y)
        return x, y

def plot_points(points, fmt='o', xmin=0, xmax=100, ymin=0, ymax=100):
    x, y = cvt_route4plot(points)
    plt.plot(x, y, fmt)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title('Points')
    plt.show()

def plot_routes(routes, fmt='-', h_ratio=2, w_ratio=3, xmin=0, xmax=100, ymin=0, ymax=100):
    # Subplot Calculation
    x = sqrt(len(routes) / (h_ratio + w_ratio))
    n_rows = ceil(h_ratio * x)
    n_cols = ceil(len(routes) / n_rows)    
    # Subplot
    fig = plt.figure()
    for index, route in enumerate(routes, 1):
        ax = fig.add_subplot(n_rows, n_cols, index)
        ax.plot(*cvt_route4plot(route), fmt)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.suptitle('Routes')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


class GUI:
    def __init__(self, trend_size=50):
        # Plot Stuff
        fig = plt.figure()
        gs = GridSpec(2, 3, figure=fig)
        self.current_ax  = plt.subplot(gs[0:2,0:2])
        self.best_ax     = plt.subplot(gs[0,2])
        self.trend_ax    = plt.subplot(gs[1,2])

        self.current_ax.set_title('Current Simulation')
        self.best_ax.set_title('Best Route')
        self.trend_ax.set_title('Fitness Trend')

        self.current_text = self.current_ax.text(0.985, 0.02, 'Fitness: 0', horizontalalignment = 'right', verticalalignment = 'bottom', transform = self.current_ax.transAxes, bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.3))
        self.best_text = self.best_ax.text(0.97, 0.035, 'Fitness: 0', horizontalalignment = 'right', verticalalignment = 'bottom', transform = self.best_ax.transAxes, bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.3))
        self.generation_text = self.trend_ax.text(0.03, 0.035,'Generation: 0', horizontalalignment = 'left', verticalalignment = 'bottom', transform = self.trend_ax.transAxes, bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.3))
        self.average_text = self.trend_ax.text(0.97, 0.035, 'Average: 0', horizontalalignment = 'right', verticalalignment = 'bottom', transform = self.trend_ax.transAxes, bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.3))

        self.current_line = self.current_ax.plot([])[0]
        self.best_line = self.best_ax.plot([])[0]
        self.trend_line = self.trend_ax.plot([])[0]
        self.scale(0, 100, 0, 100)
        plt.pause(0.0000001)

        # TSP Stuff
        self.best_route = None
        self.best_distance = float('inf')
        self.trend = deque(maxlen=trend_size)

    def scale(self, xmin, xmax, ymin, ymax):
        self.current_ax.set_xlim(xmin, xmax)
        self.current_ax.set_ylim(ymin, ymax)
        self.best_ax.set_xlim(xmin, xmax)
        self.best_ax.set_ylim(ymin, ymax)

    def plot(self, current_route, current_fitness, best_route, best_fitness, trend, generation_no, average_fitness):
        self.current_line.set_xdata(current_route[0])
        self.current_line.set_ydata(current_route[1])
        self.current_text.set_text(f'Distance: {current_fitness:.2f}m')
        self.best_line.set_xdata(best_route[0])
        self.best_line.set_ydata(best_route[1])
        self.best_text.set_text(f'Distance: {best_fitness:.2f}m')
        self.trend_line.set_xdata(list(range(generation_no - len(trend), generation_no)))
        self.trend_line.set_ydata(trend)
        self.trend_ax.relim()
        self.trend_ax.autoscale_view(True,True,True)
        self.generation_text.set_text('Generation: ' + str(generation_no))
        self.average_text.set_text(f'Distance: {average_fitness:.2f}m')
        plt.pause(0.000000001)

    def display(self, population, fitnesses, distances, generation_no):
        # Find Best
        max_index = np.argmax(fitnesses)
        best_individual = population[max_index]
        best_distance = distances[max_index]
        # Update Best?
        if best_distance < self.best_distance:
            self.best_distance = best_distance
            self.best_route = best_individual
        else:
            best_distance = self.best_distance
            best_individual = self.best_route
        
        # Get Random
        selected_index = random.choice(range(len(population)))
        selected_individual = population[selected_index]
        selected_distance = distances[selected_index]

        # Plot Data
        best_route_data = cvt_route4plot(best_individual)
        selected_route_data = cvt_route4plot(selected_individual)
        
        # Average Distance
        average_distance = sum(distances) / len(population)
        average_fitness = sum(fitnesses) / len(fitnesses)
        self.trend.append(average_fitness)

        self.plot(selected_route_data, selected_distance, best_route_data, best_distance, list(self.trend), generation_no, average_distance)
