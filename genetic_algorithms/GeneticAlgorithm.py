import numpy as np
import random
import matplotlib.pyplot as plt
from typing import TypeVar, List, Callable

T = TypeVar('T')


class GeneticAlgorithm:
    def __init__(self, fitness_function: Callable[[T], float], initial_population: List[T],
                 crossover_function: Callable[[T, T], T], mutation_function: Callable[[T], T],
                 selection_function: Callable[[List[float], int], List[T]], population_size: int = 100,
                 generations: int = 200, n_selection: int = 20, n_elite: int = 1):
        self.fitness_function = fitness_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.selection_function = selection_function

        self.initial_population = initial_population
        self.population = initial_population[:]

        self.population_size = population_size
        self.generations = generations
        self.n_selection = n_selection
        self.n_elite = n_elite

        self.best_solution = None
        self.best_fitness = float('-inf')

        self.population_history = []
        self.best_history = []

    def run(self):
        fitness_values = np.array([self.fitness_function(individual) for individual in self.population])
        for _ in range(self.generations):
            self.population_history.append(self.population)

            selected_indices = self.selection_function(fitness_values, self.n_selection)
            selected = [self.population[i] for i in selected_indices]

            children = []
            for _ in range((self.population_size - self.n_elite) // 2):
                parents = random.choices(selected, k=2)
                child1, child2 = self.crossover_function(*parents)
                children.extend((child1, child2))

            children = [self.mutation_function(child) for child in children]

            elite_indices = np.argpartition(fitness_values, -self.n_elite)[-self.n_elite:]
            elite = [self.population[i] for i in elite_indices]
            self.population = elite + children

            elite_fitness = fitness_values[elite_indices]
            if elite_fitness.max() > self.best_fitness:
                self.best_solution = elite[np.argmax(elite_fitness)]
                self.best_fitness = elite_fitness.max()
            self.best_history.append(self.best_fitness)

            new_fitness = np.array([self.fitness_function(individual) for individual in children])
            fitness_values = np.concatenate((elite_fitness, new_fitness))

    def plot(self, top_best: int = 10):
        x = []
        y = []
        for i, population in enumerate(self.population_history):
            plotted_individuals = min(len(population), top_best)
            x.extend([i] * plotted_individuals)
            population_fitnesses = [self.fitness_function(individual) for individual in population]
            population_fitnesses.sort(reverse=True)
            y.extend(population_fitnesses[:plotted_individuals])
        plt.scatter(x, y, marker='.')
        plt.plot(self.best_history, 'r')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()
