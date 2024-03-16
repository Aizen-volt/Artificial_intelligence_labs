from itertools import compress

import numpy as np
from random import choices


class KnapsackProblemFunctions:
    @staticmethod
    def initial_population(individual_size: int, population_size: int) -> list[list[bool]]:
        return [[choices([True, False])[0] for _ in range(individual_size)] for _ in range(population_size)]

    @staticmethod
    def fitness(items: dict, knapsack_max_capacity: int, individual: list[bool]) -> float:
        total_weight = np.sum(np.array(items['Weight']) * np.array(individual))
        if total_weight > knapsack_max_capacity:
            return 0
        return np.sum(np.array(items['Value']) * np.array(individual))

    @staticmethod
    def crossover(parent1: list[bool], parent2: list[bool]) -> tuple[list[bool], list[bool]]:
        crossover_point = np.random.randint(1, len(parent1))
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    @staticmethod
    def mutation(individual: list[bool], probability: int = 0.01) -> list[bool]:
        mutated_individual = individual.copy()
        for i in range(len(mutated_individual)):
            mutated_individual[i] = choices([mutated_individual[i], not mutated_individual[i]],
                                            [1 - probability, probability])[0]
        return mutated_individual

    @staticmethod
    def roulette_wheel_selection(fitness_values: list[float], n_selection: int) -> list[int]:
        total_fitness = sum(fitness_values)
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        selected_indices = np.random.choice(range(len(fitness_values)), n_selection, replace=True, p=probabilities)
        return selected_indices

    @staticmethod
    def decode_solution(items: dict, best_solution: list[bool]) -> tuple[list[str], float]:
        selected_items = list(compress(items['Name'], best_solution))
        selected_values = list(compress(items['Value'], best_solution))
        total_value = sum(selected_values)
        return selected_items, total_value
