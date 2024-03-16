import time
import pandas as pd

from GeneticAlgorithm import GeneticAlgorithm
from KnapsackProblemFunctions import KnapsackProblemFunctions as kpf


def main():
    items = pd.read_csv('knapsack-big.csv')
    knapsack_max_capacity = 6404180

    initial_population = kpf.initial_population(len(items['Value']), population_size=100)

    start_time = time.time()

    algorithm = GeneticAlgorithm(
        fitness_function=lambda individual: kpf.fitness(items, knapsack_max_capacity, individual),
        initial_population=initial_population,
        crossover_function=kpf.crossover,
        mutation_function=kpf.mutation,
        selection_function=kpf.roulette_wheel_selection
    )
    algorithm.run()

    end_time = time.time()
    total_time = end_time - start_time
    print('Time: ', total_time)

    selected_items, total_value = kpf.decode_solution(items, algorithm.best_solution)
    print("Selected items:", selected_items)
    print("Total value:", total_value)
    algorithm.plot()


if __name__ == '__main__':
    main()
