import numpy as np
import matplotlib.pyplot as plt
import time

# Function to generate random cities
def generate_cities(num_cities):
    return np.random.rand(num_cities, 2)  # Random 2D coordinates for cities

# Function to calculate the distance matrix between cities
def distance_matrix(cities):
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])
    return dist_matrix

# Function to calculate the total distance of a route
def total_distance(route, dist_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += dist_matrix[route[i], route[i+1]]
    total += dist_matrix[route[-1], route[0]]  # Return to starting city
    return total

# Genetic Algorithm for solving TSP
def genetic_algorithm_tsp(cities, population_size=1000, num_generations=1000, mutation_rate=0.01):
    num_cities = len(cities)
    dist_matrix = distance_matrix(cities)
    
    # Initialize population
    population = [np.random.permutation(num_cities) for _ in range(population_size)]

    # print(population[:10])
    
    for gen_num, generation in enumerate(range(num_generations)):
        # Evaluate fitness of each individual in the population
        fitness = [1 / total_distance(individual, dist_matrix) for individual in population]
        
        # Select parents for reproduction (tournament selection)
        parents = []
        for _ in range(population_size):
            idx1, idx2 = np.random.choice(range(population_size), size=2, replace=False)
            if fitness[idx1] > fitness[idx2]:
                parents.append(population[idx1])
            else:
                parents.append(population[idx2])
            # if(gen_num==1):
            #     print("Printing parent:")
            #     print(parents[-1])
        
        # Create offspring through ordered crossover
        offspring = []
        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i+1]
            crossover_point = np.random.randint(0, num_cities)
            child1, child2 = ordered_crossover(parent1, parent2, crossover_point)
            offspring.extend([child1, child2])
        
        # Mutate offspring
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_points = np.random.choice(range(num_cities), size=2, replace=False)
                offspring[i][mutation_points[0]], offspring[i][mutation_points[1]] = \
                    offspring[i][mutation_points[1]], offspring[i][mutation_points[0]]
        
        # Replace population with offspring
        population = offspring
        # print("10 populations:")
        # print(population[:10])
        # print("------------")
    
    # Return the best individual found
    best_individual = min(population, key=lambda x: total_distance(x, dist_matrix))
    # print(best_individual)
    return best_individual

# # Function to perform ordered crossover
# def ordered_crossover(parent1, parent2, crossover_point):
#     child1 = np.zeros_like(parent1)
#     child2 = np.zeros_like(parent2)
#     child1[:crossover_point] = parent1[:crossover_point]
#     child2[:crossover_point] = parent2[:crossover_point]
#     idx1, idx2 = crossover_point, crossover_point
#     while True:
#         if parent2[idx1] not in child1:
#             child1[idx1] = parent2[idx1]
#         if parent1[idx2] not in child2:
#             child2[idx2] = parent1[idx2]
#         idx1 = (idx1 + 1) % len(parent2)
#         idx2 = (idx2 + 1) % len(parent1)
#         if idx1 == crossover_point:
#             break

#     # print("Printing Child", child1, child2)
#     return child1, child2

# Function to perform ordered crossover
def ordered_crossover(parent1, parent2, crossover_point):
    size = len(parent1)
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)
    child1[:crossover_point] = parent1[:crossover_point]
    child2[:crossover_point] = parent2[:crossover_point]
    
    fill_child_with_remaining_cities(child1, parent2, crossover_point, size)
    fill_child_with_remaining_cities(child2, parent1, crossover_point, size)

    return child1, child2

def fill_child_with_remaining_cities(child, parent, start, end):
    p = 0
    for i in range(start, end):
        while p < end and parent[p] in child:
            p += 1
        if p >= end:
            break
        child[i] = parent[p]
        p += 1

# Function to plot the TSP route
def plot_tsp_route(cities, route):
    plt.figure(figsize=(16, 12))
    plt.scatter(cities[:, 0], cities[:, 1], c='blue', label='Cities')
    for i in range(len(route) - 1):
        plt.plot([cities[route[i], 0], cities[route[i+1], 0]], [cities[route[i], 1], cities[route[i+1], 1]], c='red')
        plt.text(cities[route[i]][0], cities[route[i]][1], str(route[i]), color='black', fontsize=10)  # Add text annotation for each point

# Add text annotation for the last point
    plt.text(cities[route[-1]][0], cities[route[-1]][1], str(route[-1]), color='black', fontsize=10)
    plt.plot([cities[route[-1], 0], cities[route[0], 0]], [cities[route[-1], 1], cities[route[0], 1]], c='red')
    plt.title('TSP Route')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()

# Generate random cities
np.random.seed(0)  # for reproducibility
cities = generate_cities(30)
# print(cities)

# Solve TSP using genetic algorithm
start_time = time.time()
best_route = genetic_algorithm_tsp(cities)
end_time = time.time()
req_time = end_time - start_time
print("The best route for covering all cities is : ")
print(best_route)
# print("\n")
print("\ntime required for calculation:", req_time, "seconds")

# Plot the TSP route
plot_tsp_route(cities, best_route)
