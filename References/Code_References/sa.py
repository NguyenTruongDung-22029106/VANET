import numpy as np

# Define the fitness function based on the contention window size
def fitness_function(contention_window_size):
    return abs(contention_window_size - 1000)

# Simulated Annealing Algorithm
def simulated_annealing(initial_solution, max_iter=1000, temp=1000, cooling_rate=0.95):
    current_solution = initial_solution
    current_fitness = fitness_function(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness

    for i in range(max_iter):
        # Generate a new solution by a small random change
        new_solution = current_solution + np.random.uniform(-50, 50)
        new_solution = np.clip(new_solution, 0, 1024)  # Ensure it's within bounds
        new_fitness = fitness_function(new_solution)

        # Acceptance probability
        delta = new_fitness - current_fitness
        if delta < 0 or np.exp(-delta / temp) > np.random.rand():
            current_solution = new_solution
            current_fitness = new_fitness

        # Update best solution
        if new_fitness < best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness

        # Decrease temperature
        temp *= cooling_rate

    return best_solution

# Initial contention window size for 10 cars
initial_contention_window_size = 85.3

# Run SA algorithm
optimal_contention_window_size = simulated_annealing(initial_contention_window_size)

print("Optimal contention window size for 10 cars:", optimal_contention_window_size)

