import numpy as np

# Define the fitness function based on the contention window size
def fitness_function(contention_window_size):

    return abs(contention_window_size - 1000)  

# Implement the PSO algorithm with the additional steps for 10 cars
def pso(contention_window_size, num_particles=10, max_iter=100):
    # Initialize parameters
    inertia_weight = 0.9
    cognitive_weight = 2
    social_weight = 2
    scaling_factor_1 = 0.5
    scaling_factor_2 = 0.3
    min_window_size = 0
    max_window_size = 1024
    dimensions = 1

    # Initialize particles with random positions and velocities
    particles_position = np.random.uniform(min_window_size, max_window_size, (num_particles, dimensions))
    particles_velocity = np.zeros((num_particles, dimensions))
    personal_best = particles_position.copy()
    global_best = particles_position[np.argmin(fitness_function(particles_position))]

    # PSO iterations
    for i in range(max_iter):
        for j in range(num_particles):
            # Update particle velocity
            particles_velocity[j] = inertia_weight * particles_velocity[j] + \
                                     cognitive_weight * np.random.rand() * (personal_best[j] - particles_position[j]) + \
                                     social_weight * np.random.rand() * (global_best - particles_position[j])
            # Update particle position using the provided formula
            rand_1 = np.random.uniform(0, 1)
            rand_2 = np.random.uniform(0, 1)
            particles_position[j] = particles_position[j] + scaling_factor_1 * rand_1 * (personal_best[j] - particles_position[j]) + \
                                    scaling_factor_2 * rand_2 * (global_best - particles_position[j])
            
            # Ensure particle positions are within bounds
            particles_position[j] = np.clip(particles_position[j], min_window_size, max_window_size)
            
            # Update personal best
            if fitness_function(particles_position[j]) < fitness_function(personal_best[j]):
                personal_best[j] = particles_position[j]
            
        # Update global best
        current_best_index = np.argmin(fitness_function(particles_position))
        if fitness_function(particles_position[current_best_index]) < fitness_function(global_best):
            global_best = particles_position[current_best_index]

    return global_best

# Initial contention window size for 10 cars
initial_contention_window_size = 85.3

# Run PSO algorithm for 10 cars
optimal_contention_window_size = pso(initial_contention_window_size, num_particles=10, max_iter=100)

print("Optimal contention window size for 10 cars:", optimal_contention_window_size)


