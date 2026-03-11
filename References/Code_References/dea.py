import numpy as np

def objective_function(window_size):
    # Đánh giá hiệu suất mạng dựa trên network_config và trả về giá trị mục tiêu
    # ...
    return abs(float((window_size - 1000)))

def differential_evolutionary_algorithm(bounds, population_size=50, scaling_factor=2, max_generations=100):
    # Khởi tạo quần thể ban đầu
    population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(population_size, len(bounds)))
    
    # Tiến hành tối ưu hóa
    for generation in range(max_generations):
        for i in range(population_size):
            # Chọn ba chỉ số ngẫu nhiên từ 0 đến population_size, loại trừ chỉ số i
            candidate_parent_indices = np.random.choice([j for j in range(population_size) if j != i], size=3, replace=False)
            candidate_parents = population[candidate_parent_indices]
            
            # Tạo vectơ con cái mới bằng cách áp dụng toán tử crossover và mutation
            candidate_child = candidate_parents[0] + scaling_factor * (candidate_parents[1] - candidate_parents[2])
            candidate_child = np.clip(candidate_child, bounds[:, 0], bounds[:, 1])  # Đảm bảo rằng vectơ con cái nằm trong ranh giới
        
            # So sánh giá trị của vectơ con cái với cha mẹ và thay thế nếu nó tốt hơn
            if objective_function(candidate_child) < objective_function(population[i]):
                population[i] = candidate_child
    
    # Trả về vectơ tốt nhất tìm được
    best_solution = population[np.argmin([objective_function(ind) for ind in population])]
    return best_solution


# Modify the bounds array to represent the range of window sizes to search within

bounds = np.array([[0, 1024]])

# Apply DEA to optimize the window size
best_window_size = differential_evolutionary_algorithm(bounds)

# Print the best window size and its corresponding objective value
print("Best window size:", best_window_size)
print("Corresponding objective function value:", objective_function(best_window_size))



