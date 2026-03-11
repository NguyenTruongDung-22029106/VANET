import numpy as np
import random

# Hàm tính toán hiệu năng mạng (Throughput, Packet Drop Rate)
def evaluate_solution(cw_size):
    # Giả lập mô phỏng thông lượng và tỷ lệ mất gói
    # Phương trình Throughput: Tăng theo sigmoid khi CW tăng
    throughput = 1000 / (1 + np.exp(-0.05 * (cw_size - 50)))  # Mô phỏng Throughput
    
    # Phương trình Packet Drop Rate (PDR): Giảm theo sigmoid khi CW tăng
    pdr = 100 / (1 + np.exp(0.05 * (cw_size - 100)))  # Mô phỏng Packet Drop Rate
    
    # Hàm mục tiêu tối ưu hóa (càng cao càng tốt)
    # Phương trình tối ưu hóa: fitness = throughput - pdr
    fitness = throughput - pdr
    return fitness, throughput, pdr

# Thuật toán Artificial Bee Colony Algorithm (ABCA)
def abca_optimization(swarm_size=30, max_iterations=100, cw_range=(32, 512)):
    # Khởi tạo quần thể giải pháp ngẫu nhiên
    population = [random.randint(*cw_range) for _ in range(swarm_size)]
    
    best_solution = None
    best_fitness = float('-inf')
    
    for iteration in range(max_iterations):
        new_population = []
        
        for cw_size in population:
            new_cw = min(max(cw_size + random.randint(-10, 10), cw_range[0]), cw_range[1])
            
            # Đánh giá giải pháp mới
            new_fitness, _, _ = evaluate_solution(new_cw)
            old_fitness, _, _ = evaluate_solution(cw_size)
            
            if new_fitness > old_fitness:
                new_population.append(new_cw)
            else:
                new_population.append(cw_size)
        
        # Pha ong quan sát: Chọn giải pháp tốt hơn từ quần thể mới
        population = sorted(new_population, key=lambda sol: evaluate_solution(sol)[0], reverse=True)
        
        # Pha ong trinh sát: Thay thế giải pháp không cải thiện
        if random.random() < 0.1:
            worst_index = random.randint(0, swarm_size - 1)
            population[worst_index] = random.randint(*cw_range)
        
        # Cập nhật giải pháp tốt nhất
        best_candidate = population[0]
        best_candidate_fitness, _, _ = evaluate_solution(best_candidate)
        if best_candidate_fitness > best_fitness:
            best_solution = best_candidate
            best_fitness = best_candidate_fitness
        
        # In kết quả mỗi vòng lặp
        print(f"Iteration {iteration + 1}: Best CW Size = {best_solution}, Fitness = {best_fitness:.4f}")
    
    return best_solution, best_fitness

# Chạy thuật toán
best_cw, best_value = abca_optimization()
print(f"\nOptimal Contention Window Size: {best_cw} with Fitness: {best_value:.4f}")

