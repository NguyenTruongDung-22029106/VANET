import numpy as np
import random

# Hàm tính toán hiệu năng mạng (Throughput, Latency, Packet Loss)
def evaluate_solution(tcp_window, udp_buffer):
    throughput = 1000 / (1 + np.exp(-0.05 * (tcp_window - 50)))  # Mô phỏng Throughput
    latency = 200 / (1 + np.exp(0.1 * (udp_buffer - 50)))  # Mô phỏng Latency
    packet_loss = 100 / (1 + np.exp(0.05 * (tcp_window + udp_buffer - 100)))  # Mô phỏng Packet Loss
    
    # Hàm mục tiêu tối ưu hóa (càng cao càng tốt)
    fitness = 0.7 * throughput - 0.2 * latency - 0.1 * packet_loss
    return fitness, throughput, latency, packet_loss

# Thuật toán Artificial Bee Colony (ABC)
def abc_optimization(swarm_size=30, max_iterations=100, tcp_range=(85, 1000), udp_range=(216, 4096)):
    # Khởi tạo quần thể giải pháp ngẫu nhiên
    population = [
        (random.randint(*tcp_range), random.randint(*udp_range)) for _ in range(swarm_size)
    ]
    
    best_solution = None
    best_fitness = float('-inf')
    
    for iteration in range(max_iterations):
        new_population = []
        
        for solution in population:
            tcp_window, udp_buffer = solution
            new_tcp = min(max(tcp_window + random.randint(-10, 10), tcp_range[0]), tcp_range[1])
            new_udp = min(max(udp_buffer + random.randint(-10, 10), udp_range[0]), udp_range[1])
            
            # Đánh giá giải pháp mới
            new_fitness, _, _, _ = evaluate_solution(new_tcp, new_udp)
            old_fitness, _, _, _ = evaluate_solution(tcp_window, udp_buffer)
            
            if new_fitness > old_fitness:
                new_population.append((new_tcp, new_udp))
            else:
                new_population.append(solution)
        
        # Pha ong quan sát: Chọn giải pháp tốt hơn từ quần thể mới
        population = sorted(new_population, key=lambda sol: evaluate_solution(*sol)[0], reverse=True)
        
        # Pha ong trinh sát: Thay thế giải pháp không cải thiện
        if random.random() < 0.1:
            worst_index = random.randint(0, swarm_size - 1)
            population[worst_index] = (random.randint(*tcp_range), random.randint(*udp_range))
        
        # Cập nhật giải pháp tốt nhất
        best_candidate = population[0]
        best_candidate_fitness, _, _, _ = evaluate_solution(*best_candidate)
        if best_candidate_fitness > best_fitness:
            best_solution = best_candidate
            best_fitness = best_candidate_fitness
        
        # In kết quả mỗi vòng lặp
        print(f"Iteration {iteration + 1}: Best Solution = {best_solution}, Fitness = {best_fitness:.4f}")
    
    return best_solution, best_fitness

# Chạy thuật toán
best_config, best_value = abc_optimization()
print(f"\nOptimal TCP Window Size & UDP Buffer Size: {best_config} with Fitness: {best_value:.4f}")

