import numpy as np

# Hàm mục tiêu đánh giá hiệu suất mạng dựa trên kích thước Contention Window (CW)
def fitness_function(contention_window_size):
    return abs(contention_window_size - 1000)  # Giả sử giá trị tối ưu của CW là 1000

# Thuật toán Artificial Bee Colony Algorithm (ABCA)
def abca(bounds, num_bees=10, max_iter=100, limit=5):
    lower_bound, upper_bound = bounds
    
    # Khởi tạo quần thể ong với vị trí CW ngẫu nhiên
    bees = np.random.uniform(lower_bound, upper_bound, num_bees)
    fitness = np.array([fitness_function(b) for b in bees])
    
    # Bộ nhớ để theo dõi số lần một ong không thể cải thiện
    trial = np.zeros(num_bees)
    
    for iteration in range(max_iter):
        # Giai đoạn Worker Bee - Tìm kiếm lân cận
        for i in range(num_bees):
            neighbor_idx = np.random.randint(0, num_bees)
            while neighbor_idx == i:
                neighbor_idx = np.random.randint(0, num_bees)
            
            phi = np.random.uniform(-1, 1)  # Hệ số điều chỉnh ngẫu nhiên
            new_bee = bees[i] + phi * (bees[i] - bees[neighbor_idx])
            new_bee = np.clip(new_bee, lower_bound, upper_bound)  # Đảm bảo trong giới hạn
            
            new_fitness = fitness_function(new_bee)
            
            # Chấp nhận giải pháp mới nếu tốt hơn
            if new_fitness < fitness[i]:
                bees[i] = new_bee
                fitness[i] = new_fitness
                trial[i] = 0  # Reset bộ đếm thử nghiệm
            else:
                trial[i] += 1  # Tăng số lần thất bại
        
        # Giai đoạn Onlooker Bee - Lựa chọn dựa trên xác suất
        prob = (1 / (1 + fitness)) / np.sum(1 / (1 + fitness))  # Xác suất chọn dựa trên fitness
        for i in range(num_bees):
            if np.random.rand() < prob[i]:
                neighbor_idx = np.random.randint(0, num_bees)
                while neighbor_idx == i:
                    neighbor_idx = np.random.randint(0, num_bees)
                
                phi = np.random.uniform(-1, 1)
                new_bee = bees[i] + phi * (bees[i] - bees[neighbor_idx])
                new_bee = np.clip(new_bee, lower_bound, upper_bound)
                new_fitness = fitness_function(new_bee)
                
                if new_fitness < fitness[i]:
                    bees[i] = new_bee
                    fitness[i] = new_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1
        
        # Giai đoạn Scout Bee - Khám phá vùng tìm kiếm mới nếu không cải thiện
        for i in range(num_bees):
            if trial[i] > limit:  # Nếu ong không cải thiện sau 'limit' lần
                bees[i] = np.random.uniform(lower_bound, upper_bound)
                fitness[i] = fitness_function(bees[i])
                trial[i] = 0
        
        # Điều kiện dừng sớm nếu tìm thấy giá trị CW tối ưu
        if np.min(fitness) == 0:
            break
    
    best_index = np.argmin(fitness)
    return bees[best_index]

# Chạy thuật toán ABCA để tìm giá trị CW tối ưu
bounds = (0, 1024)  # Giới hạn tìm kiếm CW

optimal_contention_window_size = abca(bounds, num_bees=10, max_iter=100)

print("Optimal contention window size using ABCA:", optimal_contention_window_size)

