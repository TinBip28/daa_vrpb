# vrpb_rl_simple_pg/ga_core.py
import random
import numpy as np
import time
from vrpb_ga_utils import decode_chromosome_and_calc_fitness, PENALTY_UNSERVED_CUSTOMER


# --- Các toán tử di truyền ---

def initialize_population(num_customers, pop_size):
    """Khởi tạo quần thể ban đầu với các hoán vị ngẫu nhiên của khách hàng."""
    population = []
    customer_ids = list(range(1, num_customers + 1)) # ID khách hàng từ 1 đến N
    for _ in range(pop_size):
        perm = random.sample(customer_ids, len(customer_ids))
        population.append(perm)
    return population

def selection(population_with_fitness, tournament_size):
    """
    Thực hiện chọn lọc giải đấu (Tournament Selection).
    Chọn một số lượng cha mẹ bằng kích thước quần thể để tạo thế hệ mới.
    """
    selected_parents = []
    num_parents_to_select = len(population_with_fitness)

    for _ in range(num_parents_to_select):
        tournament = random.sample(population_with_fitness, tournament_size)
        # Sắp xếp các cá thể trong giải đấu theo fitness (thấp hơn là tốt hơn)
        # Mỗi phần tử trong population_with_fitness là (chromosome, fitness, dist, pen, tours)
        winner = min(tournament, key=lambda ind: ind[1]) # ind[1] là fitness
        selected_parents.append(winner[0]) # Chỉ lấy nhiễm sắc thể của người thắng cuộc
    return selected_parents

def order_crossover(parent1, parent2, crossover_rate):
    """
    Thực hiện lai ghép thứ tự (Order Crossover - OX1).
    Trả về hai con. Con có thể giống hệt cha mẹ nếu không xảy ra lai ghép.
    """
    if random.random() > crossover_rate:
        return list(parent1), list(parent2) # Trả về bản sao của cha mẹ

    size = len(parent1)
    child1, child2 = [-1]*size, [-1]*size # Khởi tạo con với giá trị placeholder

    # Chọn ngẫu nhiên một đoạn gen (segment)
    start, end = sorted(random.sample(range(size), 2))

    # Sao chép đoạn gen từ cha mẹ sang con
    child1[start:end+1] = parent1[start:end+1]
    child2[start:end+1] = parent2[start:end+1]

    # Điền các gen còn lại cho child1 từ parent2
    # Duyệt parent2, nếu gen chưa có trong child1 thì thêm vào
    p2_idx = (end + 1) % size
    c1_idx = (end + 1) % size
    while child1[c1_idx] != -1: # Tìm vị trí trống đầu tiên sau đoạn đã copy
        c1_idx = (c1_idx + 1) % size
        if c1_idx == (end + 1) % size and -1 not in child1: break # Đã đầy

    temp_p2_fill_idx = p2_idx
    while -1 in child1: # Chừng nào child1 còn chỗ trống
        item_from_p2 = parent2[temp_p2_fill_idx]
        if item_from_p2 not in child1: # Chỉ thêm nếu gen chưa có
            child1[c1_idx] = item_from_p2
            c1_idx = (c1_idx + 1) % size
            while child1[c1_idx] != -1 : # Tìm vị trí trống tiếp theo
                 c1_idx = (c1_idx + 1) % size
                 if c1_idx == (end + 1) % size and -1 not in child1: break # Đã đầy
        temp_p2_fill_idx = (temp_p2_fill_idx + 1) % size
        if temp_p2_fill_idx == p2_idx and -1 in child1: # Đã duyệt hết parent2 mà vẫn còn trống -> lỗi
            # print("Lỗi OX1: Không thể điền child1")
            # Fallback: điền các giá trị còn thiếu từ parent1 để đảm bảo tính hợp lệ
            missing_in_child1 = [gene for gene in parent1 if gene not in child1]
            idx_fill = 0
            for i in range(size):
                if child1[i] == -1: child1[i] = missing_in_child1[idx_fill]; idx_fill+=1
            break


    # Điền các gen còn lại cho child2 từ parent1
    p1_idx = (end + 1) % size
    c2_idx = (end + 1) % size
    while child2[c2_idx] != -1:
        c2_idx = (c2_idx + 1) % size
        if c2_idx == (end + 1) % size and -1 not in child2: break

    temp_p1_fill_idx = p1_idx
    while -1 in child2:
        item_from_p1 = parent1[temp_p1_fill_idx]
        if item_from_p1 not in child2:
            child2[c2_idx] = item_from_p1
            c2_idx = (c2_idx + 1) % size
            while child2[c2_idx] != -1 :
                 c2_idx = (c2_idx + 1) % size
                 if c2_idx == (end + 1) % size and -1 not in child2: break
        temp_p1_fill_idx = (temp_p1_fill_idx + 1) % size
        if temp_p1_fill_idx == p1_idx and -1 in child2:
            # print("Lỗi OX1: Không thể điền child2")
            missing_in_child2 = [gene for gene in parent2 if gene not in child2]
            idx_fill = 0
            for i in range(size):
                if child2[i] == -1: child2[i] = missing_in_child2[idx_fill]; idx_fill+=1
            break
            
    return child1, child2

def swap_mutation(chromosome, mutation_rate):
    """Thực hiện đột biến hoán đổi (Swap Mutation)."""
    if random.random() < mutation_rate:
        if len(chromosome) >= 2:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

# --- Hàm chính của Thuật toán Di truyền ---
def run_ga_for_vrpb(instance_data, problem_type, ga_config):
    """
    Chạy thuật toán di truyền để giải bài toán VRPB.

    Args:
        instance_data (dict): Dữ liệu của bài toán.
        problem_type (str): "traditional" hoặc "improved".
        ga_config (dict): Các tham số cấu hình cho GA.

    Returns:
        tuple: (best_solution_tours, best_overall_distance)
               Trả về (None, float('inf')) nếu không tìm thấy giải pháp hợp lệ.
    """
    print(f"Khởi chạy GA cho VRPB (Loại: {problem_type})...")
    num_customers = instance_data["num_customers"]

    # Lấy các tham số GA từ config
    pop_size = ga_config.get('POPULATION_SIZE', 50)
    num_generations = ga_config.get('NUM_GENERATIONS', 100)
    mutation_rate = ga_config.get('MUTATION_RATE', 0.1)
    crossover_rate = ga_config.get('CROSSOVER_RATE', 0.8)
    tournament_size = ga_config.get('TOURNAMENT_SIZE', 5)
    elitism_count = ga_config.get('ELITISM_COUNT', 2)
    apply_ls = ga_config.get('APPLY_LOCAL_SEARCH', True)

    # 1. Khởi tạo quần thể
    population = initialize_population(num_customers, pop_size)

    best_overall_chromosome = None
    best_overall_fitness = float('inf')
    best_overall_distance_component = float('inf') # Chỉ phần quãng đường
    best_overall_tours = None

    print(f"Số thế hệ: {num_generations}, Kích thước quần thể: {pop_size}, Tỷ lệ lai ghép: {crossover_rate}, Tỷ lệ đột biến: {mutation_rate}, Local Search: {apply_ls}")

    for gen in range(num_generations):
        start_gen_time = time.time()

        # 2. Đánh giá quần thể (Tính fitness cho mỗi cá thể)
        population_with_fitness_details = [] # (chromosome, fitness, distance_comp, penalty_comp, tours_decoded)
        for chromo in population:
            fitness, dist_comp, pen_comp, tours_decoded = decode_chromosome_and_calc_fitness(
                chromo, instance_data, problem_type, apply_local_search=apply_ls
            )
            population_with_fitness_details.append((list(chromo), fitness, dist_comp, pen_comp, tours_decoded))

        # Sắp xếp quần thể theo fitness (thấp hơn là tốt hơn)
        population_with_fitness_details.sort(key=lambda ind: ind[1])

        # Cập nhật giải pháp tốt nhất toàn cục
        current_gen_best_chromosome, current_gen_best_fitness, current_gen_best_dist_comp, current_gen_best_pen_comp, current_gen_best_tours = population_with_fitness_details[0]

        if current_gen_best_fitness < best_overall_fitness:
            best_overall_fitness = current_gen_best_fitness
            best_overall_chromosome = list(current_gen_best_chromosome) # Lưu bản sao
            best_overall_distance_component = current_gen_best_dist_comp
            best_overall_tours = current_gen_best_tours
        
        avg_fitness_current_gen = np.mean([ind[1] for ind in population_with_fitness_details])

        if (gen + 1) % 10 == 0 or gen == 0 or gen == num_generations -1 :
            print(f"Thế hệ {gen+1}/{num_generations} - Fitness tốt nhất: {best_overall_fitness:.2f} (Quãng đường: {best_overall_distance_component:.2f}, Phạt: {best_overall_fitness - best_overall_distance_component:.2f}), Fitness trung bình: {avg_fitness_current_gen:.2f}")

        # 3. Tạo quần thể mới
        new_population = []

        # Giữ lại các cá thể tốt nhất (Elitism)
        for i in range(elitism_count):
            if i < len(population_with_fitness_details):
                new_population.append(list(population_with_fitness_details[i][0])) # Chỉ lấy nhiễm sắc thể

        # Điền phần còn lại của quần thể mới bằng chọn lọc, lai ghép, đột biến
        num_offspring_to_create = pop_size - len(new_population)
        
        # Chọn lọc cha mẹ từ quần thể hiện tại (đã có fitness)
        # `selection` trả về danh sách các nhiễm sắc thể cha mẹ được chọn
        potential_parents = selection(population_with_fitness_details, tournament_size)
        
        created_offspring_count = 0
        parent_pool_idx = 0
        while created_offspring_count < num_offspring_to_create:
            # Chọn 2 cha mẹ từ danh sách đã chọn lọc để lai ghép
            # Đảm bảo có đủ cha mẹ trong potential_parents hoặc lặp lại nếu cần
            parent1_chromo = potential_parents[parent_pool_idx % len(potential_parents)]
            parent_pool_idx +=1
            parent2_chromo = potential_parents[parent_pool_idx % len(potential_parents)]
            parent_pool_idx +=1

            child1_chromo, child2_chromo = order_crossover(parent1_chromo, parent2_chromo, crossover_rate)
            
            child1_chromo = swap_mutation(child1_chromo, mutation_rate)
            child2_chromo = swap_mutation(child2_chromo, mutation_rate)
            
            new_population.append(child1_chromo)
            created_offspring_count += 1
            if created_offspring_count < num_offspring_to_create:
                new_population.append(child2_chromo)
                created_offspring_count += 1
                
        population = new_population # Quần thể mới cho thế hệ tiếp theo
        # print(f"  Thời gian thế hệ: {time.time() - start_gen_time:.2f}s")

    print(f"\nGA hoàn thành. Fitness tốt nhất toàn cục: {best_overall_fitness:.2f}")

    # Kiểm tra giải pháp tốt nhất cuối cùng một lần nữa (không local search) để đảm bảo tính nhất quán
    # và kiểm tra xem nó có thực sự hợp lệ không (penalty thấp)
    if best_overall_chromosome is None:
        print("Không tìm thấy giải pháp nào.")
        return None, float('inf')

    final_fitness, final_dist, final_penalty, final_tours = decode_chromosome_and_calc_fitness(
        best_overall_chromosome, instance_data, problem_type, apply_local_search=True
    )
    
    print(f"Giải pháp cuối cùng - Quãng đường: {final_dist:.2f}, Hình phạt: {final_penalty:.2f}, Fitness: {final_fitness:.2f}")

    # Nếu hình phạt vẫn còn đáng kể, giải pháp có thể không hoàn toàn hợp lệ
    # Ngưỡng này có thể cần điều chỉnh.
    # PENALTY_UNSERVED_CUSTOMER là lớn nhất, nếu nó xuất hiện thì chắc chắn không hợp lệ.
    if final_penalty >= PENALTY_UNSERVED_CUSTOMER / 10: # Chia 10 để cho phép một số lỗi nhỏ khác
        print("CẢNH BÁO: Giải pháp tốt nhất tìm thấy bởi GA vẫn có hình phạt đáng kể hoặc không phục vụ hết khách hàng.")
        
        num_total_customers = instance_data["num_customers"]
        all_instance_customer_ids = set(range(1, num_total_customers + 1))
        served_in_final_solution = set()
        if final_tours:
            for t_nodes in final_tours:
                for node_id_in_t in t_nodes:
                    if node_id_in_t != instance_data["depot_idx"]:
                        served_in_final_solution.add(node_id_in_t)
        
        if len(served_in_final_solution) != num_total_customers:
            print(f"  Không phải tất cả khách hàng được phục vụ. Thiếu: {all_instance_customer_ids - served_in_final_solution}")
            return None, float('inf') # Coi như không hợp lệ nếu không phục vụ hết
        # Nếu phục vụ hết nhưng vẫn có penalty khác (ví dụ: quá tải nhẹ), vẫn trả về nhưng cảnh báo
        print("  Có thể có vi phạm ràng buộc khác (tải trọng, thứ tự).")

    return final_tours, final_dist # Chỉ trả về quãng đường nếu giải pháp được xem là "đủ tốt"