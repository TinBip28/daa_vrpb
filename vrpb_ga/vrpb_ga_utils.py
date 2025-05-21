# vrpb_rl_simple_pg/vrpb_ga_utils.py
import random
import numpy as np
from utils import calculate_distance, get_node_type_from_index # Đảm bảo utils.py có các hàm này

# --- Các hằng số hình phạt (Penalties) ---
# Có thể được truyền vào từ file thực thi chính nếu muốn linh hoạt hơn
PENALTY_CAPACITY_MULTIPLIER = 200
PENALTY_ORDER = 5000
PENALTY_UNSERVED_CUSTOMER = 200000

def apply_2_opt_on_tour(tour_nodes, coords):
    """
    Áp dụng thuật toán 2-opt để cải thiện một lộ trình đơn lẻ.
    Một lộ trình bao gồm điểm bắt đầu và kết thúc là depot.
    Ví dụ: [0, 1, 2, 0]
    """
    if not tour_nodes or len(tour_nodes) <= 3: # Cần ít nhất 2 khách hàng (4 nút)
        return tour_nodes

    current_tour = list(tour_nodes)
    num_nodes_in_tour = len(current_tour)
    improved = True

    while improved:
        improved = False
        min_change_in_iteration = 0 # Theo dõi thay đổi tốt nhất trong một vòng lặp 2-opt

        # Chỉ lặp qua các nút khách hàng, bỏ qua depot ở đầu và cuối
        # Cạnh (i, i+1) và (j, j+1)
        # Ta sẽ tạo các cạnh mới (i, j) và (i+1, j+1)
        # và đảo ngược đoạn từ i+1 đến j.
        # Các chỉ số i, j ở đây là chỉ số trong list current_tour.
        # Depot đầu là current_tour[0], depot cuối là current_tour[num_nodes_in_tour-1]
        # Khách hàng nằm từ current_tour[1] đến current_tour[num_nodes_in_tour-2]

        for i in range(num_nodes_in_tour - 3): # i từ 0 (depot) đến num_nodes_in_tour - 4
                                               # để i+1 không phải là depot cuối
            for j in range(i + 2, num_nodes_in_tour - 1): # j từ i+2 đến num_nodes_in_tour - 2
                                                          # để j không phải là depot cuối và (j, j+1) hợp lệ

                # original_edge1 = (current_tour[i], current_tour[i+1])
                # original_edge2 = (current_tour[j], current_tour[j+1])
                # new_edge1 = (current_tour[i], current_tour[j])
                # new_edge2 = (current_tour[i+1], current_tour[j+1])

                # Kiểm tra nếu i là depot và j là khách hàng cuối cùng trước depot cuối
                # hoặc i+1 là khách hàng đầu tiên và j+1 là depot cuối.
                # Điều này đảm bảo các cạnh đang xem xét là hợp lệ.
                # current_tour[0] ... current_tour[i] -- current_tour[i+1] ... current_tour[j] -- current_tour[j+1] ... current_tour[end]

                original_dist = (calculate_distance(coords[current_tour[i]], coords[current_tour[i+1]]) +
                                 calculate_distance(coords[current_tour[j]], coords[current_tour[j+1]]))

                new_dist = (calculate_distance(coords[current_tour[i]], coords[current_tour[j]]) +
                            calculate_distance(coords[current_tour[i+1]], coords[current_tour[j+1]]))

                change = new_dist - original_dist

                if change < min_change_in_iteration - 1e-9: # Cải thiện đáng kể (tránh lỗi float)
                    min_change_in_iteration = change
                    # Thực hiện swap (đảo ngược đoạn giữa i+1 và j)
                    segment_to_reverse = current_tour[i+1 : j+1]
                    segment_to_reverse.reverse()
                    current_tour[i+1 : j+1] = segment_to_reverse
                    improved = True
                    # Sau khi thực hiện một cải tiến, bắt đầu lại vòng lặp while (best improvement strategy)
                    # hoặc tiếp tục tìm kiếm trong vòng lặp for hiện tại (first improvement strategy)
                    # Ở đây đang dùng first improvement trong vòng for, rồi lặp lại while nếu có cải thiện.
    return current_tour


def decode_chromosome_and_calc_fitness(chromosome_perm, instance_data, problem_type, apply_local_search=True):
    """
    Giải mã một hoán vị khách hàng (nhiễm sắc thể) thành các lộ trình,
    tính toán tổng quãng đường và các hình phạt.
    Tùy chọn áp dụng tìm kiếm cục bộ 2-opt cho mỗi lộ trình.

    Args:
        chromosome_perm (list): Hoán vị các ID khách hàng.
        instance_data (dict): Dữ liệu của bài toán VRPB.
        problem_type (str): "traditional" hoặc "improved".
        apply_local_search (bool): True nếu muốn áp dụng 2-opt.

    Returns:
        tuple: (fitness, total_distance, total_penalty, tours)
    """
    tours = []
    current_tour_nodes = [instance_data["depot_idx"]]
    current_lh_payload = 0
    current_bh_collected = 0
    backhaul_started_in_current_tour = False # Dùng cho 'traditional' VRPB

    demands = instance_data["demands"]
    coords = instance_data["coords"]
    vehicle_capacity = instance_data["vehicle_capacity"]
    num_linehaul = instance_data["num_linehaul"]
    depot_idx = instance_data["depot_idx"]
    num_total_customers = instance_data["num_customers"]

    customers_to_route = list(chromosome_perm)
    routed_customers_this_chromosome = set()


    # Xây dựng các lộ trình từ hoán vị khách hàng
    idx_customer_in_permutation = 0
    while idx_customer_in_permutation < len(customers_to_route):
        cust_idx = customers_to_route[idx_customer_in_permutation]
        cust_demand_val = demands[cust_idx]
        cust_type = get_node_type_from_index(cust_idx, num_linehaul)

        can_add_to_current_tour = True
        # Kiểm tra ràng buộc khi thêm khách hàng vào lộ trình hiện tại
        if cust_type == "linehaul":
            if current_lh_payload + cust_demand_val > vehicle_capacity:
                can_add_to_current_tour = False
            if problem_type == "traditional" and backhaul_started_in_current_tour:
                can_add_to_current_tour = False # Không thêm linehaul nếu đã có backhaul trong tour
        elif cust_type == "backhaul":
            if current_bh_collected + abs(cust_demand_val) > vehicle_capacity:
                can_add_to_current_tour = False
        
        if can_add_to_current_tour:
            current_tour_nodes.append(cust_idx)
            routed_customers_this_chromosome.add(cust_idx)
            if cust_type == "linehaul":
                current_lh_payload += cust_demand_val
            elif cust_type == "backhaul":
                current_bh_collected += abs(cust_demand_val)
                if problem_type == "traditional":
                    backhaul_started_in_current_tour = True
            idx_customer_in_permutation += 1
        else:
            # Không thể thêm khách hàng này vào tour hiện tại -> kết thúc tour hiện tại
            if len(current_tour_nodes) > 1: # Nếu tour hiện tại có khách hàng
                current_tour_nodes.append(depot_idx)
                tours.append(list(current_tour_nodes))
            
            # Bắt đầu tour mới với khách hàng hiện tại (nếu có)
            current_tour_nodes = [depot_idx]
            current_lh_payload = 0
            current_bh_collected = 0
            backhaul_started_in_current_tour = False
            # Khách hàng hiện tại (cust_idx) sẽ được xem xét lại để bắt đầu tour mới ở lần lặp tiếp theo
            # của vòng while, nếu can_add_to_current_tour là False thì nó sẽ được thử thêm vào tour mới ngay.
            # Logic này đảm bảo khách hàng không bị bỏ qua.
            # Nếu khách hàng đầu tiên trong một hoán vị không thể tạo thành tour (ví dụ: quá tải)
            # thì sẽ có vấn đề. Giả định instance hợp lệ để một khách hàng luôn có thể được phục vụ bởi một xe rỗng.
    
    # Thêm tour cuối cùng nếu còn khách hàng
    if len(current_tour_nodes) > 1:
        current_tour_nodes.append(depot_idx)
        tours.append(list(current_tour_nodes))

    # --- Áp dụng Tìm kiếm Cục bộ (2-opt) ---
    if apply_local_search:
        optimized_tours = []
        for tour in tours:
            # 2-opt cần ít nhất 2 khách hàng (4 nút bao gồm depot) để có thể hoán đổi cạnh
            # Nếu tour là [0, C1, 0], không làm gì cả.
            # Nếu tour là [0, C1, C2, 0], có thể áp dụng.
            if len(tour) >= 4:
                optimized_tours.append(apply_2_opt_on_tour(tour, coords))
            else:
                optimized_tours.append(tour)
        tours = optimized_tours

    # --- Tính toán tổng quãng đường và hình phạt cho các lộ trình (đã được tối ưu nếu có) ---
    total_distance = 0
    total_penalty = 0
    final_served_customers_in_solution = set()

    for tour_nodes in tours:
        if not tour_nodes or len(tour_nodes) < 2: continue

        tour_dist_iter = 0
        current_lh_payload_eval = 0
        current_bh_collected_eval = 0
        tour_backhaul_started_eval = False
        tour_has_order_violation = False

        # Kiểm tra lại ràng buộc cho mỗi tour sau khi đã hình thành (và có thể đã local search)
        for i in range(len(tour_nodes) - 1):
            u, v = tour_nodes[i], tour_nodes[i+1]
            total_distance += calculate_distance(coords[u], coords[v]) # Tổng quãng đường chung
            tour_dist_iter += calculate_distance(coords[u], coords[v])


            if v != depot_idx: # Chỉ xử lý nếu v là khách hàng
                final_served_customers_in_solution.add(v)
                demand_v = demands[v]
                type_v = get_node_type_from_index(v, num_linehaul)

                if type_v == "linehaul":
                    current_lh_payload_eval += demand_v
                    if problem_type == "traditional" and tour_backhaul_started_eval:
                        tour_has_order_violation = True
                elif type_v == "backhaul":
                    current_bh_collected_eval += abs(demand_v)
                    if problem_type == "traditional":
                        tour_backhaul_started_eval = True
                
                # Kiểm tra tải trọng tại mỗi bước dừng ở khách hàng v
                if current_lh_payload_eval > vehicle_capacity:
                    total_penalty += (current_lh_payload_eval - vehicle_capacity) * PENALTY_CAPACITY_MULTIPLIER
                    # Reset payload để tránh phạt nhiều lần cho cùng một lượng quá tải trong tour
                    # Hoặc phạt một lần cuối tour. Phạt tại mỗi bước sẽ nghiêm khắc hơn.
                    # Ở đây, ta cộng dồn hình phạt nếu nó tiếp tục quá tải.
                if current_bh_collected_eval > vehicle_capacity:
                    total_penalty += (current_bh_collected_eval - vehicle_capacity) * PENALTY_CAPACITY_MULTIPLIER
        
        if tour_has_order_violation:
            total_penalty += PENALTY_ORDER
    
    # Hình phạt nếu không phải tất cả khách hàng trong instance được phục vụ
    all_instance_customer_ids = set(range(1, num_total_customers + 1))
    if len(final_served_customers_in_solution) != num_total_customers:
        missing_customers = len(all_instance_customer_ids - final_served_customers_in_solution)
        total_penalty += PENALTY_UNSERVED_CUSTOMER * missing_customers

    fitness = total_distance + total_penalty
    return fitness, total_distance, total_penalty, tours