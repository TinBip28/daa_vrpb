# vrpb_rl_simple_pg/vrpb_ga_utils.py
import random
import numpy as np
from utils import calculate_distance, get_node_type_from_index # Đảm bảo utils.py có các hàm này

# --- Các hằng số hình phạt (Penalties) ---
# Có thể được truyền vào từ file thực thi chính nếu muốn linh hoạt hơn
PENALTY_CAPACITY_MULTIPLIER = 200
PENALTY_ORDER = 5000
PENALTY_UNSERVED_CUSTOMER = 200000

def is_traditional_tour_valid(tour_nodes, instance_data):
    """
    Kiểm tra xem một lộ trình đơn lẻ có tuân thủ các quy tắc của VRPB Truyền thống không.
    1. Tất cả linehaul phải đến trước tất cả backhaul.
    2. Tổng demand của đoạn linehaul không vượt quá dung lượng.
    3. Tổng demand (absolute) của đoạn backhaul không vượt quá dung lượng.
    Lộ trình tour_nodes bao gồm depot ở đầu và cuối.
    """
    if not tour_nodes or len(tour_nodes) < 2: # Tour rỗng hoặc không hợp lệ
        return True # Coi như hợp lệ nếu không có gì để kiểm tra

    demands = instance_data["demands"] # LH > 0, BH < 0
    vehicle_capacity = instance_data["vehicle_capacity"]
    num_linehaul_customers = instance_data["num_linehaul"]
    depot_idx = instance_data["depot_idx"]

    customer_nodes_in_tour = [node for node in tour_nodes if node != depot_idx]
    if not customer_nodes_in_tour: # Tour chỉ có depot [0,0]
        return True

    # 1. Kiểm tra thứ tự Linehaul - Backhaul
    linehaul_phase_ended = False
    for node_idx in customer_nodes_in_tour:
        node_type = get_node_type_from_index(node_idx, num_linehaul_customers)
        if node_type == "linehaul":
            if linehaul_phase_ended:
                return False # Lỗi: Linehaul sau khi đã có Backhaul
        elif node_type == "backhaul":
            linehaul_phase_ended = True

    # 2. Kiểm tra dung lượng cho đoạn Linehaul và Backhaul riêng biệt
    current_linehaul_demand_sum = 0
    current_backhaul_demand_sum = 0

    for node_idx in customer_nodes_in_tour:
        node_type = get_node_type_from_index(node_idx, num_linehaul_customers)
        demand_val = demands[node_idx]

        if node_type == "linehaul":
            current_linehaul_demand_sum += demand_val
            if current_linehaul_demand_sum > vehicle_capacity + 1e-6: # Thêm epsilon để tránh lỗi float
                return False
        elif node_type == "backhaul":
            current_backhaul_demand_sum += abs(demand_val)
            if current_backhaul_demand_sum > vehicle_capacity + 1e-6:
                return False
    
    return True


def apply_2_opt_on_tour(tour_nodes, coords, instance_data, problem_type):
    """
    Áp dụng thuật toán 2-opt để cải thiện một lộ trình đơn lẻ,
    có kiểm tra ràng buộc nếu problem_type là "traditional".
    """
    if not tour_nodes or len(tour_nodes) <= 3: # Cần ít nhất 2 khách hàng (4 nút)
        return tour_nodes

    current_tour = list(tour_nodes)
    num_nodes_in_tour = len(current_tour)
    
    # Chỉ áp dụng kiểm tra ràng buộc nếu problem_type là "traditional"
    needs_constraint_check = (problem_type == "traditional")

    improved = True
    while improved:
        improved = False
        
        for i in range(num_nodes_in_tour - 3): # i từ 0 (depot) đến num_nodes_in_tour - 4
            for j in range(i + 2, num_nodes_in_tour - 1): # j từ i+2 đến num_nodes_in_tour - 2
                # Tạo tour mới sau khi hoán đổi
                new_segment = current_tour[i+1 : j+1]
                new_segment.reverse()
                temp_new_tour = current_tour[:i+1] + new_segment + current_tour[j+1:]
                
                # Tính toán quãng đường đầy đủ của tour mới
                new_full_dist = 0
                for k in range(len(temp_new_tour) - 1):
                    new_full_dist += calculate_distance(coords[temp_new_tour[k]], coords[temp_new_tour[k+1]])

                # Tính toán quãng đường đầy đủ của tour hiện tại
                current_full_dist = 0
                for k in range(len(current_tour) - 1):
                    current_full_dist += calculate_distance(coords[current_tour[k]], coords[current_tour[k+1]])

                if new_full_dist < current_full_dist - 1e-9: # Cải thiện đáng kể
                    is_valid_move = True
                    if needs_constraint_check:
                        # KIỂM TRA RÀNG BUỘC TRUYỀN THỐNG TRƯỚC KHI CHẤP NHẬN HOÁN ĐỔI
                        if not is_traditional_tour_valid(temp_new_tour, instance_data):
                            is_valid_move = False
                    
                    if is_valid_move:
                        current_tour = temp_new_tour
                        improved = True
                        break # Tìm thấy cải thiện, thoát j và i để bắt đầu lại while
            if improved:
                break
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
    
    # Thêm tour cuối cùng nếu còn khách hàng
    if len(current_tour_nodes) > 1:
        current_tour_nodes.append(depot_idx)
        tours.append(list(current_tour_nodes))

    # --- Áp dụng Tìm kiếm Cục bộ (2-opt) ---
    if apply_local_search:
        optimized_tours = []
        for tour in tours: # Dòng này định nghĩa 'tour' trong vòng lặp
            if not tour or len(tour) < 2:
                continue # Bỏ qua các tour rỗng hoặc quá ngắn không có khách hàng

            if len(tour) >= 4:
                # TRUYỀN ĐẦY ĐỦ THAM SỐ instance_data VÀ problem_type VÀO HÀM apply_2_opt_on_tour
                optimized_tours.append(apply_2_opt_on_tour(tour, coords, instance_data, problem_type))
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