# vrpb_bnb/bnb_solver.py
import numpy as np
import heapq 
from utils import calculate_distance, get_node_type_from_index
from bnb_node import BnBNode

class VRPBBnBSolver:
    def __init__(self, instance_data, problem_type="traditional"):
        self.coords = np.array(instance_data["coords"])
        self.demands = np.array(instance_data["demands"])
        self.num_total_customers = instance_data["num_customers"]
        self.num_linehaul_customers = instance_data["num_linehaul"]
        self.num_nodes = instance_data["num_nodes"] 
        self.vehicle_capacity = instance_data["vehicle_capacity"]
        self.problem_type = problem_type # "traditional" hoặc "improved"
        self.depot_idx = 0

        self.best_solution_tour = None 
        self.upper_bound = float('inf') 
        self.nodes_processed_count = 0

    def _calculate_tour_distance(self, tour):
        dist = 0
        for i in range(len(tour) - 1):
            dist += calculate_distance(self.coords[tour[i]], self.coords[tour[i+1]])
        return dist

    def _get_lower_bound(self, current_node_bnb):
        # Cận dưới đơn giản: chi phí đã đi. Cần cải thiện để cắt tỉa tốt hơn.
        h_n = 0
        if np.all(current_node_bnb.visited_mask[1:]): 
            if current_node_bnb.tour[-1] != self.depot_idx:
                 h_n = calculate_distance(self.coords[current_node_bnb.tour[-1]], self.coords[self.depot_idx])
        return current_node_bnb.cost + h_n

    def solve(self, time_limit_seconds=None):
        self.best_solution_tour = None
        self.upper_bound = float('inf')
        self.nodes_processed_count = 0
        
        initial_visited_mask = np.zeros(self.num_nodes, dtype=bool)
        initial_visited_mask[self.depot_idx] = True
        initial_node = BnBNode(
            tour=[self.depot_idx], 
            cost=0.0, 
            visited_mask=initial_visited_mask,
            linehaul_load=0.0,
            backhaul_load=0.0,
            linehaul_phase_active=True # Cho traditional, ban đầu luôn là pha linehaul
        )

        priority_queue = []
        heapq.heappush(priority_queue, (self._get_lower_bound(initial_node), initial_node))

        import time
        start_time = time.time()

        while priority_queue:
            if time_limit_seconds is not None and (time.time() - start_time) > time_limit_seconds:
                print(f"BnB ({self.problem_type}): Đã đạt giới hạn thời gian {time_limit_seconds} giây.")
                break

            _, current_bnb_node = heapq.heappop(priority_queue)
            self.nodes_processed_count += 1

            if current_bnb_node.cost >= self.upper_bound:
                continue

            all_customers_served = np.all(current_bnb_node.visited_mask[1:])
            if all_customers_served:
                cost_to_return_to_depot = calculate_distance(
                    self.coords[current_bnb_node.tour[-1]], 
                    self.coords[self.depot_idx]
                )
                final_tour_cost = current_bnb_node.cost + cost_to_return_to_depot
                
                if final_tour_cost < self.upper_bound:
                    self.upper_bound = final_tour_cost
                    self.best_solution_tour = current_bnb_node.tour + [self.depot_idx]
                    # print(f"BnB ({self.problem_type}): Node {self.nodes_processed_count}. Giải pháp mới: Cost={self.upper_bound:.2f}")
                continue

            last_node_in_tour = current_bnb_node.tour[-1]

            for next_node_idx in range(1, self.num_nodes): 
                if not current_bnb_node.visited_mask[next_node_idx]:
                    can_visit_this_node = True
                    prospective_lh_load = current_bnb_node.linehaul_load
                    prospective_bh_load = current_bnb_node.backhaul_load
                    prospective_lh_phase = current_bnb_node.linehaul_phase_active # Quan trọng cho traditional
                    
                    demand_of_next_node = self.demands[next_node_idx]
                    type_of_next_node = get_node_type_from_index(next_node_idx, self.num_linehaul_customers)

                    # 1. Kiểm tra tải trọng (áp dụng cho cả hai loại VRPB)
                    if type_of_next_node == "linehaul":
                        if prospective_lh_load + demand_of_next_node > self.vehicle_capacity:
                            can_visit_this_node = False
                    elif type_of_next_node == "backhaul":
                        if prospective_bh_load + abs(demand_of_next_node) > self.vehicle_capacity:
                            can_visit_this_node = False
                    
                    if not can_visit_this_node: continue

                    # 2. Kiểm tra ràng buộc thứ tự (chủ yếu cho VRPB Truyền thống)
                    if self.problem_type == "traditional":
                        if type_of_next_node == "linehaul":
                            # Nếu đang KHÔNG ở pha linehaul (tức là đã có backhaul được chọn trước đó),
                            # thì không được chọn thêm linehaul.
                            if not prospective_lh_phase: 
                                can_visit_this_node = False
                        elif type_of_next_node == "backhaul":
                            # Nếu đang ở pha linehaul (prospective_lh_phase is True),
                            # chỉ được chọn backhaul nếu TẤT CẢ khách hàng linehaul CHƯA ĐƯỢC THĂM
                            # không còn khách hàng linehaul nào hợp lệ để đi nữa (hoặc không còn KH linehaul nào).
                            # Đây là một cách đơn giản hóa: nếu còn bất kỳ KH linehaul nào chưa thăm, ưu tiên chúng.
                            if prospective_lh_phase:
                                any_unvisited_linehaul_remains = False
                                for lh_idx_check in range(1, self.num_linehaul_customers + 1):
                                    if not current_bnb_node.visited_mask[lh_idx_check]:
                                        # (Cần kiểm tra sâu hơn xem KH linehaul này có thực sự đi được không,
                                        # nhưng để đơn giản, chỉ cần nó tồn tại là chưa nên đi backhaul)
                                        any_unvisited_linehaul_remains = True
                                        break
                                if any_unvisited_linehaul_remains:
                                    can_visit_this_node = False
                                # else: Nếu không còn KH linehaul nào, thì có thể bắt đầu backhaul (prospective_lh_phase sẽ chuyển False)
                    # Cho VRPB Cải tiến (improved), không có ràng buộc cứng nhắc về prospective_lh_phase.
                    # Việc tải trọng cho phép cả linehaul và backhaul cùng lúc cần logic phức tạp hơn về
                    # cách quản lý tổng tải trọng trên xe nếu dùng chung khoang.
                    # Logic hiện tại: prospective_lh_load và prospective_bh_load là tổng nhu cầu đã cam kết cho mỗi loại,
                    # và mỗi loại không được vượt capacity. Điều này giống như xe có 2 khoang riêng.
                    # Nếu xe chỉ có 1 khoang, thì phải là:
                    # (hàng linehaul đang trên xe để giao) + (hàng backhaul đã thu) <= capacity.
                    # Hiện tại, chúng ta giữ logic tải trọng đơn giản này cho cả hai.

                    if not can_visit_this_node: continue

                    new_cost_to_next_node = current_bnb_node.cost + calculate_distance(self.coords[last_node_in_tour], self.coords[next_node_idx])
                    
                    if new_cost_to_next_node >= self.upper_bound:
                        continue

                    new_tour_plan = current_bnb_node.tour + [next_node_idx]
                    new_visited_mask = current_bnb_node.visited_mask.copy()
                    new_visited_mask[next_node_idx] = True
                    
                    updated_lh_load = prospective_lh_load
                    updated_bh_load = prospective_bh_load
                    updated_lh_phase = prospective_lh_phase # Mặc định giữ nguyên

                    if type_of_next_node == "linehaul":
                        updated_lh_load += demand_of_next_node
                    elif type_of_next_node == "backhaul":
                        updated_bh_load += abs(demand_of_next_node)
                        # Trong traditional, khi chọn backhaul đầu tiên, pha linehaul kết thúc
                        if self.problem_type == "traditional" and prospective_lh_phase:
                            updated_lh_phase = False 
                    
                    child_node = BnBNode(
                        tour=new_tour_plan,
                        cost=new_cost_to_next_node,
                        visited_mask=new_visited_mask,
                        linehaul_load=updated_lh_load,
                        backhaul_load=updated_bh_load,
                        linehaul_phase_active=updated_lh_phase
                    )
                    heapq.heappush(priority_queue, (self._get_lower_bound(child_node), child_node))

        if self.best_solution_tour:
            final_distance = self._calculate_tour_distance(self.best_solution_tour)
            self.upper_bound = final_distance
            return self.upper_bound, [self.best_solution_tour]
        else:
            return float('inf'), []