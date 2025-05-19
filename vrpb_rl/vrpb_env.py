# vrpb_rl_simple_pg/vrpb_env.py
import numpy as np
import copy
from utils import calculate_distance, get_node_type_from_index

class VRPBEnv:
    def __init__(self, instance_data, problem_type="traditional"):
        self.coords = np.array(instance_data["coords"])
        self.demands = np.array(instance_data["demands"])
        self.num_total_customers = instance_data["num_customers"] 
        self.num_linehaul_customers = instance_data["num_linehaul"] 
        self.num_nodes = instance_data["num_nodes"] 
        self.vehicle_capacity = instance_data["vehicle_capacity"]
        self.problem_type = problem_type

        self.depot_idx = 0
        self.action_space_size = self.num_nodes
        
        self.global_visited_mask = np.zeros(self.num_nodes, dtype=bool)
        self.current_location_idx = self.depot_idx
        self.current_linehaul_payload = 0 
        self.current_backhaul_collected = 0 
        self.visited_in_current_tour_mask = np.zeros(self.num_nodes, dtype=bool)
        self.linehauls_served_in_tour_flag = False 
        self.current_tour_plan = [] 
        self.total_distance_travelled_in_tour = 0.0
        self.reset(full_reset=True)

    def _get_state_features(self):
        features = []
        features.append(self.current_location_idx / self.num_nodes)
        features.append(self.current_linehaul_payload / (self.vehicle_capacity + 1e-9))
        features.append(self.current_backhaul_collected / (self.vehicle_capacity + 1e-9))
        
        unvisited_linehaul_count = 0
        for i in range(1, self.num_linehaul_customers + 1):
            if not self.global_visited_mask[i]:
                unvisited_linehaul_count += 1
        features.append(unvisited_linehaul_count / (self.num_linehaul_customers + 1e-6))

        num_actual_backhaul_customers = self.num_total_customers - self.num_linehaul_customers
        unvisited_backhaul_count = 0
        if num_actual_backhaul_customers > 0:
            for i in range(self.num_linehaul_customers + 1, self.num_nodes):
                if not self.global_visited_mask[i]:
                    unvisited_backhaul_count += 1
            features.append(unvisited_backhaul_count / (num_actual_backhaul_customers + 1e-6))
        else:
            features.append(0.0)
        
        if self.problem_type == "traditional":
            features.append(1.0 if self.linehauls_served_in_tour_flag else 0.0)
        else:
            features.append(0.0) 
        
        return np.array(features, dtype=np.float32)

    def get_state_dim(self):
        return 6

    def reset(self, full_reset=True):
        if full_reset:
            self.global_visited_mask = np.zeros(self.num_nodes, dtype=bool)
            self.global_visited_mask[self.depot_idx] = True 
        
        self.current_location_idx = self.depot_idx
        self.current_linehaul_payload = 0 
        self.current_backhaul_collected = 0 
        self.visited_in_current_tour_mask = np.zeros(self.num_nodes, dtype=bool)
        self.visited_in_current_tour_mask[self.depot_idx] = True
        self.linehauls_served_in_tour_flag = False
        self.current_tour_plan = [self.depot_idx]
        self.total_distance_travelled_in_tour = 0.0
        return self._get_state_features()

    def _get_valid_actions_mask(self):
        valid_actions = np.zeros(self.num_nodes, dtype=bool)
        all_customers_served_globally = np.all(self.global_visited_mask[1:])

        # Trường hợp đặc biệt: Nếu tất cả khách hàng đã được phục vụ và xe không ở depot
        if all_customers_served_globally and self.current_location_idx != self.depot_idx:
            valid_actions[self.depot_idx] = True # Hành động duy nhất hợp lệ là quay về depot
            return valid_actions

        # Trường hợp đặc biệt: Nếu tất cả khách hàng đã được phục vụ VÀ xe ĐANG ở depot
        if all_customers_served_globally and self.current_location_idx == self.depot_idx:
            # Không còn hành động nào hợp lệ khác (episode nên kết thúc)
            return valid_actions # Trả về mảng toàn False

        for next_node_idx in range(self.num_nodes):
            if next_node_idx == self.depot_idx:
                # Cho phép quay về depot nếu tour không rỗng (đã có KH trong kế hoạch)
                # Hoặc nếu không còn lựa chọn nào khác cho khách hàng
                can_return_to_depot = False
                if len(self.current_tour_plan) > 1:
                    can_return_to_depot = True
                else: # Tour rỗng, kiểm tra xem có KH nào đi được không
                    has_other_valid_customer_move = False
                    for temp_idx in range(1, self.num_nodes): # Kiểm tra các KH khác
                        if not self.global_visited_mask[temp_idx]: # Nếu có KH chưa thăm
                            # (Cần kiểm tra thêm các ràng buộc khác cho temp_idx ở đây nếu muốn chính xác tuyệt đối,
                            # nhưng để đơn giản, chỉ cần có KH chưa thăm là có thể chưa cần về depot)
                            has_other_valid_customer_move = True
                            break
                    if not has_other_valid_customer_move: # Nếu không còn KH nào khác để đi
                        can_return_to_depot = True # Có thể về depot (dù tour rỗng)

                if can_return_to_depot:
                    valid_actions[next_node_idx] = True
                continue

            if self.global_visited_mask[next_node_idx]:
                continue

            demand_at_next_node = self.demands[next_node_idx]
            node_type = get_node_type_from_index(next_node_idx, self.num_linehaul_customers)

            temp_linehaul_payload = self.current_linehaul_payload
            temp_backhaul_collected = self.current_backhaul_collected
            can_serve_capacity = True

            if node_type == "linehaul":
                projected_lh_needed = temp_linehaul_payload + demand_at_next_node
                if projected_lh_needed > self.vehicle_capacity:
                    can_serve_capacity = False
                if self.problem_type == "traditional" and self.current_backhaul_collected > 0:
                     can_serve_capacity = False
            elif node_type == "backhaul":
                projected_bh_collected = temp_backhaul_collected + abs(demand_at_next_node)
                if projected_bh_collected > self.vehicle_capacity:
                    can_serve_capacity = False
            
            if not can_serve_capacity:
                continue

            can_serve_sequence = True
            if self.problem_type == "traditional":
                linehauls_in_plan_not_yet_visited = False
                for planned_node_idx in self.current_tour_plan:
                    if planned_node_idx != self.depot_idx and \
                       get_node_type_from_index(planned_node_idx, self.num_linehaul_customers) == "linehaul" and \
                       not self.visited_in_current_tour_mask[planned_node_idx]:
                        linehauls_in_plan_not_yet_visited = True
                        break
                
                if node_type == "linehaul":
                    if self.current_backhaul_collected > 0 :
                         can_serve_sequence = False
                elif node_type == "backhaul":
                    if self.current_linehaul_payload > 0 and linehauls_in_plan_not_yet_visited:
                        can_serve_sequence = False
            
            if not can_serve_sequence:
                continue
            
            valid_actions[next_node_idx] = True
        
        return valid_actions

    def step(self, action_node_idx):
        current_valid_mask = self._get_valid_actions_mask()
        if not current_valid_mask[action_node_idx]:
            # print(f"!!! ENV: Hành động không hợp lệ được thực hiện: {action_node_idx} từ {self.current_location_idx} !!!")
            return self._get_state_features(), -500, True, {"error": "Invalid action chosen by agent", "done_tour": True, "current_tour_nodes": list(self.current_tour_plan), "current_tour_distance": self.total_distance_travelled_in_tour}

        distance = calculate_distance(self.coords[self.current_location_idx], self.coords[action_node_idx])
        self.total_distance_travelled_in_tour += distance
        reward = -distance 

        previous_location_idx = self.current_location_idx # Lưu lại vị trí trước khi cập nhật
        self.current_location_idx = action_node_idx
        
        # Chỉ thêm vào current_tour_plan nếu nó khác với điểm cuối cùng (tránh depot -> depot ở đầu)
        # Hoặc nếu current_tour_plan rỗng (không nên xảy ra vì reset đã thêm depot)
        if not self.current_tour_plan or self.current_tour_plan[-1] != action_node_idx:
            self.current_tour_plan.append(action_node_idx)
        elif self.current_tour_plan[-1] == action_node_idx and action_node_idx == self.depot_idx and len(self.current_tour_plan) == 1:
            # Trường hợp đặc biệt: tour chỉ có [0] và action là 0 -> không thêm lại
            pass
        else: # Các trường hợp khác (ví dụ, đi tới 1 KH rồi lại chọn chính KH đó - không nên hợp lệ)
             self.current_tour_plan.append(action_node_idx)


        done_episode = False 
        done_tour = False    

        if action_node_idx == self.depot_idx:
            done_tour = True # Hành động hiện tại là quay về depot -> tour này kết thúc
            # Episode chỉ kết thúc nếu TẤT CẢ KH đã được phục vụ VÀ xe đã về depot
            if np.all(self.global_visited_mask[1:]):
                done_episode = True
        else: # Đi đến một khách hàng
            self.global_visited_mask[action_node_idx] = True
            self.visited_in_current_tour_mask[action_node_idx] = True

            node_type = get_node_type_from_index(action_node_idx, self.num_linehaul_customers)
            demand = self.demands[action_node_idx]

            if node_type == "linehaul":
                self.current_linehaul_payload += demand
            elif node_type == "backhaul":
                self.current_backhaul_collected += abs(demand)
            
            if self.problem_type == "traditional":
                all_planned_lh_visited_in_tour = True
                has_planned_lh = False
                for node_idx_in_plan in self.current_tour_plan:
                    if node_idx_in_plan != self.depot_idx and \
                       get_node_type_from_index(node_idx_in_plan, self.num_linehaul_customers) == "linehaul":
                        has_planned_lh = True
                        if not self.visited_in_current_tour_mask[node_idx_in_plan]:
                            all_planned_lh_visited_in_tour = False
                            break
                if has_planned_lh and all_planned_lh_visited_in_tour:
                    self.linehauls_served_in_tour_flag = True
            
            # Kiểm tra lại điều kiện kết thúc episode sau khi phục vụ khách hàng
            # (Nếu đây là KH cuối cùng và bước tiếp theo BẮT BUỘC là về depot)
            if np.all(self.global_visited_mask[1:]):
                # Nếu tất cả KH đã xong, nhưng xe chưa ở depot, episode chưa kết thúc.
                # Agent phải học cách tự quay về depot.
                # _get_valid_actions_mask sẽ chỉ cho phép về depot.
                pass 
        
        # Ghi đè done_episode một lần cuối để đảm bảo tính chính xác tuyệt đối
        # Episode chỉ kết thúc khi tất cả khách hàng đã được phục vụ VÀ xe đang ở depot.
        all_customers_globally_served_final_check = np.all(self.global_visited_mask[1:])
        is_at_depot_final_check = (self.current_location_idx == self.depot_idx)

        if all_customers_globally_served_final_check and is_at_depot_final_check:
            done_episode = True
        else:
            done_episode = False
        
        # done_tour chỉ đơn giản là xe có về depot ở bước này không
        if self.current_location_idx == self.depot_idx:
            done_tour = True
        else:
            done_tour = False
            if done_episode: # Nếu episode xong mà xe không ở depot -> lỗi logic
                # print("LỖI LOGIC ENV: done_episode=True nhưng xe không ở depot!")
                done_episode = False # Sửa lại cho an toàn


        info = {
            "done_tour": done_tour, 
            "current_tour_distance": self.total_distance_travelled_in_tour, 
            "current_tour_nodes": list(self.current_tour_plan) # Trả về bản sao
        }
        return self._get_state_features(), reward, done_episode, info
