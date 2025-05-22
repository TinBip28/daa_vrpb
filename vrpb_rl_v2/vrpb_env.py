import numpy as np
import copy
# Giả sử bạn có một tệp utils.py với các hàm này
# from utils import calculate_distance 
# Để mã này tự chạy được, tôi sẽ định nghĩa tạm calculate_distance ở đây
def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

class VRPBEnv_MathFormulation:
    DEBUG_MODE_CLASS_VAR = False 

    def __init__(self, instance_data, problem_type="traditional"):
        self.coords = np.array(instance_data["coords"])
        self.demands = np.array(instance_data["demands"]) # LH > 0, BH < 0
        self.num_total_customers = instance_data["num_customers"]
        self.num_linehaul_customers = instance_data["num_linehaul"] 
        self.num_nodes = instance_data["num_nodes"] 
        self.num_vehicles = instance_data["num_vehicles"]
        self.vehicle_capacity = instance_data["vehicle_capacity"]
        self.problem_type = problem_type 
        self.depot_idx = 0
        self.action_space_size = self.num_nodes

        self.global_visited_mask = np.zeros(self.num_nodes, dtype=bool)
        self.current_location_idx = self.depot_idx
        self.current_vehicle_load = 0.0 
        self.linehaul_phase_active_in_tour = True 
        
        self.visited_in_current_tour_mask = np.zeros(self.num_nodes, dtype=bool)
        self.current_tour_plan = []
        self.total_distance_travelled_in_tour = 0.0
        
        self.tour_number_debug = 0
        self.step_number_debug = 0

        self.reset(full_reset=True)

    def get_node_type_from_idx(self, node_idx):
        if node_idx == self.depot_idx:
            return "depot"
        if node_idx > 0 and node_idx <= self.num_linehaul_customers:
            return "linehaul"
        return "backhaul"

    def get_state_dim(self):
        return 5 

    def reset(self, full_reset=True):
        if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR:
            if full_reset: self.tour_number_debug = 1 
            else: self.tour_number_debug +=1 # Tăng số tour nếu không phải full reset (đếm số tour trong episode)
            self.step_number_debug = 0
            print(f"\n{'='*10} ENV RESET (MathFormulation) {'='*10}")
            print(f"Full Reset: {full_reset}, Tour Number (in episode): {self.tour_number_debug}, ProblemType: {self.problem_type}")

        if full_reset:
            self.global_visited_mask = np.zeros(self.num_nodes, dtype=bool)
            self.global_visited_mask[self.depot_idx] = True

        self.current_location_idx = self.depot_idx
        self.visited_in_current_tour_mask = np.zeros(self.num_nodes, dtype=bool)
        self.visited_in_current_tour_mask[self.depot_idx] = True
        self.current_tour_plan = [self.depot_idx]
        self.total_distance_travelled_in_tour = 0.0

        # --- LOGIC KHỞI TẠO TẢI TRỌNG ĐƯỢC LÀM RÕ RÀNG ---
        if full_reset: 
            # Bắt đầu một EPISODE mới: D0 = Capacity (xe đầy hàng linehaul) cho cả hai loại
            self.current_vehicle_load = self.vehicle_capacity
            self.linehaul_phase_active_in_tour = True
            if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR: 
                print(f"Full Episode Reset: Load set to capacity ({self.vehicle_capacity:.2f}). LH_active=True for first tour.")
        else: # Bắt đầu một TOUR MỚI trong cùng một episode (full_reset=False)
            if self.problem_type == "traditional":
                # VRPB Truyền thống: Mỗi tour mới từ depot cũng bắt đầu đầy tải linehaul
                self.current_vehicle_load = self.vehicle_capacity
                self.linehaul_phase_active_in_tour = True
                if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR: print(f"TraditionalVRPB New Tour Reset: Load set to capacity ({self.vehicle_capacity:.2f}). LH_active=True.")
            elif self.problem_type == "improved":
                # VRPB Cải tiến: Tour mới từ depot có thể bắt đầu RỖNG để linh hoạt
                self.current_vehicle_load = 0.0 
                self.linehaul_phase_active_in_tour = True # Vẫn cho phép chọn LH trước nếu muốn
                if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR: print(f"ImprovedVRPB New Tour Reset: Load set to 0. LH_active=True (can choose LH or BH).")
        
        if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR:
            print(f"ResetState End: Loc={self.current_location_idx}, Load={self.current_vehicle_load:.2f}, "
                  f"LH_active={self.linehaul_phase_active_in_tour}")
            # print(f"GlobalVisited: {self.global_visited_mask.astype(int)}") # In ở đầu reset nếu cần
        return self._get_state_features()

    def _get_state_features(self):
        features = []
        features.append(self.current_location_idx / (self.num_nodes - 1 + 1e-9) ) 
        features.append(self.current_vehicle_load / (self.vehicle_capacity + 1e-9))
        unvisited_linehaul_count = 0
        if self.num_linehaul_customers > 0:
            for i in range(1, self.num_linehaul_customers + 1):
                if not self.global_visited_mask[i]:
                    unvisited_linehaul_count += 1
            features.append(unvisited_linehaul_count / (self.num_linehaul_customers + 1e-6))
        else:
            features.append(0.0)
        num_actual_backhaul_customers = self.num_total_customers - self.num_linehaul_customers
        unvisited_backhaul_count = 0
        if num_actual_backhaul_customers > 0:
            for i in range(self.num_linehaul_customers + 1, self.num_total_customers + 1):
                if not self.global_visited_mask[i]:
                    unvisited_backhaul_count += 1
            features.append(unvisited_backhaul_count / (num_actual_backhaul_customers + 1e-6))
        else:
            features.append(0.0)
        if self.problem_type == "traditional":
            features.append(1.0 if self.linehaul_phase_active_in_tour else 0.0)
        else: 
            features.append(0.0) 
        current_dim = self.get_state_dim()
        if len(features) < current_dim: features.extend([0.0] * (current_dim - len(features)))
        elif len(features) > current_dim: features = features[:current_dim]
        return np.array(features, dtype=np.float32)

    def _get_valid_actions_mask(self):
        # ... (Giữ nguyên logic của _get_valid_actions_mask từ phiên bản trước,
        #      nó đã bao gồm các kiểm tra cho is_at_depot_start_tour và problem_type) ...
        if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR:
            print(f"--- Mask Gen (Tour {self.tour_number_debug}, Step {self.step_number_debug}): "
                  f"CurrentLoc={self.current_location_idx}, Load={self.current_vehicle_load:.2f}, "
                  f"LHActive={self.linehaul_phase_active_in_tour}, TourPlanLen={len(self.current_tour_plan)}")
        valid_actions = np.zeros(self.num_nodes, dtype=bool)
        all_customers_served_globally = np.all(self.global_visited_mask[1:])
        if all_customers_served_globally:
            if self.current_location_idx != self.depot_idx: valid_actions[self.depot_idx] = True
            return valid_actions
        still_unvisited_linehauls_globally = False
        is_at_depot_start_tour = (self.current_location_idx == self.depot_idx and len(self.current_tour_plan) == 1)
        if self.problem_type == "traditional" and is_at_depot_start_tour:
            for i in range(1, self.num_linehaul_customers + 1):
                if not self.global_visited_mask[i]:
                    still_unvisited_linehauls_globally = True; break
        for next_node_idx in range(1, self.num_nodes): 
            node_debug_prefix = f"MaskCust {next_node_idx}:"
            can_serve_this_node = True 
            if self.global_visited_mask[next_node_idx]:
                # if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR: print(f"{node_debug_prefix} Masked: Globally visited.")
                continue 
            demand_at_next_node = self.demands[next_node_idx] 
            node_type = self.get_node_type_from_idx(next_node_idx)
            if self.problem_type == "traditional" and is_at_depot_start_tour and \
               node_type == "backhaul" and still_unvisited_linehauls_globally:
                can_serve_this_node = False
            if not can_serve_this_node: continue
            if node_type == "linehaul":
                if self.problem_type == "improved" and is_at_depot_start_tour:
                    if demand_at_next_node > self.vehicle_capacity + 1e-6: 
                        can_serve_this_node = False
                else: 
                    if demand_at_next_node > self.current_vehicle_load + 1e-6: 
                        can_serve_this_node = False
            elif node_type == "backhaul":
                if self.problem_type == "improved" and is_at_depot_start_tour:
                    if abs(demand_at_next_node) > self.vehicle_capacity + 1e-6:
                        can_serve_this_node = False
                else: 
                    if (self.current_vehicle_load - demand_at_next_node) > self.vehicle_capacity + 1e-6:
                        can_serve_this_node = False
            if not can_serve_this_node: continue
            if self.problem_type == "traditional":
                if node_type == "linehaul" and not self.linehaul_phase_active_in_tour:
                    can_serve_this_node = False
            if not can_serve_this_node: continue
            valid_actions[next_node_idx] = True
        can_move_to_any_customer = np.any(valid_actions[1:])
        if self.current_location_idx == self.depot_idx: 
            if not can_move_to_any_customer and not all_customers_served_globally:
                valid_actions[self.depot_idx] = True
        else: 
            if not can_move_to_any_customer: 
                valid_actions[self.depot_idx] = True 
            elif len(self.current_tour_plan) > 1: 
                valid_actions[self.depot_idx] = True
        if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR: print(f"--- Mask Gen End (MathFormulation). Final Valid Actions: {valid_actions.astype(int)}")
        # if not np.any(valid_actions) and not all_customers_served_globally:
        #      if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR: print(f"WARNING (Mask): No valid actions found but not all customers served! ...")
        return valid_actions


    def step(self, action_node_idx):
        # ... (Giữ nguyên logic của step từ phiên bản trước,
        #      nó đã bao gồm các kiểm tra cho is_first_customer_in_tour và problem_type) ...
        self.step_number_debug +=1
        if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR:
            print(f"\n>>> ENV STEP (Tour {self.tour_number_debug}, Step {self.step_number_debug}): Action={action_node_idx}")
            # print(f"Before Step: Loc={self.current_location_idx}, Load={self.current_vehicle_load:.2f}, LHActive={self.linehaul_phase_active_in_tour}, TourPlanLength={len(self.current_tour_plan)}, DistInTour={self.total_distance_travelled_in_tour:.2f}")

        info = {"done_tour": False, "current_tour_distance": self.total_distance_travelled_in_tour,
                "current_tour_nodes": list(self.current_tour_plan), "error": None}
        
        # An toàn hơn nếu kiểm tra lại valid_mask ở đây, mặc dù agent nên chọn từ mask hợp lệ
        current_valid_mask_check = self._get_valid_actions_mask()
        if not current_valid_mask_check[action_node_idx]:
            info["error"] = f"Invalid action {action_node_idx} chosen by agent (detected in step, valid: {current_valid_mask_check.astype(int)})"
            if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR: print(f"<<< ENV STEP END (ERROR): {info['error']}")
            return self._get_state_features(), -1000.0, True, info # Phạt cực nặng và kết thúc episode

        distance = calculate_distance(self.coords[self.current_location_idx], self.coords[action_node_idx])
        self.total_distance_travelled_in_tour += distance
        reward_dist = -distance 
        previous_location_idx = self.current_location_idx
        self.current_location_idx = action_node_idx
        is_first_customer_in_tour = (len(self.current_tour_plan) == 1 and action_node_idx != self.depot_idx and previous_location_idx == self.depot_idx)
        if not self.current_tour_plan or self.current_tour_plan[-1] != action_node_idx:
            self.current_tour_plan.append(action_node_idx)
        elif self.current_tour_plan[-1] == action_node_idx and action_node_idx == self.depot_idx and len(self.current_tour_plan) == 1:
             pass
        done_episode = False 
        done_tour = False    
        if action_node_idx == self.depot_idx: 
            done_tour = True
            if np.all(self.global_visited_mask[1:]): done_episode = True
        else: 
            self.global_visited_mask[action_node_idx] = True
            self.visited_in_current_tour_mask[action_node_idx] = True 
            demand = self.demands[action_node_idx] 
            node_type = self.get_node_type_from_idx(action_node_idx)
            load_before_service = self.current_vehicle_load
            if self.problem_type == "improved" and is_first_customer_in_tour:
                if node_type == "linehaul":
                    self.current_vehicle_load = demand 
                    if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR: print(f"ImprovedVRPB FirstCust (LH): Loaded {demand:.1f}. New Load = {self.current_vehicle_load:.1f}")
                elif node_type == "backhaul":
                    self.current_vehicle_load = -demand 
                    if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR: print(f"ImprovedVRPB FirstCust (BH): Picked up {-demand:.1f}. New Load = {self.current_vehicle_load:.1f}")
            else: 
                self.current_vehicle_load -= demand 
            if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR and not (self.problem_type == "improved" and is_first_customer_in_tour) : 
                print(f"Step: Serviced {node_type} node {action_node_idx} (Demand: {demand:.1f}). Load: {load_before_service:.1f} -> {self.current_vehicle_load:.1f}")
            if self.problem_type == "traditional":
                if node_type == "backhaul" and self.linehaul_phase_active_in_tour:
                    self.linehaul_phase_active_in_tour = False
                    # if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR: print(f"Trad. VRPB: LH phase ended due to BH pickup at {action_node_idx}.")
                if self.current_vehicle_load < 1e-6 and self.linehaul_phase_active_in_tour: 
                    self.linehaul_phase_active_in_tour = False
                    # if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR: print(f"Trad. VRPB: LH phase auto-ended (vehicle empty after LH).")
            if np.all(self.global_visited_mask[1:]): pass 
        all_globally_served = np.all(self.global_visited_mask[1:])
        at_depot = (self.current_location_idx == self.depot_idx)
        if all_globally_served and at_depot: done_episode = True
        if at_depot and len(self.current_tour_plan) > 1 : done_tour = True                                         
        info.update({"done_tour": done_tour, "current_tour_distance": self.total_distance_travelled_in_tour,
                     "current_tour_nodes": list(self.current_tour_plan)})
        if VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR:
            # print(f"<<< ENV STEP END: NextStateFeatures={self._get_state_features()}, RewardDist={reward_dist:.2f}, DoneEp={done_episode}, DoneTour={done_tour}")
            # if info["error"]: print(f"Error in step: {info['error']}")
            pass
        return self._get_state_features(), reward_dist, done_episode, info

