import numpy as np
import torch
import os
import copy 
from vrpb_env import VRPBEnv_MathFormulation 
from data_loader import load_instance_from_csv    
from reinforce_agent import REINFORCEAgent
from policy_network import SimplePolicyNetwork 
try:
    from utils import visualize_vrpb_solution, calculate_distance
except ImportError:
    print("Cảnh báo: Không tìm thấy utils.py, một số chức năng có thể không hoạt động (ví dụ: visualize).")
    def visualize_vrpb_solution(instance_data, all_tours_nodes, total_distance, title="VRPB Solution"):
        print(f"Visualize (Placeholder): {title}, Distance: {total_distance}, Tours: {all_tours_nodes}")
    def calculate_distance(coord1, coord2):
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

DEBUG_SCRIPT_MODE = False 

def calculate_single_tour_distance(tour_nodes, coords):
    dist = 0
    for i in range(len(tour_nodes) - 1):
        dist += calculate_distance(coords[tour_nodes[i]], coords[tour_nodes[i+1]])
    return dist

def get_node_type_from_idx_local(node_idx, num_linehaul_customers, depot_idx=0):
    """Hàm cục bộ để lấy loại nút, tương tự như trong VRPBEnv."""
    if node_idx == depot_idx:
        return "depot"
    if node_idx > 0 and node_idx <= num_linehaul_customers:
        return "linehaul"
    return "backhaul"

def is_traditional_tour_globally_valid(tour_nodes, instance_data):
    """
    Kiểm tra xem một lộ trình đơn lẻ có tuân thủ các quy tắc của VRPB Truyền thống không.
    1. Tất cả linehaul phải đến trước tất cả backhaul.
    2. Tổng demand của đoạn linehaul không vượt quá dung lượng.
    3. Tổng demand (absolute) của đoạn backhaul không vượt quá dung lượng.
    Lộ trình tour_nodes bao gồm depot ở đầu và cuối.
    """
    if not tour_nodes or len(tour_nodes) < 2: # Tour rỗng hoặc không hợp lệ
        return True # Coi như hợp lệ nếu không có gì để kiểm tra

    coords = instance_data["coords"]
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
        node_type = get_node_type_from_idx_local(node_idx, num_linehaul_customers, depot_idx)
        if node_type == "linehaul":
            if linehaul_phase_ended:
                if DEBUG_SCRIPT_MODE: print(f"  [2-Opt Constraint Check] Invalid Order: LH {node_idx} after BH phase in tour {tour_nodes}")
                return False # Lỗi: Linehaul sau khi đã có Backhaul
        elif node_type == "backhaul":
            linehaul_phase_ended = True

    # 2. Kiểm tra dung lượng cho đoạn Linehaul và Backhaul riêng biệt
    current_linehaul_demand_sum = 0
    current_backhaul_demand_sum = 0
    in_backhaul_segment = False

    for node_idx in customer_nodes_in_tour:
        node_type = get_node_type_from_idx_local(node_idx, num_linehaul_customers, depot_idx)
        demand_val = demands[node_idx]

        if node_type == "linehaul":
            current_linehaul_demand_sum += demand_val
            if current_linehaul_demand_sum > vehicle_capacity + 1e-6:
                if DEBUG_SCRIPT_MODE: print(f"  [2-Opt Constraint Check] LH Capacity Exceeded: Sum={current_linehaul_demand_sum} > Cap={vehicle_capacity} in tour {tour_nodes}")
                return False
        elif node_type == "backhaul":
            if not in_backhaul_segment: # Chuyển sang đoạn backhaul
                in_backhaul_segment = True
                # Tại thời điểm này, current_linehaul_demand_sum là tổng demand của tất cả LH trong tour.
            
            current_backhaul_demand_sum += abs(demand_val) # Demand của backhaul là âm
            if current_backhaul_demand_sum > vehicle_capacity + 1e-6:
                if DEBUG_SCRIPT_MODE: print(f"  [2-Opt Constraint Check] BH Capacity Exceeded: Sum={current_backhaul_demand_sum} > Cap={vehicle_capacity} in tour {tour_nodes}")
                return False
    
    return True


def apply_2_opt_to_single_tour(original_tour_nodes, instance_data, problem_type):
    """
    Áp dụng thuật toán 2-Opt, có kiểm tra ràng buộc cho VRPB Truyền thống.
    """
    coords = instance_data["coords"] # Lấy coords từ instance_data

    if not original_tour_nodes or len(original_tour_nodes) < 4: 
        return original_tour_nodes

    current_best_tour = copy.deepcopy(original_tour_nodes) 
    current_best_distance = calculate_single_tour_distance(current_best_tour, coords)
    num_nodes_in_tour = len(current_best_tour)
    
    improved = True
    while improved:
        improved = False
        for i in range(1, num_nodes_in_tour - 2): 
            for j in range(i + 1, num_nodes_in_tour - 1): 
                new_tour = current_best_tour[:i] + current_best_tour[i:j+1][::-1] + current_best_tour[j+1:]
                new_distance = calculate_single_tour_distance(new_tour, coords)

                if new_distance < current_best_distance - 1e-9:
                    is_valid_move = True
                    if problem_type == "traditional":
                        if not is_traditional_tour_globally_valid(new_tour, instance_data):
                            is_valid_move = False
                            if DEBUG_SCRIPT_MODE: print(f"    [2-Opt] Move from {current_best_tour} to {new_tour} (dist {new_distance:.2f}) REJECTED due to traditional constraint violation.")
                    
                    if is_valid_move:
                        if DEBUG_SCRIPT_MODE: print(f"    [2-Opt] Move Accepted: Dist {current_best_distance:.2f} -> {new_distance:.2f}. New tour: {new_tour}")
                        current_best_tour = new_tour
                        current_best_distance = new_distance
                        improved = True
                        break 
            if improved:
                break 
    
    # if DEBUG_SCRIPT_MODE and calculate_single_tour_distance(original_tour_nodes, coords) > current_best_distance + 1e-5 :
    #     pass 
    return current_best_tour

def solve_one_instance_with_trained_agent(env_instance, agent, max_tours_per_instance=None, max_steps_per_tour=None):
    if DEBUG_SCRIPT_MODE: print(f"\n[Eval Agent] Bắt đầu đánh giá instance (Problem: {env_instance.problem_type})...")
    agent.policy_network.eval() 
    all_generated_tours_segments = []

    # Lấy số xe từ instance_data nếu có, nếu không thì dựa vào số khách hàng
    # Giả sử env_instance.instance_data tồn tại và chứa 'max_vehicles'
    # Cần đảm bảo env_instance có thuộc tính instance_data được gán từ script huấn luyện
    num_vehicles_available = env_instance.num_total_customers # Giá trị mặc định
    if hasattr(env_instance, 'instance_data') and 'max_vehicles' in env_instance.instance_data:
        num_vehicles_available = env_instance.instance_data['max_vehicles']
        if DEBUG_SCRIPT_MODE: print(f"[Eval Agent] Using max_vehicles from instance_data: {num_vehicles_available}")


    if max_tours_per_instance is None:
        max_tours_per_instance = num_vehicles_available + 3 
    if max_steps_per_tour is None:
        max_steps_per_tour = env_instance.num_nodes + 10

    current_state = env_instance.reset(full_reset=True)
    num_tours_generated = 0
    
    with torch.no_grad(): 
        while not np.all(env_instance.global_visited_mask[1:]) and num_tours_generated < max_tours_per_instance:
            if num_tours_generated > 0: 
                current_state = env_instance.reset(full_reset=False) 
            
            if DEBUG_SCRIPT_MODE: print(f"[Eval Agent] Bắt đầu Tour {num_tours_generated + 1}")

            info_last_step_of_tour = {
                "current_tour_nodes": list(env_instance.current_tour_plan),
                "current_tour_distance": env_instance.total_distance_travelled_in_tour,
                "done_tour": False,
                "error": None
            }
            temp_done_episode_globally_this_tour = False

            for step_in_tour in range(max_steps_per_tour):
                valid_actions_mask = env_instance._get_valid_actions_mask()
                if DEBUG_SCRIPT_MODE and step_in_tour < 3 : 
                    print(f"  [Eval Agent] Tour {num_tours_generated + 1}, Step {step_in_tour}: ValidMask={valid_actions_mask.astype(int)}")

                if not np.any(valid_actions_mask):
                    info_last_step_of_tour["done_tour"] = (env_instance.current_location_idx == env_instance.depot_idx)
                    info_last_step_of_tour["error"] = "Agent stuck (no valid actions in eval)"
                    if DEBUG_SCRIPT_MODE: print(f"  [Eval Agent] Tour {num_tours_generated+1}, Step {step_in_tour}: Agent bị kẹt. Vị trí hiện tại={env_instance.current_location_idx}. GlobalVisited={env_instance.global_visited_mask.astype(int)}")
                    break

                action, _ = agent.select_action(current_state, valid_actions_mask)
                if DEBUG_SCRIPT_MODE and step_in_tour < 3:
                     print(f"  [Eval Agent] Tour {num_tours_generated + 1}, Step {step_in_tour}: Hành động được chọn={action}")

                next_state, _, temp_done_episode_globally_this_tour, info_step = env_instance.step(action)
                info_last_step_of_tour = info_step 
                current_state = next_state

                if info_last_step_of_tour.get("error"):
                    if DEBUG_SCRIPT_MODE: print(f"  [Eval Agent] Lỗi từ env.step() trong Tour {num_tours_generated+1}, Step {step_in_tour}: {info_last_step_of_tour['error']}. Hủy đánh giá.")
                    agent.policy_network.train() 
                    return float('inf'), [] 
                
                if info_last_step_of_tour.get("done_tour", False): 
                    if DEBUG_SCRIPT_MODE: print(f"  [Eval Agent] Tour {num_tours_generated+1} kết thúc (done_tour=True). Kế hoạch tour: {info_last_step_of_tour.get('current_tour_nodes')}")
                    break
                if temp_done_episode_globally_this_tour: 
                    if DEBUG_SCRIPT_MODE: print(f"  [Eval Agent] Episode kết thúc toàn cục trong Tour {num_tours_generated+1}.")
                    break
            
            tour_segment_from_env = list(info_last_step_of_tour.get("current_tour_nodes", []))
            if tour_segment_from_env and len(tour_segment_from_env) > 1:
                 all_generated_tours_segments.append(tour_segment_from_env)

            num_tours_generated += 1
            if temp_done_episode_globally_this_tour: break 
            if not np.all(env_instance.global_visited_mask[1:]) and num_tours_generated >= max_tours_per_instance:
                if DEBUG_SCRIPT_MODE: print(f"[Eval Agent] Đã đạt số tour tối đa ({max_tours_per_instance}), nhưng chưa phục vụ hết khách hàng.")
                break
    
    agent.policy_network.train() 

    if DEBUG_SCRIPT_MODE:
        print(f"\n[Eval Agent] Kết thúc quá trình tạo tour. Tổng số tour được tạo: {len(all_generated_tours_segments)}")

    all_customers_finally_served = np.all(env_instance.global_visited_mask[1:])
    if not all_customers_finally_served:
        if DEBUG_SCRIPT_MODE: print("[Eval Agent] KIỂM TRA THẤT BẠI: Không phải tất cả khách hàng đã được phục vụ vào cuối quá trình đánh giá.")
        return float('inf'), []

    if not all_generated_tours_segments: 
        if all_customers_finally_served and env_instance.num_total_customers == 0: 
             if DEBUG_SCRIPT_MODE: print("[Eval Agent] KIỂM TRA OK: Không có khách hàng, không cần tour.")
             return 0.0, [[env_instance.depot_idx, env_instance.depot_idx]] 
        if DEBUG_SCRIPT_MODE: print("[Eval Agent] KIỂM TRA THẤT BẠI: Tất cả khách hàng đã được phục vụ (hoặc không có KH) nhưng không có đoạn tour nào được tạo/giữ lại.")
        return float('inf'), []

    valid_and_completed_tours_for_output = []
    recalculated_total_distance = 0.0
    if DEBUG_SCRIPT_MODE: print("\n[Eval Agent] Kiểm tra tính hợp lệ của từng đoạn tour...")
    for i_tour, tour_segment in enumerate(all_generated_tours_segments):
        if DEBUG_SCRIPT_MODE and len(tour_segment) > 10 : print(f"  Kiểm tra đoạn tour {i_tour+1} (dài, >10 nút)...")
        # elif DEBUG_SCRIPT_MODE: print(f"  Kiểm tra đoạn tour {i_tour+1}: {tour_segment}") # Bỏ bớt log
        
        if not tour_segment or len(tour_segment) < 2:
            # if DEBUG_SCRIPT_MODE: print(f"    Đoạn tour {i_tour+1} quá ngắn hoặc rỗng. Bỏ qua.")
            continue 

        if tour_segment[0] != env_instance.depot_idx or tour_segment[-1] != env_instance.depot_idx:
            if DEBUG_SCRIPT_MODE: print(f"[Eval Agent] KIỂM TRA THẤT BẠI: Đoạn tour {i_tour+1} ({tour_segment}) không bắt đầu hoặc kết thúc tại depot.")
            return float('inf'), [] 

        current_segment_distance = calculate_single_tour_distance(tour_segment, env_instance.coords)
        
        if len(tour_segment) > 2: 
            recalculated_total_distance += current_segment_distance
            valid_and_completed_tours_for_output.append(tour_segment)
            # if DEBUG_SCRIPT_MODE: print(f"    Đoạn tour {i_tour+1} hợp lệ (phục vụ khách hàng). Quãng đường: {current_segment_distance:.2f}")
        elif len(tour_segment) == 2 and env_instance.num_total_customers == 0 and not valid_and_completed_tours_for_output:
             valid_and_completed_tours_for_output.append(tour_segment) 
             # if DEBUG_SCRIPT_MODE: print(f"    Đoạn tour {i_tour+1} hợp lệ (tour rỗng, không có khách hàng).")
        # elif len(tour_segment) == 2 : 
            # if DEBUG_SCRIPT_MODE: print(f"    Đoạn tour {i_tour+1} là tour rỗng [0,0] nhưng có thể có khách hàng. Không thêm vào output cuối cùng trừ khi là lựa chọn duy nhất cho 0 khách hàng.")

    if not valid_and_completed_tours_for_output: 
        if all_customers_finally_served and env_instance.num_total_customers == 0: 
             if DEBUG_SCRIPT_MODE: print("[Eval Agent] KIỂM TRA OK: Không có khách hàng, trả về giải pháp tour rỗng.")
             return 0.0, [[env_instance.depot_idx, env_instance.depot_idx]] 
        if DEBUG_SCRIPT_MODE: print("[Eval Agent] KIỂM TRA THẤT BẠI: Tất cả khách hàng đã được phục vụ nhưng không có tour nào hợp lệ/có ý nghĩa trong output cuối cùng.")
        return float('inf'), []
    
    if DEBUG_SCRIPT_MODE: print(f"[Eval Agent] KIỂM TRA THÀNH CÔNG. Tổng quãng đường ban đầu: {recalculated_total_distance:.2f}, Số tours: {len(valid_and_completed_tours_for_output)}")
    return recalculated_total_distance, valid_and_completed_tours_for_output

def run_training_main(problem_type, data_filepath="A1.csv"):
    MODEL_NAME_BASE = f"{problem_type}_vrpb" 

    REWARD_PER_STEP = -0.05  
    BONUS_COMPLETE_EPISODE = 3000.0 
    PENALTY_NOT_ALL_SERVED = -5000.0 
    PENALTY_END_EPISODE_NOT_AT_DEPOT = -2000.0
    
    if DEBUG_SCRIPT_MODE:
        VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR = True
    else:
        VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR = False

    instance_data = load_instance_from_csv(data_filepath)
    if not instance_data:
        print(f"Không thể tải dữ liệu từ {data_filepath}. Kết thúc.")
        return

    env = VRPBEnv_MathFormulation(instance_data, problem_type=problem_type)
    env.instance_data = instance_data # Gán instance_data vào env

    state_dim = env.get_state_dim()
    action_space_size = env.action_space_size

    agent = REINFORCEAgent(state_dim, action_space_size, hidden_dim=128, learning_rate=1e-4, gamma=0.99)
    agent.load_model(f"{MODEL_NAME_BASE}_best.pth") # Tải model tốt nhất nếu có
    num_episodes_train = 50 
    max_steps_per_tour_train = env.num_nodes + 10 
    log_interval = 1 
    
    best_eval_distance = float('inf')
    successful_eval_count = 0
    all_episode_actual_distances_log = [] 

    if DEBUG_SCRIPT_MODE:
        print(f"\n{'*' * 10} Bắt đầu huấn luyện cho {problem_type.upper()} VRPB {'*' * 10}")
        print(f"Instance: {data_filepath}, Số KH: {env.num_total_customers}, LH: {env.num_linehaul_customers}, Capacity: {env.vehicle_capacity}, MaxVehicles: {env.instance_data.get('max_vehicles', 'N/A')}")
        print(f"State_dim: {state_dim}, Action_space: {action_space_size}")
        print(f"LR: {agent.optimizer.defaults['lr']}, Gamma: {agent.gamma}, Episodes: {num_episodes_train}")

    for i_episode in range(1, num_episodes_train + 1):
        if DEBUG_SCRIPT_MODE and i_episode % (log_interval // 10 if log_interval >= 10 else 1) == 0 : 
             print(f"\n\n{'#'*20} EPISODE {i_episode} START {'#'*20}")
        
        current_episode_rewards = []
        current_episode_log_probs = []
        current_episode_actual_distance = 0.0
        
        state = env.reset(full_reset=True)
        episode_done_by_env = False

        for tour_idx in range(env.num_total_customers + 3): 
            if DEBUG_SCRIPT_MODE and i_episode % (log_interval // 10 if log_interval >= 10 else 1) == 0 :
                print(f"\n-- Tour {tour_idx + 1} in Episode {i_episode} --")
            if tour_idx > 0: 
                state = env.reset(full_reset=False)
            
            if np.all(env.global_visited_mask[1:]):
                if env.current_location_idx != env.depot_idx:
                    if DEBUG_SCRIPT_MODE and i_episode % (log_interval // 10 if log_interval >= 10 else 1) == 0 : print("All customers served globally, but vehicle not at depot. Agent needs to return.")
                else: 
                    if DEBUG_SCRIPT_MODE and i_episode % (log_interval // 10 if log_interval >= 10 else 1) == 0 : print("All customers served globally and vehicle at depot. Ending episode early.")
                    episode_done_by_env = True 
                    break 

            for step_idx_in_tour in range(max_steps_per_tour_train):
                action, log_prob = agent.select_action(state, env._get_valid_actions_mask())
                next_state, reward_from_env, episode_done_by_env, info = env.step(action)
                
                actual_distance_this_step = 0
                if info.get("error") is None : 
                    actual_distance_this_step = -reward_from_env 
                current_episode_actual_distance += actual_distance_this_step
                
                step_reward = reward_from_env 
                if info.get("error") is None: 
                    step_reward += REWARD_PER_STEP

                if DEBUG_SCRIPT_MODE and i_episode % (log_interval // 10 if log_interval >= 10 else 1) == 0:
                    if step_idx_in_tour < 3 or info.get("error") is not None or step_idx_in_tour == max_steps_per_tour_train -1 : 
                         print(f"  TrainStep {step_idx_in_tour}: Act={action}, EnvRwd(dist_comp)={reward_from_env:.2f}, StepRwd(for_learning)={step_reward:.2f}, DoneEp={episode_done_by_env}, DoneTour={info.get('done_tour')}, Err={info.get('error')}")
                
                current_episode_log_probs.append(log_prob)
                current_episode_rewards.append(step_reward)
                state = next_state

                if episode_done_by_env or info.get("done_tour", False):
                    break 
            
            if episode_done_by_env: 
                break 

        all_served_final = np.all(env.global_visited_mask[1:])
        at_depot_final = (env.current_location_idx == env.depot_idx)
        final_reward_mod_info = "None"

        if all_served_final and at_depot_final:
            if current_episode_rewards: current_episode_rewards[-1] += BONUS_COMPLETE_EPISODE
            else: current_episode_rewards.append(BONUS_COMPLETE_EPISODE)
            final_reward_mod_info = f"BONUS_COMPLETE ({BONUS_COMPLETE_EPISODE})"
        elif not all_served_final:
            if current_episode_rewards: current_episode_rewards[-1] += PENALTY_NOT_ALL_SERVED
            else: current_episode_rewards.append(PENALTY_NOT_ALL_SERVED)
            final_reward_mod_info = f"PENALTY_NOT_ALL_SERVED ({PENALTY_NOT_ALL_SERVED})"
        elif all_served_final and not at_depot_final: 
            if current_episode_rewards: current_episode_rewards[-1] += PENALTY_END_EPISODE_NOT_AT_DEPOT
            else: current_episode_rewards.append(PENALTY_END_EPISODE_NOT_AT_DEPOT)
            final_reward_mod_info = f"PENALTY_END_EPISODE_NOT_AT_DEPOT ({PENALTY_END_EPISODE_NOT_AT_DEPOT})"

        if DEBUG_SCRIPT_MODE and i_episode % (log_interval // 10 if log_interval >= 10 else 1) == 0 :
            print(f"EPISODE {i_episode} END: AllServed={all_served_final}, AtDepot={at_depot_final}. FinalRewardMod: {final_reward_mod_info}")
            print(f"Total rewards for episode before update: {sum(current_episode_rewards if current_episode_rewards else [0]):.2f}")
            print(f"Actual distance for episode (for logging): {current_episode_actual_distance:.2f}")

        if current_episode_log_probs and current_episode_rewards:
            agent.rewards = current_episode_rewards
            agent.log_probs = current_episode_log_probs
            loss_val = agent.update_policy()
            agent.clear_memory()
        else:
            loss_val = 0.0
            if DEBUG_SCRIPT_MODE and i_episode % (log_interval // 10 if log_interval >= 10 else 1) == 0 : print(f"EPISODE {i_episode}: No log_probs or rewards, skipping policy update.")
        
        all_episode_actual_distances_log.append(current_episode_actual_distance)

        if i_episode % log_interval == 0:
            if len(all_episode_actual_distances_log) >= log_interval:
                avg_train_dist_log = np.mean(all_episode_actual_distances_log[-log_interval:])
            elif all_episode_actual_distances_log: 
                avg_train_dist_log = np.mean(all_episode_actual_distances_log)
            else: 
                avg_train_dist_log = float('nan')
            
            original_env_debug_mode = VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR
            VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR = False 
            
            eval_env = VRPBEnv_MathFormulation(instance_data, problem_type=problem_type)
            eval_env.instance_data = instance_data # Gán cho eval_env

            eval_dist, eval_tours = solve_one_instance_with_trained_agent(eval_env, agent, 
                                                                         max_tours_per_instance= eval_env.num_vehicles + 10, 
                                                                         max_steps_per_tour=eval_env.num_nodes + 20) 
            
            VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR = original_env_debug_mode 

            eval_status = "Success" if eval_dist != float('inf') else "Failed"
            print(f"Ep {i_episode}/{num_episodes_train} ({problem_type}), "
                  f"AvgTrainDist (last {log_interval}): {avg_train_dist_log:.2f}, "
                  f"EvalDist: {eval_dist if eval_status == 'Success' else 'N/A'}, EvalTours: {len(eval_tours) if eval_tours else 0}, "
                  f"Eval: {eval_status}, Loss: {loss_val:.4e}")

            if eval_status == "Success":
                successful_eval_count += 1
                if eval_dist < best_eval_distance:
                    best_eval_distance = eval_dist
                    agent.save_model(f"{MODEL_NAME_BASE}_best.pth")
                    print(f"  >> New best eval_dist: {best_eval_distance:.2f}. Model saved.")
    
    print(f"\nHoàn thành huấn luyện cho {problem_type} trên {data_filepath}.")
    agent.save_model(f"{MODEL_NAME_BASE}_final.pth")
    VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR = False

    print(f"\nĐánh giá cuối cùng với model tốt nhất (nếu có):")
    best_model_path = f"{MODEL_NAME_BASE}_best.pth"
    if successful_eval_count > 0 and os.path.exists(best_model_path):
        print(f"Đang tải model tốt nhất từ: {best_model_path}")
        VRPBEnv_MathFormulation.DEBUG_MODE_CLASS_VAR = False
        eval_env_final = VRPBEnv_MathFormulation(instance_data, problem_type=problem_type)
        eval_env_final.instance_data = instance_data # Gán cho eval_env_final
        agent.load_model(best_model_path)
        final_dist_rl, final_tours_rl = solve_one_instance_with_trained_agent(eval_env_final, agent)

        if final_dist_rl != float('inf') and final_tours_rl:
            print(f"\n--- Kết quả từ RL Agent (Trước 2-Opt) ---")
            print(f"Tổng quãng đường: {final_dist_rl:.2f} với {len(final_tours_rl)} tours.")
            
            if DEBUG_SCRIPT_MODE: print("\n--- Áp dụng 2-Opt Post-Processing ---")
            improved_tours_after_2opt = []
            improved_total_distance_after_2opt = 0

            for tour_rl in final_tours_rl:
                # Truyền instance_data và problem_type vào hàm 2-Opt
                optimized_single_tour = apply_2_opt_to_single_tour(list(tour_rl), eval_env_final.instance_data, problem_type)
                
                improved_tours_after_2opt.append(optimized_single_tour)
                improved_total_distance_after_2opt += calculate_single_tour_distance(optimized_single_tour, eval_env_final.coords)
            
            print(f"\n--- Kết quả SAU KHI ÁP DỤNG 2-Opt ---")
            print(f"Tổng quãng đường (đã cải thiện 2-Opt): {improved_total_distance_after_2opt:.2f} với {len(improved_tours_after_2opt)} tours.")
            for i, tour in enumerate(improved_tours_after_2opt): 
                print(f"  Lộ trình tối ưu {i+1}: {tour}")
            
            if callable(visualize_vrpb_solution):
                 visualize_vrpb_solution(instance_data, improved_tours_after_2opt, improved_total_distance_after_2opt, 
                                         title=f"VRPB {problem_type} (A1 Data - RL + 2-Opt)")
        else:
            print(f"Model tốt nhất không tạo được giải pháp hợp lệ ban đầu để áp dụng Local Search.")
    else:
        print("Không có model tốt nhất nào được lưu hoặc không có đánh giá thành công nào.")

if __name__ == '__main__':
    # Chạy cho VRPB Truyền thống
    run_training_main(problem_type="traditional", data_filepath="dataset_GJ/Vehicle-Routing-Problem-with-Backhaul/GJ/A1.csv")
