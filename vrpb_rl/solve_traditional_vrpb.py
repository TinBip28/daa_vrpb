# vrpb_rl_simple_pg/solve_traditional_vrpb.py
import numpy as np
import torch
from vrpb_env import VRPBEnv # Sử dụng VRPBEnv đã được tinh chỉnh
from reinforce_agent import REINFORCEAgent
from generate_data import load_vrpb_instance_from_excel
from utils import visualize_vrpb_solution, get_node_type_from_index, calculate_distance

# Hàm solve_one_instance_with_trained_agent (giữ nguyên như phiên bản nghiêm ngặt đã thảo luận)
def solve_one_instance_with_trained_agent(env_instance, agent, max_tours_per_instance=None, max_steps_per_tour=None):
    agent.policy_network.eval() 
    all_generated_tours_segments = [] 
    
    if max_tours_per_instance is None:
        max_tours_per_instance = env_instance.num_total_customers + 2 
    if max_steps_per_tour is None:
        max_steps_per_tour = env_instance.num_nodes + 7 

    current_state = env_instance.reset(full_reset=True) 
    num_tours_generated = 0
    
    info_last_step_of_tour = {} 

    with torch.no_grad(): 
        while not np.all(env_instance.global_visited_mask[1:]) and num_tours_generated < max_tours_per_instance:
            if num_tours_generated > 0: 
                current_state = env_instance.reset(full_reset=False) 
            
            info_last_step_of_tour = {"current_tour_nodes": list(env_instance.current_tour_plan), 
                                      "current_tour_distance": env_instance.total_distance_travelled_in_tour,
                                      "done_tour": False}
            
            temp_done_episode_globally_this_tour = False 
            for step_in_tour in range(max_steps_per_tour):
                valid_actions_mask = env_instance._get_valid_actions_mask() 
                if not np.any(valid_actions_mask): 
                    info_last_step_of_tour = {
                        "current_tour_nodes": list(env_instance.current_tour_plan),
                        "current_tour_distance": env_instance.total_distance_travelled_in_tour,
                        "done_tour": (env_instance.current_location_idx == env_instance.depot_idx)
                    }
                    break 
                
                action, _ = agent.select_action(current_state, valid_actions_mask)
                next_state, reward, temp_done_episode_globally_this_tour, info_last_step_of_tour = env_instance.step(action)
                current_state = next_state 
                
                if info_last_step_of_tour.get("done_tour", False): 
                    break 
                if temp_done_episode_globally_this_tour: 
                    break 
            
            tour_segment_from_env = list(info_last_step_of_tour.get("current_tour_nodes", []))
            if tour_segment_from_env and len(tour_segment_from_env) > 1:
                all_generated_tours_segments.append(tour_segment_from_env)
            
            num_tours_generated += 1
            if temp_done_episode_globally_this_tour: 
                break 
            if not np.all(env_instance.global_visited_mask[1:]) and num_tours_generated >= max_tours_per_instance:
                break 
            
    agent.policy_network.train() 

    all_customers_finally_served = np.all(env_instance.global_visited_mask[1:])
    
    if not all_customers_finally_served:
        return float('inf'), [] 

    if not all_generated_tours_segments:
        if all_customers_finally_served and env_instance.num_total_customers == 0 : 
             return 0.0, [[env_instance.depot_idx, env_instance.depot_idx]]
        return float('inf'), [] 

    valid_and_completed_tours_for_output = []
    recalculated_total_distance = 0.0

    for i_tour, tour_segment in enumerate(all_generated_tours_segments):
        if not tour_segment or len(tour_segment) < 2 : 
            if not (len(tour_segment) == 2 and tour_segment[0] == env_instance.depot_idx and tour_segment[1] == env_instance.depot_idx):
                return float('inf'), [] 
            continue 

        if tour_segment[0] != env_instance.depot_idx:
            return float('inf'), [] 

        if tour_segment[-1] != env_instance.depot_idx:
            return float('inf'), []

        current_segment_distance = 0.0
        for i in range(len(tour_segment) - 1):
            current_segment_distance += calculate_distance(env_instance.coords[tour_segment[i]], 
                                                          env_instance.coords[tour_segment[i+1]])
        
        if len(tour_segment) > 2: 
            recalculated_total_distance += current_segment_distance
            valid_and_completed_tours_for_output.append(tour_segment)
        elif len(tour_segment) == 2 and not valid_and_completed_tours_for_output and i_tour == len(all_generated_tours_segments) -1:
            if env_instance.num_total_customers == 0:
                 valid_and_completed_tours_for_output.append(tour_segment) 
    
    if not valid_and_completed_tours_for_output and all_customers_finally_served and env_instance.num_total_customers > 0:
        return float('inf'), []
    
    if not valid_and_completed_tours_for_output and env_instance.num_total_customers == 0: 
        return 0.0, [[env_instance.depot_idx, env_instance.depot_idx]]

    return recalculated_total_distance, valid_and_completed_tours_for_output


def main_traditional_vrpb(): # Đổi tên hàm cho phù hợp nếu bạn đang chạy kịch bản nhiều tour
    # --- Cấu hình đồng bộ với solve_improved_vrpb.py (phiên bản strict_train_v3) ---
    DATA_FILENAME = "data_10_customers_corner_depot.xlsx" 
    MODEL_NAME_BASE = "traditional_vrpb_policy" # Đổi tên model để phân biệt
    PROBLEM_TYPE = "traditional"                         # Loại bài toán
    
    # Các hằng số thưởng/phạt (đồng bộ)
    UTILIZATION_WEIGHT = 15.0 
    PENALTY_NOT_ALL_SERVED_TRAINING = -2000
    PENALTY_TOUR_NOT_AT_DEPOT_TRAINING = 500
    BONUS_COMPLETE_EPISODE_TRAINING = 750 

    instance_data = load_vrpb_instance_from_excel(DATA_FILENAME)
    if not instance_data:
        print(f"Không thể tải dữ liệu từ {DATA_FILENAME}. Hãy chạy generate_data.py trước.")
        return

    env = VRPBEnv(instance_data, problem_type=PROBLEM_TYPE)
    state_dim = env.get_state_dim()
    action_space_size = env.action_space_size 

    # Cấu hình agent đồng bộ
    agent = REINFORCEAgent(state_dim, action_space_size, hidden_dim=128, learning_rate=1e-4, gamma=0.995)
    # agent.load_model(f"{MODEL_NAME_BASE}_best.pth")

    # Tham số huấn luyện đồng bộ
    num_episodes_train = 300 
    max_steps_per_tour_train = env.num_nodes + 10 
    log_interval = 10
    best_eval_distance = float('inf')
    all_episode_total_distances_train_log = [] 
    successful_eval_count_log = 0 

    # --- Vòng lặp huấn luyện chính (GIỐNG HỆT solve_improved_vrpb.py) ---
    for i_episode in range(1, num_episodes_train + 1):
        state = env.reset(full_reset=True) 
        episode_log_probs = []
        episode_rewards = []
        current_instance_total_distance_for_log = 0 
        
        done_episode_from_env_overall = False 
        
        for tour_count in range(env.num_total_customers + 2): 
            if tour_count > 0: 
                state = env.reset(full_reset=False) 
            
            tour_log_probs_segment = [] 
            tour_rewards_segment = []   
            
            info_tour_loop = {} # Khởi tạo ở đây để đảm bảo nó có giá trị
            done_episode_from_env_this_tour = False 
            
            for t_step_tour in range(max_steps_per_tour_train):
                valid_actions_mask = env._get_valid_actions_mask() 
                if not np.any(valid_actions_mask): 
                    if env.current_location_idx != env.depot_idx: 
                        if tour_rewards_segment: tour_rewards_segment[-1] += PENALTY_TOUR_NOT_AT_DEPOT_TRAINING / 2
                        else: tour_rewards_segment.append(PENALTY_TOUR_NOT_AT_DEPOT_TRAINING / 2)
                    # Cập nhật info_tour_loop trước khi break nếu kẹt
                    info_tour_loop = { 
                        "current_tour_nodes": list(env.current_tour_plan),
                        "current_tour_distance": env.total_distance_travelled_in_tour,
                        "done_tour": (env.current_location_idx == env.depot_idx)
                    }
                    break 
                
                action, log_prob = agent.select_action(state, valid_actions_mask) 
                next_state, reward, done_episode_from_env_this_tour, info_tour_loop = env.step(action)
                
                tour_log_probs_segment.append(log_prob)
                tour_rewards_segment.append(reward)
                current_instance_total_distance_for_log -= reward 
                state = next_state 
                
                if info_tour_loop.get("done_tour", False): 
                    if done_episode_from_env_this_tour: 
                        done_episode_from_env_overall = True
                    break 
            
            current_tour_plan_for_bonus = info_tour_loop.get("current_tour_nodes", [])
            if tour_rewards_segment and current_tour_plan_for_bonus and len(current_tour_plan_for_bonus) > 1 :
                linehaul_demand_in_tour = 0; backhaul_demand_in_tour = 0
                for node_idx in current_tour_plan_for_bonus:
                    if node_idx != env.depot_idx:
                        node_type = get_node_type_from_index(node_idx, env.num_linehaul_customers)
                        if node_type == "linehaul": linehaul_demand_in_tour += env.demands[node_idx]
                        elif node_type == "backhaul": backhaul_demand_in_tour += abs(env.demands[node_idx])
                utilization_bonus = 0.0
                if env.vehicle_capacity > 0: 
                    if linehaul_demand_in_tour > 0: utilization_bonus += (linehaul_demand_in_tour / env.vehicle_capacity )
                    if backhaul_demand_in_tour > 0: utilization_bonus += (backhaul_demand_in_tour / env.vehicle_capacity )
                if tour_rewards_segment: tour_rewards_segment[-1] += UTILIZATION_WEIGHT * utilization_bonus

            episode_log_probs.extend(tour_log_probs_segment)
            episode_rewards.extend(tour_rewards_segment)

            if done_episode_from_env_overall :
                break 
        
        final_check_all_served_training = np.all(env.global_visited_mask[1:])
        final_check_at_depot_training = (env.current_location_idx == env.depot_idx)

        if final_check_all_served_training and final_check_at_depot_training:
            if episode_rewards: episode_rewards[-1] += BONUS_COMPLETE_EPISODE_TRAINING
            else: episode_rewards.append(BONUS_COMPLETE_EPISODE_TRAINING)
        elif not final_check_all_served_training: 
            if episode_rewards: episode_rewards[-1] += PENALTY_NOT_ALL_SERVED_TRAINING
            else: episode_rewards.append(PENALTY_NOT_ALL_SERVED_TRAINING)
        elif final_check_all_served_training and not final_check_at_depot_training: 
            if episode_rewards: episode_rewards[-1] += PENALTY_TOUR_NOT_AT_DEPOT_TRAINING 
            else: episode_rewards.append(PENALTY_TOUR_NOT_AT_DEPOT_TRAINING)

        agent.rewards = episode_rewards
        agent.log_probs = episode_log_probs
        loss_value = agent.update_policy() 
        agent.clear_memory() 
        all_episode_total_distances_train_log.append(current_instance_total_distance_for_log)

        if i_episode % log_interval == 0:
            avg_dist_last_log = np.mean(all_episode_total_distances_train_log[-log_interval:]) if all_episode_total_distances_train_log else 0.0
            eval_env_log = VRPBEnv(instance_data, problem_type=PROBLEM_TYPE) 
            eval_distance, eval_tours_nodes = solve_one_instance_with_trained_agent(eval_env_log, agent, 
                                                                                  max_tours_per_instance=env.num_total_customers + 2, 
                                                                                  max_steps_per_tour=env.num_nodes + 7)
            num_eval_tours = len(eval_tours_nodes) if eval_tours_nodes else 0
            
            eval_status_msg = "Success" if eval_distance != float('inf') else "Failed"
            print(f"Ep {i_episode} ({PROBLEM_TYPE}), AvgTrainDist: {avg_dist_last_log:.2f}, EvalDist: {eval_distance if eval_distance != float('inf') else 'N/A'}, EvalTours: {num_eval_tours}, Eval: {eval_status_msg}, Loss: {loss_value:.8e}")
            
            if eval_distance != float('inf') and eval_distance < best_eval_distance:
                best_eval_distance = eval_distance
                successful_eval_count_log +=1
                agent.save_model(f"{MODEL_NAME_BASE}_best.pth")
                print(f"  >> New best eval distance for {PROBLEM_TYPE}: {best_eval_distance:.2f} with {num_eval_tours} tours. Model saved.")
        
    print(f"Hoàn thành huấn luyện cho {PROBLEM_TYPE}.")
    agent.save_model(f"{MODEL_NAME_BASE}_final.pth") 

    print(f"\nĐánh giá cuối cùng ({PROBLEM_TYPE}) với model tốt nhất:")
    if best_eval_distance != float('inf'): 
        eval_env_final = VRPBEnv(instance_data, problem_type=PROBLEM_TYPE)
        agent.load_model(f"{MODEL_NAME_BASE}_best.pth") 
        final_dist, final_tours = solve_one_instance_with_trained_agent(eval_env_final, agent,
                                                                        max_tours_per_instance=env.num_total_customers + 2, 
                                                                        max_steps_per_tour=env.num_nodes + 7)
        
        if final_dist != float('inf') and final_tours:
            print(f"Tổng quãng đường (best model): {final_dist:.2f} với {len(final_tours)} tours.")
            for i, tour in enumerate(final_tours): print(f"  Lộ trình {i+1}: {tour}")
            visualize_vrpb_solution(instance_data, final_tours, final_dist, title=f"VRPB {PROBLEM_TYPE} Solution (Best Model)")
        else:
            print(f"Model tốt nhất cho {PROBLEM_TYPE} không tạo được giải pháp hợp lệ khi đánh giá cuối cùng.")
    else:
        print(f"Không có model tốt nhất nào được lưu cho {PROBLEM_TYPE} (hoặc không có đánh giá thành công nào).")

if __name__ == "__main__":
    main_traditional_vrpb()
