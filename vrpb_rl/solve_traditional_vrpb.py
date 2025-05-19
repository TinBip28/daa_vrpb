# vrpb_rl_simple_pg/solve_traditional_vrpb.py
import numpy as np
import torch
from vrpb_env import VRPBEnv
from reinforce_agent import REINFORCEAgent
from generate_data import load_vrpb_instance_from_excel
from utils import visualize_vrpb_solution, get_node_type_from_index # Đảm bảo import get_node_type_from_index

# Hàm solve_one_instance_with_trained_agent (giữ nguyên như trong solve_improved_vrpb.py)
def solve_one_instance_with_trained_agent(env_instance, agent, max_tours_per_instance=None, max_steps_per_tour=None):
    agent.policy_network.eval()
    all_tours_nodes = []
    total_overall_distance = 0.0
    
    if max_tours_per_instance is None:
        max_tours_per_instance = env_instance.num_total_customers + 5 
    if max_steps_per_tour is None:
        max_steps_per_tour = env_instance.num_nodes * 2

    current_state = env_instance.reset(full_reset=True)
    num_tours_generated = 0
    done_episode_globally = False 

    with torch.no_grad():
        while not np.all(env_instance.global_visited_mask[1:]) and num_tours_generated < max_tours_per_instance:
            if num_tours_generated > 0:
                current_state = env_instance.reset(full_reset=False)
            
            info = {"current_tour_nodes": env_instance.current_tour_plan, 
                    "current_tour_distance": env_instance.total_distance_travelled_in_tour,
                    "done_tour": False}

            for step_in_tour in range(max_steps_per_tour):
                valid_actions_mask = env_instance._get_valid_actions_mask()
                if not np.any(valid_actions_mask):
                    break 
                
                action, _ = agent.select_action(current_state, valid_actions_mask)
                next_state, reward, done_episode_globally, info = env_instance.step(action)
                current_state = next_state
                
                if info.get("done_tour", False): break
                if done_episode_globally: break 
            
            current_tour_nodes_from_env = info.get("current_tour_nodes", env_instance.current_tour_plan)
            current_tour_distance_from_env = info.get("current_tour_distance", env_instance.total_distance_travelled_in_tour)

            if len(current_tour_nodes_from_env) > 2:
                all_tours_nodes.append(list(current_tour_nodes_from_env))
                total_overall_distance += current_tour_distance_from_env
            elif len(current_tour_nodes_from_env) == 2 and \
                 current_tour_nodes_from_env[0] == env_instance.depot_idx and \
                 current_tour_nodes_from_env[1] == env_instance.depot_idx:
                pass 
                
            num_tours_generated += 1
            if done_episode_globally: break
            
    agent.policy_network.train() 
    return total_overall_distance, all_tours_nodes


def main_traditional_vrpb():
    # --- Cấu hình cho quá trình huấn luyện và đánh giá (ĐỒNG BỘ VỚI IMPROVED) ---
    DATA_FILENAME = "data_10_customers_corner_depot.xlsx" 
    MODEL_NAME_BASE = "traditional_vrpb_policy" 
    PROBLEM_TYPE = "traditional"                
    UTILIZATION_WEIGHT = 20.0 

    instance_data = load_vrpb_instance_from_excel(DATA_FILENAME)
    if not instance_data:
        print(f"Không thể tải dữ liệu từ {DATA_FILENAME}. Hãy chạy generate_data.py trước.")
        return

    env = VRPBEnv(instance_data, problem_type=PROBLEM_TYPE)
    state_dim = env.get_state_dim()
    action_space_size = env.action_space_size 

    agent = REINFORCEAgent(state_dim, action_space_size, hidden_dim=64, learning_rate=5e-4, gamma=0.99)
    # agent.load_model(f"{MODEL_NAME_BASE}_best.pth")

    num_episodes_train = 100 
    max_steps_per_tour_train = env.num_nodes + 5
    log_interval = 1 
    best_eval_distance = float('inf')
    all_episode_total_distances = []

    for i_episode in range(1, num_episodes_train + 1):
        state = env.reset(full_reset=True)
        episode_log_probs = []
        episode_rewards = []
        current_instance_total_distance_debug = 0
        num_tours_this_instance = 0
        instance_solved_this_episode = False
        last_completed_tour_plan_for_bonus = None
        info_tour_loop = {"done_tour": False} 

        for _ in range(env.num_total_customers + 5): 
            if num_tours_this_instance > 0:
                state = env.reset(full_reset=False)
            
            tour_log_probs = []
            tour_rewards = []
            
            for t_step_tour in range(max_steps_per_tour_train):
                valid_actions_mask = env._get_valid_actions_mask()
                if not np.any(valid_actions_mask):
                    if env.current_location_idx != env.depot_idx: tour_rewards.append(-200)
                    break 
                
                action, log_prob = agent.select_action(state, valid_actions_mask) 
                next_state, reward, done_episode_globally, info_tour_loop = env.step(action)
                
                tour_log_probs.append(log_prob)
                tour_rewards.append(reward)
                current_instance_total_distance_debug -= reward
                state = next_state
                
                if info_tour_loop.get("done_tour", False):
                    last_completed_tour_plan_for_bonus = list(env.current_tour_plan)
                    break
                if done_episode_globally:
                    instance_solved_this_episode = True
                    last_completed_tour_plan_for_bonus = list(env.current_tour_plan)
                    break
            
            if tour_rewards and last_completed_tour_plan_for_bonus:
                linehaul_demand_in_tour = 0
                backhaul_demand_in_tour = 0
                for node_idx in last_completed_tour_plan_for_bonus:
                    if node_idx != env.depot_idx:
                        node_type = get_node_type_from_index(node_idx, env.num_linehaul_customers)
                        if node_type == "linehaul":
                            linehaul_demand_in_tour += env.demands[node_idx]
                        elif node_type == "backhaul":
                            backhaul_demand_in_tour += abs(env.demands[node_idx])
                
                utilization_bonus = 0
                if linehaul_demand_in_tour > 0:
                    utilization_bonus += (linehaul_demand_in_tour / (env.vehicle_capacity + 1e-9) )
                if backhaul_demand_in_tour > 0:
                    utilization_bonus += (backhaul_demand_in_tour / (env.vehicle_capacity + 1e-9) )
                
                if tour_rewards: 
                    tour_rewards[-1] += UTILIZATION_WEIGHT * utilization_bonus
                last_completed_tour_plan_for_bonus = None 

            episode_log_probs.extend(tour_log_probs)
            episode_rewards.extend(tour_rewards)
            num_tours_this_instance += 1
            if instance_solved_this_episode: break
        
        agent.rewards = episode_rewards
        agent.log_probs = episode_log_probs
        loss_value = agent.update_policy()
        agent.clear_memory()
        all_episode_total_distances.append(current_instance_total_distance_debug)

        if i_episode % log_interval == 0:
            avg_dist_last_log = np.mean(all_episode_total_distances[-log_interval:]) if all_episode_total_distances else 0.0
            eval_distance, eval_tours_nodes = solve_one_instance_with_trained_agent(env, agent, 
                                                                                  max_tours_per_instance=env.num_total_customers + 2, 
                                                                                  max_steps_per_tour=env.num_nodes + 2)
            num_eval_tours = len(eval_tours_nodes) if eval_tours_nodes else 0
            
            print(f"Ep {i_episode} ({PROBLEM_TYPE}), AvgTrainDist: {avg_dist_last_log:.2f}, EvalDist: {eval_distance:.2f}, EvalTours: {num_eval_tours}, Loss: {loss_value:.8e}")
            
            if eval_distance != float('inf') and eval_distance < best_eval_distance and eval_distance > 0 and num_eval_tours > 0:
                best_eval_distance = eval_distance
                print(f"best_eval_distance:{best_eval_distance}")
                agent.save_model(f"{MODEL_NAME_BASE}_best.pth")
        
    print(f"Hoàn thành huấn luyện cho {PROBLEM_TYPE}.")
    agent.save_model(f"{MODEL_NAME_BASE}_final.pth")

    print(f"\nĐánh giá cuối cùng ({PROBLEM_TYPE}) với model tốt nhất:")
    if best_eval_distance != float('inf'): 
        agent.load_model(f"{MODEL_NAME_BASE}_best.pth") 
        final_dist, final_tours = solve_one_instance_with_trained_agent(env, agent) 
        print(f"Tổng quãng đường (best model): {final_dist:.2f} với {len(final_tours)} tours.")
        for i, tour in enumerate(final_tours): print(f"  Lộ trình {i+1}: {tour}")

        if final_tours:
            # SỬA LỖI Ở ĐÂY: Truyền final_dist vào hàm visualize_vrpb_solution
            visualize_vrpb_solution(instance_data, final_tours, final_dist, title=f"VRPB {PROBLEM_TYPE} Solution (Best Model)")
    else:
        print("Không có model tốt nhất nào được lưu để visualize.")

if __name__ == "__main__":
    main_traditional_vrpb()
