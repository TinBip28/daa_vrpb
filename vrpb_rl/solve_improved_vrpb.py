# vrpb_rl_simple_pg/solve_improved_vrpb.py
import numpy as np
import torch
from vrpb_env import VRPBEnv
from reinforce_agent import REINFORCEAgent 
from generate_data import load_vrpb_instance_from_excel
from utils import visualize_vrpb_solution, get_node_type_from_index, calculate_distance

def solve_one_instance_with_trained_agent(env_instance, agent, max_tours_per_instance=None, max_steps_per_tour=None):
    """
    Giải một thực thể VRPB bằng agent đã huấn luyện.
    ĐÁNH GIÁ CỰC KỲ NGHIÊM NGẶT:
    1. Tất cả khách hàng PHẢI được phục vụ.
    2. MỌI tour được tạo ra PHẢI bắt đầu VÀ kết thúc tại depot.
    Nếu không, giải pháp bị coi là không hợp lệ (distance = inf).
    """
    agent.policy_network.eval() 
    all_generated_tours_segments = [] 
    
    if max_tours_per_instance is None:
        max_tours_per_instance = env_instance.num_total_customers + 2 # Giới hạn chặt hơn một chút
    if max_steps_per_tour is None:
        max_steps_per_tour = env_instance.num_nodes + 5 # Cho phép nhiều bước hơn để hoàn thành tour

    current_state = env_instance.reset(full_reset=True) 
    num_tours_generated = 0
    
    info_last_step_of_tour = {} # Sẽ lưu info từ env.step()

    with torch.no_grad(): 
        while not np.all(env_instance.global_visited_mask[1:]) and num_tours_generated < max_tours_per_instance:
            if num_tours_generated > 0: 
                current_state = env_instance.reset(full_reset=False) 
            
            # current_tour_this_segment sẽ được xây dựng trong vòng lặp dưới
            # và được lấy từ env.current_tour_plan sau khi tour kết thúc
            
            temp_done_episode_globally_this_tour = False # Cờ từ env.step()
            for step_in_tour in range(max_steps_per_tour):
                valid_actions_mask = env_instance._get_valid_actions_mask() 
                if not np.any(valid_actions_mask): 
                    # print(f"[Eval] Tour {num_tours_generated + 1}, Step {step_in_tour}: No valid actions. Current loc: {env_instance.current_location_idx}")
                    break 
                
                action, _ = agent.select_action(current_state, valid_actions_mask)
                next_state, reward, temp_done_episode_globally_this_tour, info_last_step_of_tour = env_instance.step(action)
                current_state = next_state 
                
                if info_last_step_of_tour.get("done_tour", False): # Tour kết thúc (xe về depot ở bước này)
                    break 
                if temp_done_episode_globally_this_tour: # Tất cả KH đã được phục vụ VÀ xe về depot ở bước này
                    break 
            
            # Lưu lại đoạn tour vừa được tạo, dựa trên current_tour_plan của env
            # vì nó được cập nhật chính xác nhất trong env.step()
            tour_segment_from_env = list(info_last_step_of_tour.get("current_tour_nodes", []))
            if tour_segment_from_env and len(tour_segment_from_env) > 1: # Chỉ lưu nếu có ít nhất [0,X]
                all_generated_tours_segments.append(tour_segment_from_env)
            
            num_tours_generated += 1
            if temp_done_episode_globally_this_tour: 
                break 
            if not np.all(env_instance.global_visited_mask[1:]) and num_tours_generated >= max_tours_per_instance:
                # print("[Eval] Max tours reached, but not all customers served.")
                break 
            
    agent.policy_network.train() 

    # --- KIỂM TRA TÍNH HỢP LỆ CỰC KỲ NGHIÊM NGẶT ---
    all_customers_finally_served = np.all(env_instance.global_visited_mask[1:])
    
    if not all_customers_finally_served:
        # print("[Eval] NGHIÊM NGẶT LỖI: KHÔNG PHẢI TẤT CẢ KHÁCH HÀNG ĐƯỢC PHỤC VỤ.")
        return float('inf'), [] 

    if not all_generated_tours_segments:
        if all_customers_finally_served: # Lạ, tất cả KH xong nhưng không có tour?
             # print("[Eval] NGHIÊM NGẶT LỖI: Tất cả KH xong nhưng không có tour nào được ghi nhận.")
             return float('inf'), [] 
        # else: Trường hợp không có tour và không xong KH đã bị bắt ở trên

    valid_and_completed_tours_for_output = []
    recalculated_total_distance = 0.0

    for i_tour, tour_segment in enumerate(all_generated_tours_segments):
        if not tour_segment or len(tour_segment) < 2 : # Ít nhất phải là [depot, depot] nếu rỗng
            # print(f"[Eval] NGHIÊM NGẶT CẢNH BÁO: Tour segment {i_tour+1} rỗng hoặc quá ngắn ({tour_segment}). Bỏ qua.")
            continue # Bỏ qua các segment không hợp lệ này

        # 1. Mỗi tour PHẢI bắt đầu từ depot
        if tour_segment[0] != env_instance.depot_idx:
            # print(f"[Eval] NGHIÊM NGẶT LỖI: Tour segment {i_tour+1} ({tour_segment}) không bắt đầu từ depot.")
            return float('inf'), [] 

        # 2. Mỗi tour PHẢI kết thúc tại depot
        if tour_segment[-1] != env_instance.depot_idx:
            # print(f"[Eval] NGHIÊM NGẶT LỖI: Tour segment {i_tour+1} ({tour_segment}) không kết thúc tại depot.")
            return float('inf'), []

        # Nếu tour hợp lệ (bắt đầu và kết thúc tại depot)
        current_segment_distance = 0.0
        for i in range(len(tour_segment) - 1):
            current_segment_distance += calculate_distance(env_instance.coords[tour_segment[i]], 
                                                          env_instance.coords[tour_segment[i+1]])
        
        # Chỉ thêm vào output nếu tour này có phục vụ ít nhất 1 khách hàng (dài hơn 2 điểm)
        if len(tour_segment) > 2:
            recalculated_total_distance += current_segment_distance
            valid_and_completed_tours_for_output.append(tour_segment)
        elif len(tour_segment) == 2 and tour_segment[0] == env_instance.depot_idx and tour_segment[1] == env_instance.depot_idx:
            # Đây là tour rỗng [0,0], vẫn có thể được tạo ra.
            # Nếu không có tour nào khác phục vụ KH, giải pháp sẽ không hợp lệ (do all_customers_finally_served).
            # Nếu có các tour khác, tour rỗng này không đóng góp vào quãng đường.
            # Quyết định có thêm vào output hay không tùy thuộc vào bạn muốn hiển thị nó không.
            # Hiện tại, ta không thêm tour rỗng vào output cuối cùng nếu có các tour khác.
            if not valid_and_completed_tours_for_output and i_tour == len(all_generated_tours_segments) -1 : # Nếu đây là tour duy nhất và rỗng
                 valid_and_completed_tours_for_output.append(tour_segment) # Vẫn thêm để có gì đó trả về nếu KH đã xong
            pass


    # Nếu sau khi lọc, không còn tour nào có ý nghĩa (ví dụ chỉ toàn tour [0,0])
    # nhưng tất cả KH đã được phục vụ (trường hợp này rất lạ, có thể do 1 KH duy nhất ở depot?)
    if not valid_and_completed_tours_for_output and all_customers_finally_served:
        # print("[Eval] NGHIÊM NGẶT CẢNH BÁO: Không có tour nào có ý nghĩa được hình thành dù tất cả KH đã xong.")
        # Kiểm tra xem có phải chỉ có depot không
        if env_instance.num_total_customers == 0: # Không có khách hàng nào
            return 0.0, [[env_instance.depot_idx, env_instance.depot_idx]] # Tour rỗng là hợp lệ
        return float('inf'), []


    return recalculated_total_distance, valid_and_completed_tours_for_output


def main_improved_vrpb():
    # --- Cấu hình cho quá trình huấn luyện và đánh giá ---
    DATA_FILENAME = "data_10_customers_corner_depot.xlsx" 
    MODEL_NAME_BASE = "improved_vrpb_policy" # Đổi tên model để không ghi đè
    PROBLEM_TYPE = "improved"
    UTILIZATION_WEIGHT = 15.0 # Có thể thử nghiệm với giá trị này
    PENALTY_NOT_ALL_SERVED = -2000 # Hình phạt khi không phục vụ hết KH trong training
    PENALTY_TOUR_NOT_AT_DEPOT_TRAINING = -500 # Hình phạt nếu tour training không về depot (khi chưa xong hết KH)
    BONUS_COMPLETE_EPISODE_TRAINING = 750 # Thưởng lớn khi hoàn thành đúng episode training

    instance_data = load_vrpb_instance_from_excel(DATA_FILENAME)
    if not instance_data:
        print(f"Không thể tải dữ liệu từ {DATA_FILENAME}. Hãy chạy generate_data.py trước.")
        return

    env = VRPBEnv(instance_data, problem_type=PROBLEM_TYPE)
    state_dim = env.get_state_dim() 
    action_space_size = env.action_space_size 

    agent = REINFORCEAgent(state_dim, action_space_size, hidden_dim=128, learning_rate=2e-4, gamma=0.99)
    # agent.load_model(f"{MODEL_NAME_BASE}_best.pth") 

    num_episodes_train = 300 # Tăng số episodes
    max_steps_per_tour_train = env.num_nodes + 7 # Tăng nhẹ số bước cho mỗi tour
    log_interval = 10
    best_eval_distance = float('inf') 
    all_episode_total_distances_train_log = [] # Để log AvgTrainDist
    successful_eval_count_log = 0 # Để theo dõi số lần eval thành công

    for i_episode in range(1, num_episodes_train + 1):
        state = env.reset(full_reset=True) 
        
        episode_log_probs = []
        episode_rewards = []
        
        current_instance_total_distance_for_log = 0 
        num_tours_this_instance = 0 
        
        # Cờ theo dõi trạng thái hoàn thành của episode huấn luyện hiện tại
        all_customers_served_this_training_episode = False
        last_tour_ended_at_depot_this_training_episode = False
        
        info_tour_loop = {} # Khởi tạo trước vòng lặp tour

        for tour_count in range(env.num_total_customers + 2): # Giới hạn số tour trong một episode huấn luyện
            if num_tours_this_instance > 0:
                state = env.reset(full_reset=False) # Reset cho tour mới
            
            tour_log_probs_segment = [] 
            tour_rewards_segment = []   
            
            # Cờ từ env.step() cho tour hiện tại
            done_episode_from_env_this_tour = False 
            
            for t_step_tour in range(max_steps_per_tour_train):
                valid_actions_mask = env._get_valid_actions_mask() 
                if not np.any(valid_actions_mask): 
                    if env.current_location_idx != env.depot_idx: 
                        tour_rewards_segment.append(PENALTY_TOUR_NOT_AT_DEPOT_TRAINING / 2) # Phạt vì kẹt
                    break 
                
                action, log_prob = agent.select_action(state, valid_actions_mask) 
                next_state, reward, done_episode_from_env_this_tour, info_tour_loop = env.step(action)
                
                tour_log_probs_segment.append(log_prob)
                tour_rewards_segment.append(reward)
                current_instance_total_distance_for_log -= reward 
                state = next_state 
                
                if info_tour_loop.get("done_tour", False): # Tour kết thúc (xe về depot ở bước này)
                    last_tour_ended_at_depot_this_training_episode = True
                    if done_episode_from_env_this_tour: # Nếu cũng là lúc tất cả KH xong
                        all_customers_served_this_training_episode = True
                    break 
                if done_episode_from_env_this_tour: # Tất cả KH xong VÀ xe về depot
                    all_customers_served_this_training_episode = True
                    last_tour_ended_at_depot_this_training_episode = True # Vì done_episode_from_env chỉ true khi về depot
                    break
            
            # Áp dụng bonus utilization cho tour segment vừa hoàn thành
            current_tour_plan_for_bonus = info_tour_loop.get("current_tour_nodes", [])
            if tour_rewards_segment and current_tour_plan_for_bonus and len(current_tour_plan_for_bonus) > 1 :
                linehaul_demand_in_tour = 0
                backhaul_demand_in_tour = 0
                for node_idx in current_tour_plan_for_bonus:
                    if node_idx != env.depot_idx:
                        node_type = get_node_type_from_index(node_idx, env.num_linehaul_customers)
                        if node_type == "linehaul":
                            linehaul_demand_in_tour += env.demands[node_idx]
                        elif node_type == "backhaul":
                            backhaul_demand_in_tour += abs(env.demands[node_idx])
                
                utilization_bonus = 0.0
                if env.vehicle_capacity > 0: # Tránh chia cho 0
                    if linehaul_demand_in_tour > 0:
                        utilization_bonus += (linehaul_demand_in_tour / env.vehicle_capacity )
                    if backhaul_demand_in_tour > 0:
                        utilization_bonus += (backhaul_demand_in_tour / env.vehicle_capacity )
                
                if tour_rewards_segment: 
                    tour_rewards_segment[-1] += UTILIZATION_WEIGHT * utilization_bonus

            episode_log_probs.extend(tour_log_probs_segment)
            episode_rewards.extend(tour_rewards_segment)
            num_tours_this_instance += 1

            if all_customers_served_this_training_episode and last_tour_ended_at_depot_this_training_episode:
                break # Thoát khỏi vòng lặp tạo tour nếu đã hoàn thành đúng
        
        # --- Áp dụng thưởng/phạt cuối episode huấn luyện ---
        final_check_all_served = np.all(env.global_visited_mask[1:])
        final_check_at_depot = (env.current_location_idx == env.depot_idx)

        if final_check_all_served and final_check_at_depot:
            if episode_rewards:
                episode_rewards[-1] += BONUS_COMPLETE_EPISODE_TRAINING
            else: # Nếu không có reward nào mà vẫn hoàn thành (rất hiếm)
                episode_rewards.append(BONUS_COMPLETE_EPISODE_TRAINING)
        elif not final_check_all_served: # Nếu không phục vụ hết khách
            if episode_rewards:
                episode_rewards[-1] += PENALTY_NOT_ALL_SERVED 
            else:
                episode_rewards.append(PENALTY_NOT_ALL_SERVED)
        elif final_check_all_served and not final_check_at_depot: # Phục vụ hết KH nhưng không về depot
            if episode_rewards:
                episode_rewards[-1] += PENALTY_TOUR_NOT_AT_DEPOT_TRAINING # Dùng lại hình phạt tour không về depot
            else:
                episode_rewards.append(PENALTY_TOUR_NOT_AT_DEPOT_TRAINING)


        agent.rewards = episode_rewards
        agent.log_probs = episode_log_probs
        loss_value = agent.update_policy() 
        agent.clear_memory() 
        all_episode_total_distances_train_log.append(current_instance_total_distance_for_log)

        if i_episode % log_interval == 0:
            avg_dist_last_log = np.mean(all_episode_total_distances_train_log[-log_interval:]) if all_episode_total_distances_train_log else 0.0
            eval_distance, eval_tours_nodes = solve_one_instance_with_trained_agent(env, agent, 
                                                                                  max_tours_per_instance=env.num_total_customers + 2, 
                                                                                  max_steps_per_tour=env.num_nodes + 5)
            num_eval_tours = len(eval_tours_nodes) if eval_tours_nodes else 0
            
            eval_status_msg = "Success" if eval_distance != float('inf') else "Failed"
            print(f"Ep {i_episode} ({PROBLEM_TYPE}), AvgTrainDist: {avg_dist_last_log:.2f}, EvalDist: {eval_distance if eval_distance != float('inf') else 'N/A'}, EvalTours: {num_eval_tours}, Eval: {eval_status_msg}, Loss: {loss_value:.8e}")
            
            if eval_distance != float('inf') and eval_distance < best_eval_distance:
                best_eval_distance = eval_distance
                successful_eval_count_log +=1
                agent.save_model(f"{MODEL_NAME_BASE}_best.pth")
                print(f"  >> New best eval distance: {best_eval_distance:.2f} with {num_eval_tours} tours. Model saved.")
            elif eval_distance == float('inf'):
                pass # Không làm gì nếu eval thất bại
        
    print(f"Hoàn thành huấn luyện cho {PROBLEM_TYPE}.")
    agent.save_model(f"{MODEL_NAME_BASE}_final.pth") 

    print(f"\nĐánh giá cuối cùng ({PROBLEM_TYPE}) với model tốt nhất:")
    if best_eval_distance != float('inf'): 
        eval_env_final = VRPBEnv(instance_data, problem_type=PROBLEM_TYPE)
        agent.load_model(f"{MODEL_NAME_BASE}_best.pth") 
        final_dist, final_tours = solve_one_instance_with_trained_agent(eval_env_final, agent,
                                                                        max_tours_per_instance=env.num_total_customers + 2, 
                                                                        max_steps_per_tour=env.num_nodes + 5)
        
        if final_dist != float('inf') and final_tours:
            print(f"Tổng quãng đường (best model): {final_dist:.2f} với {len(final_tours)} tours.")
            for i, tour in enumerate(final_tours): print(f"  Lộ trình {i+1}: {tour}")
            visualize_vrpb_solution(instance_data, final_tours, final_dist, title=f"VRPB {PROBLEM_TYPE} Solution (Best Model)")
        else:
            print("Model tốt nhất không tạo được giải pháp hợp lệ khi đánh giá cuối cùng (hoặc không có tour nào).")
    else:
        print("Không có model tốt nhất nào được lưu (hoặc không có đánh giá thành công nào) để visualize.")

if __name__ == "__main__":
    main_improved_vrpb()
