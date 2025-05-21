# vrpb_rl_simple_pg/solve_improved_vrpb_ga.py
import time
import os
# Sử dụng hàm load_vrpb_instance_from_excel
from generate_data import load_vrpb_instance_from_excel, generate_vrpb_instance_to_excel
from utils import visualize_vrpb_solution
from ga_core import run_ga_for_vrpb # Đảm bảo ga_core.py và vrpb_ga_utils.py ở đúng vị trí

def main_improved_ga_vrpb():
    # --- Cấu hình Instance và Loại bài toán ---
    DATA_FILENAME = "data_10_customers_corner_depot.xlsx"  # Tên file Excel chứa dữ liệu instance
    PROBLEM_TYPE = "improved" 

   

    # --- Cấu hình Tham số GA ---
    ga_params_config = {
        'POPULATION_SIZE': 10,
        'NUM_GENERATIONS': 50,
        'MUTATION_RATE': 0.05,
        'CROSSOVER_RATE': 0.85,
        'TOURNAMENT_SIZE': 5,
        'ELITISM_COUNT': 3,
        'APPLY_LOCAL_SEARCH': True
    }
    
    # Tải dữ liệu từ file Excel duy nhất
    instance_data = load_vrpb_instance_from_excel(DATA_FILENAME)
    
    if not instance_data:
        print(f"Không thể tải dữ liệu instance từ file Excel: {DATA_FILENAME}")
        return

    print(f"Bắt đầu giải bài toán VRPB Cải tiến cho instance: {DATA_FILENAME}")
    print(f"Số khách hàng: {instance_data['num_customers']}")
    print(f"Tham số GA: Pop={ga_params_config['POPULATION_SIZE']}, Gen={ga_params_config['NUM_GENERATIONS']}, LS={ga_params_config['APPLY_LOCAL_SEARCH']}")

    start_time = time.time()
    best_tours, best_distance = run_ga_for_vrpb(instance_data, PROBLEM_TYPE, ga_params_config)
    end_time = time.time()

    print(f"\nTổng thời gian thực thi GA: {end_time - start_time:.2f} giây.")

    if best_tours and best_distance != float('inf'):
        print(f"\nGiải pháp tốt nhất (Cải tiến) - Tổng quãng đường: {best_distance:.2f}")
        print(f"Số lộ trình: {len(best_tours)}")
        visualize_vrpb_solution(instance_data, best_tours, best_distance, 
                                title=f"GA - VRPB Cải tiến ({DATA_FILENAME})",)
    else:
        print("\nGA không tìm thấy giải pháp hợp lệ hoặc không có giải pháp với quãng đường hữu hạn.")

if __name__ == "__main__":
    main_improved_ga_vrpb()