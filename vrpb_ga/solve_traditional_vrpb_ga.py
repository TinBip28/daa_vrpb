# vrpb_ga/solve_traditional_vrpb_ga.py
import time
import os
# Sử dụng hàm load CSV mới
from generate_data import load_instance_from_csv_for_ga # Đảm bảo tên file và hàm đúng
from utils import visualize_vrpb_solution # Import từ utils trong cùng thư mục vrpb_ga
from ga_core import run_ga_for_vrpb

def main_traditional_ga_vrpb():
    # --- Cấu hình Instance và Loại bài toán ---
    # Thay đổi đường dẫn gốc này cho phù hợp với cấu trúc thư mục của bạn
    # Ví dụ: nếu thư mục dataset_GJ nằm ngoài thư mục daa_vrpb-main
    # DATA_ROOT_DIR = "../../dataset_GJ/Vehicle-Routing-Problem-with-Backhaul/GJ"
    # Hoặc nếu dataset_GJ nằm trong daa_vrpb-main:
    DATA_ROOT_DIR = "dataset_GJ/Vehicle-Routing-Problem-with-Backhaul/GJ"
    
    INSTANCE_NAME = "E1.csv" # Chọn file instance
    
    DATA_FILENAME = os.path.join(DATA_ROOT_DIR, INSTANCE_NAME)
    

    PROBLEM_TYPE = "traditional"

    # --- Cấu hình Tham số GA ---
    ga_params_config = {
        'POPULATION_SIZE': 100, # Có thể điều chỉnh
        'NUM_GENERATIONS': 100, # Có thể điều chỉnh
        'MUTATION_RATE': 0.05,
        'CROSSOVER_RATE': 0.9,
        'TOURNAMENT_SIZE': 5,
        'ELITISM_COUNT': 3,
        'APPLY_LOCAL_SEARCH': True
    }
    
    instance_data = load_instance_from_csv_for_ga(DATA_FILENAME)
    
    if not instance_data:
        print(f"Không thể tải dữ liệu instance từ file CSV: {DATA_FILENAME}")
        return

    print(f"\nBắt đầu giải bài toán VRPB Truyền thống cho instance: {DATA_FILENAME}")
    print(f"Số khách hàng: {instance_data['num_customers']}")
    print(f"Số xe tối đa: {instance_data.get('num_vehicles', 'Không giới hạn')}") # In ra số xe
    print(f"Tham số GA: Pop={ga_params_config['POPULATION_SIZE']}, Gen={ga_params_config['NUM_GENERATIONS']}, LS={ga_params_config['APPLY_LOCAL_SEARCH']}")

    start_time = time.time()
    best_tours, best_distance = run_ga_for_vrpb(instance_data, PROBLEM_TYPE, ga_params_config)
    end_time = time.time()

    print(f"\nTổng thời gian thực thi GA: {end_time - start_time:.2f} giây.")

    if best_tours and best_distance != float('inf'):
        print(f"\nGiải pháp tốt nhất (Truyền thống) - Tổng quãng đường: {best_distance:.2f}")
        print(f"Số lộ trình thực tế: {len(best_tours)}")
        
        # Đếm số xe thực sự có khách hàng trong giải pháp tốt nhất
        actual_vehicles_in_best_solution = 0
        for tour in best_tours:
            if len(tour) > 2 : # Có ít nhất 1 khách hàng
                actual_vehicles_in_best_solution +=1
        print(f"Số xe có lộ trình (có KH): {actual_vehicles_in_best_solution} / {instance_data.get('num_vehicles', 'Không giới hạn')} (cho phép)")

        if "coords" in instance_data and "num_linehaul" in instance_data and "num_customers" in instance_data:
            visualize_vrpb_solution(instance_data, best_tours, best_distance,
                                    title=f"GA - VRPB Truyền thống ({INSTANCE_NAME})")
        else:
            print("Không đủ thông tin instance_data để visualize.")
            for i, tour in enumerate(best_tours): print(f"  Lộ trình {i+1}: {tour}")
    else:
        print("\nGA không tìm thấy giải pháp hợp lệ hoặc không có giải pháp với quãng đường hữu hạn.")

if __name__ == "__main__":
    main_traditional_ga_vrpb()