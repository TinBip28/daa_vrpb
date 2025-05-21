# vrpb_bnb/solve_vrpb_with_bnb.py
import time
from generate_data import load_vrpb_instance_from_excel 
from bnb_solver import VRPBBnBSolver
from utils import visualize_vrpb_solution 

def main_bnb_solver():
    DATA_FILENAME = "data_20_customers_corner_depot.xlsx" 
    TIME_LIMIT_SECONDS = 10 # Giảm giới hạn thời gian cho instance nhỏ

    instance_data = load_vrpb_instance_from_excel(DATA_FILENAME)
    if not instance_data:
        return

    print(f"\n--- Giải VRPB Truyền thống bằng Nhánh và Cận (1 xe) cho file {DATA_FILENAME} ---")
    solver_trad = VRPBBnBSolver(instance_data, problem_type="traditional") # Gọi với problem_type="traditional"
    start_time_trad = time.time()
    distance_trad, routes_trad = solver_trad.solve(time_limit_seconds=TIME_LIMIT_SECONDS)
    end_time_trad = time.time()
    print(f"Thời gian giải (Traditional BnB): {end_time_trad - start_time_trad:.2f} giây")
    print(f"Số nút đã xử lý: {solver_trad.nodes_processed_count}")

    if routes_trad:
        print(f"  Quãng đường tốt nhất (Traditional BnB): {distance_trad:.2f}")
        for i, r in enumerate(routes_trad):
            print(f"  Lộ trình {i+1}: {r}")
        visualize_vrpb_solution(instance_data, routes_trad, distance_trad, title=f"VRPB Traditional (BnB) - {DATA_FILENAME}")
    else:
        print("  Không tìm thấy giải pháp cho VRPB Truyền thống.")

    print(f"\n--- Giải VRPB Cải tiến bằng Nhánh và Cận (1 xe) cho file {DATA_FILENAME} ---")
    solver_imp = VRPBBnBSolver(instance_data, problem_type="improved") # Gọi với problem_type="improved"
    start_time_imp = time.time()
    distance_imp, routes_imp = solver_imp.solve(time_limit_seconds=TIME_LIMIT_SECONDS)
    end_time_imp = time.time()
    print(f"Thời gian giải (Improved BnB): {end_time_imp - start_time_imp:.2f} giây")
    print(f"Số nút đã xử lý: {solver_imp.nodes_processed_count}")
    
    if routes_imp:
        print(f"  Quãng đường tốt nhất (Improved BnB): {distance_imp:.2f}")
        for i, r in enumerate(routes_imp):
            print(f"  Lộ trình {i+1}: {r}")
        visualize_vrpb_solution(instance_data, routes_imp, distance_imp, title=f"VRPB Improved (BnB) - {DATA_FILENAME}")
    else:
        print("  Không tìm thấy giải pháp cho VRPB Cải tiến.")

if __name__ == "__main__":
    main_bnb_solver()