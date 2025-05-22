import pandas as pd
import numpy as np

def load_instance_from_csv(filepath="A1.csv"):
    """
    Tải dữ liệu instance từ file A1.csv và chuyển đổi sang định dạng instance_data.
    File A1.csv có các cột: type,node_id,x,y,demand,Q,k,L,B
    - type 0: Depot và thông số chung
    - type 1: Khách hàng Linehaul
    - type 2: Khách hàng Backhaul

    Quy ước demand trong instance_data trả về:
    - Linehaul: demand > 0
    - Backhaul: demand < 0 (theo paper DRL, delta_j < 0 cho pickup)
    - Depot: demand = 0
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {filepath}")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file CSV '{filepath}': {e}")
        return None

    # Lấy thông tin depot và các tham số chung từ dòng đầu tiên (type 0)
    depot_info_row = df[df['type'] == 0]
    if depot_info_row.empty:
        print(f"Lỗi: Không tìm thấy thông tin depot (type 0) trong file {filepath}")
        return None
    depot_info = depot_info_row.iloc[0]

    depot_coord = np.array([depot_info['x'], depot_info['y']])
    vehicle_capacity = depot_info['Q']
    nums_vehicles = depot_info['k']
    # Các thông số khác như k (số xe), L (giới hạn quãng đường), B có thể được thêm nếu cần
    # Khởi tạo danh sách cho instance_data
    # Node 0 luôn là depot
    coords_list = [depot_coord.tolist()]
    demands_list = [0] # Demand của depot là 0
    # node_ids_internal sẽ là index trong các list này (0 cho depot, 1 đến N cho KH)

    num_linehaul = 0
    num_backhaul = 0

    # Khách hàng Linehaul (type 1)
    linehaul_customers_df = df[df['type'] == 1].sort_values(by='node_id') # Sắp xếp để có thứ tự nhất quán
    for _, row in linehaul_customers_df.iterrows():
        coords_list.append([row['x'], row['y']])
        demands_list.append(float(row['demand'])) # Đảm bảo demand là float và dương
        num_linehaul += 1

    # Khách hàng Backhaul (type 2)
    backhaul_customers_df = df[df['type'] == 2].sort_values(by='node_id') # Sắp xếp
    for _, row in backhaul_customers_df.iterrows():
        coords_list.append([row['x'], row['y']])
        # Theo DRL formulation, delta_j < 0 cho pickup.
        # Nếu demand trong file đã là âm thì giữ nguyên, nếu dương thì đổi dấu.
        # Để an toàn, dùng abs() rồi nhân -1.
        demands_list.append(-abs(float(row['demand'])))
        num_backhaul += 1

    num_total_customers = num_linehaul + num_backhaul
    num_nodes = num_total_customers + 1 # Bao gồm depot

    instance_data = {
        "coords": coords_list,
        "demands": demands_list, # Quy ước: LH > 0, BH < 0
        "num_customers": num_total_customers,
        "num_linehaul": num_linehaul, # Số lượng khách hàng linehaul
        "num_nodes": num_nodes,
        "vehicle_capacity": float(vehicle_capacity),
        "num_vehicles": int(nums_vehicles), # Số xe (nếu có)
        "depot_idx": 0,
        # Các thông tin khác có thể hữu ích
        "max_vehicles": int(depot_info.get('k', 1000)), # Số xe tối đa (nếu có)
        "max_route_length": float(depot_info.get('L', float('inf'))), # Giới hạn quãng đường (nếu có)
    }
    
    print(f"Đã tải dữ liệu từ {filepath}:")
    print(f"  Tổng số nút (bao gồm depot): {instance_data['num_nodes']}")
    print(f"  Số khách hàng Linehaul: {instance_data['num_linehaul']}")
    print(f"  Số khách hàng Backhaul: {num_backhaul}")
    print(f"  Dung lượng xe: {instance_data['vehicle_capacity']}")
    # print(f"  Demands (đã xử lý dấu): {instance_data['demands']}")
    return instance_data

if __name__ == '__main__':
    # Test thử hàm load
    data = load_instance_from_csv("dataset_GJ/Vehicle-Routing-Problem-with-Backhaul/GJ/A1.csv") # Đảm bảo A1.csv ở cùng thư mục
    if data:
        print("\nKiểm tra dữ liệu đã tải:")
        print(f"  Tọa độ Depot: {data['coords'][data['depot_idx']]}")
        print(f"  Demand Depot: {data['demands'][data['depot_idx']]}")
        print(f" Số lượng xe: {data['num_vehicles']}")
        if data['num_linehaul'] > 0:
            print(f"  Tọa độ KH Linehaul đầu tiên (Node 1): {data['coords'][1]}")
            print(f"  Demand KH Linehaul đầu tiên (Node 1): {data['demands'][1]}")
        if data['num_customers'] > data['num_linehaul']:
            first_bh_idx = data['num_linehaul'] + 1
            print(f"  Tọa độ KH Backhaul đầu tiên (Node {first_bh_idx}): {data['coords'][first_bh_idx]}")
            print(f"  Demand KH Backhaul đầu tiên (Node {first_bh_idx}): {data['demands'][first_bh_idx]}")
