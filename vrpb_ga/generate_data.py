# vrpb_ga/generate_data.py (Hoặc bạn có thể tạo file data_loader_ga.py)
import pandas as pd
import numpy as np

def generate_vrpb_instance_to_excel(num_customers, num_linehaul, vehicle_capacity,
                                    depot_location="corner", filename="vrpb_instance.xlsx"):
    if num_linehaul > num_customers:
        raise ValueError("Số lượng khách hàng linehaul không thể lớn hơn tổng số khách hàng.")

    assumed_grid_size = 100.0

    if depot_location == "corner":
        depot_coord = np.array([5.0, 5.0])
        customer_min_coord = 0.0
        customer_max_coord = assumed_grid_size
    elif depot_location == "center":
        depot_coord = np.array([assumed_grid_size / 2, assumed_grid_size / 2])
        customer_min_coord = 0.0
        customer_max_coord = assumed_grid_size
    else:
        depot_coord = np.array([assumed_grid_size / 2, assumed_grid_size / 2])
        customer_min_coord = 0.0
        customer_max_coord = assumed_grid_size
        print(f"Cảnh báo: depot_location '{depot_location}' không hợp lệ. Sử dụng 'center'.")

    coords_list = []
    demands_list = []
    node_ids_list = []
    node_types_list = []

    node_ids_list.append(0) # Node ID trong file Excel, không phải index nội bộ
    coords_list.append(depot_coord.tolist())
    demands_list.append(0)
    node_types_list.append("depot")

    customer_coords_generated = np.random.uniform(low=customer_min_coord, high=customer_max_coord, size=(num_customers, 2))
    
    for i in range(num_customers):
        node_id = i + 1 # Node ID trong file Excel
        node_ids_list.append(node_id)
        coords_list.append(customer_coords_generated[i].tolist())
        
        if node_id <= num_linehaul:
            demands_list.append(np.random.randint(5, 20))
            node_types_list.append("linehaul")
        else:
            demands_list.append(-np.random.randint(5, 20)) # Backhaul có demand âm
            node_types_list.append("backhaul")

    node_data_df = pd.DataFrame({
        'Node_ID': node_ids_list,
        'X_Coord': [c[0] for c in coords_list],
        'Y_Coord': [c[1] for c in coords_list],
        'Demand': demands_list,
        'Type': node_types_list
    })

    num_nodes_total = num_customers + 1
    instance_info_df = pd.DataFrame({
        'Parameter': [
            'num_customers', 
            'num_linehaul', 
            'vehicle_capacity', 
            'num_nodes',
            'depot_location'
            # Các thông tin khác như num_vehicles (k) có thể được thêm ở đây nếu file Excel cần
        ],
        'Value': [
            num_customers, 
            num_linehaul, 
            vehicle_capacity,
            num_nodes_total,
            depot_location if depot_location in ["center", "corner"] else "center"
        ]
    })

    try:
        with pd.ExcelWriter(filename) as writer:
            node_data_df.to_excel(writer, sheet_name='Nodes', index=False)
            instance_info_df.to_excel(writer, sheet_name='InstanceInfo', index=False)
        print(f"Đã tạo file dữ liệu VRPB Excel: {filename} (Depot: {depot_location})")
    except Exception as e:
        print(f"Lỗi khi ghi file Excel: {e}")


def load_vrpb_instance_from_excel(filename="vrpb_instance.xlsx"):
    try:
        xls = pd.ExcelFile(filename)
        nodes_df = pd.read_excel(xls, 'Nodes').sort_values(by='Node_ID').reset_index(drop=True)
        info_df = pd.read_excel(xls, 'InstanceInfo').set_index('Parameter')['Value']

        coords = nodes_df[['X_Coord', 'Y_Coord']].values.tolist()
        demands = nodes_df['Demand'].values.tolist() # Giữ nguyên dấu demand từ file
        
        instance_data = {
            "coords": coords,
            "demands": demands,
            "num_customers": int(info_df.loc['num_customers']),
            "num_linehaul": int(info_df.loc['num_linehaul']),
            "num_nodes": int(info_df.loc['num_nodes']),
            "vehicle_capacity": float(info_df.loc['vehicle_capacity']),
            "depot_idx": 0, # Mặc định depot là node đầu tiên (index 0)
            "num_vehicles": int(info_df.get('num_vehicles', info_df.get('k', int(info_df.loc['num_customers'])))) # Lấy 'k' hoặc 'num_vehicles'
        }
        print(f"Đã tải dữ liệu VRPB từ file Excel: {filename}")
        return instance_data
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file Excel {filename}")
        return None
    except KeyError as e:
        print(f"Lỗi: Thiếu thông tin (tham số: {e}) trong sheet 'InstanceInfo' của file Excel {filename}.")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file Excel '{filename}': {e}")
        return None


def load_instance_from_csv_for_ga(filepath):
    """
    Tải dữ liệu instance từ file CSV theo định dạng của dataset_GJ
    và chuyển đổi sang định dạng instance_data mà GA hiện tại có thể sử dụng.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {filepath}")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file CSV '{filepath}': {e}")
        return None

    depot_info_row = df[df['type'] == 0]
    if depot_info_row.empty:
        print(f"Lỗi: Không tìm thấy thông tin depot (type 0) trong file {filepath}")
        return None
    depot_info = depot_info_row.iloc[0]

    depot_coord = np.array([depot_info['x'], depot_info['y']])
    vehicle_capacity = depot_info['Q']
    num_vehicles_from_file = int(depot_info['k']) # Đọc số lượng xe từ cột 'k'

    coords_list = [depot_coord.tolist()] # Node 0 là depot
    demands_list = [0.0]                 # Demand của depot là 0
    
    # Sẽ map node_id gốc từ file CSV sang index nội bộ (1 đến N cho khách hàng)
    # GA làm việc với các index từ 1 đến num_customers
    
    num_linehaul = 0
    num_backhaul = 0

    # Tách khách hàng linehaul và backhaul, sắp xếp theo node_id gốc để đảm bảo thứ tự nhất quán
    linehaul_customers_df = df[df['type'] == 1].sort_values(by='node_id').reset_index(drop=True)
    backhaul_customers_df = df[df['type'] == 2].sort_values(by='node_id').reset_index(drop=True)

    # Thêm khách hàng linehaul vào lists
    for _, row in linehaul_customers_df.iterrows():
        coords_list.append([row['x'], row['y']])
        demands_list.append(float(row['demand'])) # Demand dương cho linehaul
        num_linehaul += 1

    # Thêm khách hàng backhaul vào lists
    # Các index của backhaul sẽ tiếp nối ngay sau linehaul
    for _, row in backhaul_customers_df.iterrows():
        coords_list.append([row['x'], row['y']])
        demands_list.append(-abs(float(row['demand']))) # Demand âm cho backhaul
        num_backhaul += 1

    num_total_customers = num_linehaul + num_backhaul
    num_nodes_instance = num_total_customers + 1 # Bao gồm depot

    instance_data = {
        "coords": coords_list,
        "demands": demands_list,
        "num_customers": num_total_customers,
        "num_linehaul": num_linehaul, # Số khách hàng linehaul này sẽ được dùng bởi get_node_type_from_index
        "num_nodes": num_nodes_instance,
        "vehicle_capacity": float(vehicle_capacity),
        "depot_idx": 0,
        "num_vehicles": num_vehicles_from_file # Số xe tối đa cho phép
    }
    
    print(f"Đã tải dữ liệu từ file CSV: {filepath} cho GA")
    print(f"  Tổng số nút (bao gồm depot): {instance_data['num_nodes']}")
    print(f"  Số khách hàng Linehaul: {instance_data['num_linehaul']}")
    print(f"  Số khách hàng Backhaul: {num_backhaul}")
    print(f"  Dung lượng xe: {instance_data['vehicle_capacity']}")
    print(f"  Số xe tối đa cho phép: {instance_data['num_vehicles']}")
    return instance_data

if __name__ == '__main__':
    # Test thử hàm load CSV
    # Đảm bảo bạn có file A1.csv trong đường dẫn tương đối hoặc tuyệt đối chính xác
    # Ví dụ: filepath_csv = "../dataset_GJ/Vehicle-Routing-Problem-with-Backhaul/GJ/A1.csv"
    # Hoặc nếu dataset_GJ cùng cấp với thư mục chứa vrpb_ga:
    filepath_csv = "../../dataset_GJ/Vehicle-Routing-Problem-with-Backhaul/GJ/A1.csv" 
    # Điều chỉnh đường dẫn này cho phù hợp với cấu trúc thư mục của bạn

    # Kiểm tra xem file có tồn tại không trước khi load
    import os
    if os.path.exists(filepath_csv):
        data_csv = load_instance_from_csv_for_ga(filepath_csv)
        if data_csv:
            print("\nKiểm tra dữ liệu đã tải từ CSV:")
            print(f"  Tọa độ Depot: {data_csv['coords'][data_csv['depot_idx']]}")
            print(f"  Demand Depot: {data_csv['demands'][data_csv['depot_idx']]}")
            print(f"  Số lượng xe: {data_csv['num_vehicles']}")
            if data_csv['num_linehaul'] > 0:
                print(f"  Tọa độ KH Linehaul đầu tiên (Node 1 theo index nội bộ): {data_csv['coords'][1]}")
                print(f"  Demand KH Linehaul đầu tiên (Node 1 theo index nội bộ): {data_csv['demands'][1]}")
            if data_csv['num_customers'] > data_csv['num_linehaul']:
                first_bh_internal_idx = data_csv['num_linehaul'] + 1
                print(f"  Tọa độ KH Backhaul đầu tiên (Node {first_bh_internal_idx} theo index nội bộ): {data_csv['coords'][first_bh_internal_idx]}")
                print(f"  Demand KH Backhaul đầu tiên (Node {first_bh_internal_idx} theo index nội bộ): {data_csv['demands'][first_bh_internal_idx]}")
    else:
        print(f"Lỗi: Không tìm thấy file dữ liệu CSV tại '{filepath_csv}'. Vui lòng kiểm tra đường dẫn.")

    # # Test tạo file Excel (nếu cần)
    # generate_vrpb_instance_to_excel(
    #     num_customers=5,
    #     num_linehaul=2,
    #     vehicle_capacity=50,
    #     depot_location="corner",
    #     filename="test_ga_excel.xlsx"
    # )
    # data_excel = load_vrpb_instance_from_excel("test_ga_excel.xlsx")
    # if data_excel:
    #     print("\nKiểm tra dữ liệu đã tải từ Excel:")
    #     print(data_excel)