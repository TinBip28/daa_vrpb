# vrpb_rl_simple_pg/generate_data.py
import numpy as np
import pandas as pd

def generate_vrpb_instance_to_excel(num_customers, num_linehaul, vehicle_capacity, 
                                    depot_location="corner", filename="vrpb_instance.xlsx"): # Bỏ grid_size
    """
    Tạo một thực thể VRPB và lưu vào file Excel.
    Args:
        num_customers (int): Tổng số khách hàng.
        num_linehaul (int): Số lượng khách hàng linehaul (phải <= num_customers).
        vehicle_capacity (int): Tải trọng của xe.
        depot_location (str): Vị trí của depot. "center" hoặc "corner".
        filename (str): Tên file Excel để lưu.
    """
    if num_linehaul > num_customers:
        raise ValueError("Số lượng khách hàng linehaul không thể lớn hơn tổng số khách hàng.")

    assumed_grid_size = 100.0 # Sử dụng một kích thước lưới giả định bên trong hàm

    # --- Tạo dữ liệu ---
    # Xác định tọa độ depot
    if depot_location == "corner":
        depot_coord = np.array([5.0, 5.0]) # Ví dụ: (5,5) trong lưới [0, assumed_grid_size]
        customer_min_coord = 0.0
        customer_max_coord = assumed_grid_size
    elif depot_location == "center":
        depot_coord = np.array([assumed_grid_size / 2, assumed_grid_size / 2])
        customer_min_coord = 0.0
        customer_max_coord = assumed_grid_size
    else: 
        depot_coord = np.array([assumed_grid_size / 2, assumed_grid_size / 2]) # Mặc định là center
        customer_min_coord = 0.0
        customer_max_coord = assumed_grid_size
        print(f"Cảnh báo: depot_location '{depot_location}' không hợp lệ. Sử dụng 'center'.")

    
    coords_list = []
    demands_list = []
    node_ids_list = []
    node_types_list = []

    # Depot (Node_ID 0)
    node_ids_list.append(0)
    coords_list.append(depot_coord.tolist())
    demands_list.append(0)
    node_types_list.append("depot")

    # Khách hàng (Node_ID từ 1 đến num_customers)
    # Tạo tọa độ khách hàng trong khoảng [customer_min_coord, customer_max_coord]
    customer_coords_generated = np.random.uniform(low=customer_min_coord, high=customer_max_coord, size=(num_customers, 2))
    
    for i in range(num_customers):
        node_id = i + 1 
        node_ids_list.append(node_id)
        coords_list.append(customer_coords_generated[i].tolist())
        
        if node_id <= num_linehaul:
            demands_list.append(np.random.randint(5, 20)) 
            node_types_list.append("linehaul")
        else:
            demands_list.append(-np.random.randint(5, 20)) 
            node_types_list.append("backhaul")

    # --- Chuẩn bị DataFrame ---
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
            # 'grid_size', # Bỏ grid_size khỏi thông tin lưu trữ
            'depot_location'
        ],
        'Value': [
            num_customers, 
            num_linehaul, 
            vehicle_capacity,
            num_nodes_total,
            # assumed_grid_size, # Không cần lưu grid_size giả định nữa
            depot_location if depot_location in ["center", "corner"] else "center"
        ]
    })

    # --- Lưu vào file Excel ---
    try:
        with pd.ExcelWriter(filename) as writer:
            node_data_df.to_excel(writer, sheet_name='Nodes', index=False)
            instance_info_df.to_excel(writer, sheet_name='InstanceInfo', index=False)
        print(f"Đã tạo file dữ liệu VRPB: {filename} (Depot: {depot_location})")
    except Exception as e:
        print(f"Lỗi khi ghi file Excel: {e}")


def load_vrpb_instance_from_excel(filename="vrpb_instance.xlsx"):
    """
    Tải một thực thể VRPB từ file Excel và chuyển thành định dạng dict 
    mà VRPBEnv mong đợi.
    """
    try:
        xls = pd.ExcelFile(filename)
        nodes_df = pd.read_excel(xls, 'Nodes').sort_values(by='Node_ID').reset_index(drop=True)
        info_df = pd.read_excel(xls, 'InstanceInfo').set_index('Parameter')['Value']

        coords = nodes_df[['X_Coord', 'Y_Coord']].values.tolist()
        demands = nodes_df['Demand'].values.tolist()
        
        instance_data = {
            "coords": coords,
            "demands": demands,
            "num_customers": int(info_df['num_customers']),
            "num_linehaul": int(info_df['num_linehaul']),
            "num_nodes": int(info_df['num_nodes']),
            "vehicle_capacity": int(info_df['vehicle_capacity']),
            # grid_size không còn được lưu, nhưng có thể thêm depot_location_info nếu cần
            "depot_location_info": str(info_df.get('depot_location', 'unknown')) 
        }
        print(f"Đã tải dữ liệu VRPB từ: {filename}")
        return instance_data
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {filename}")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file Excel '{filename}': {e}")
        return None


if __name__ == "__main__":
    # Ví dụ tạo dữ liệu với depot ở góc (không cần grid_size)
    generate_vrpb_instance_to_excel(
        num_customers=20, 
        num_linehaul=7, 
        vehicle_capacity=150,
        depot_location="corner", 
        filename="data_20_customers_corner_depot.xlsx"
    )
    
    generate_vrpb_instance_to_excel(
        num_customers=10, 
        num_linehaul=3, 
        vehicle_capacity=100,
        depot_location="corner",
        filename="data_10_customers_corner_depot.xlsx"
    )

    # Ví dụ tạo dữ liệu với depot ở giữa
    generate_vrpb_instance_to_excel(
        num_customers=20, 
        num_linehaul=10, 
        vehicle_capacity=200,
        depot_location="center",
        filename="data_20_customers_center_depot.xlsx"
    )

    generate_vrpb_instance_to_excel(
        num_customers=100,
        num_linehaul=30,
        vehicle_capacity=1000,
        depot_location="corner",
        filename="data_100_customers_corner_depot.xlsx"
    )

    print("\nKiểm tra tải dữ liệu:")
    loaded_data_corner = load_vrpb_instance_from_excel("data_10_customers_corner_depot.xlsx")
    if loaded_data_corner:
        print(f"Tải thành công data_10_customers_corner_depot.xlsx. Depot info: {loaded_data_corner.get('depot_location_info')}")

    loaded_data_center = load_vrpb_instance_from_excel("data_20_customers_center_depot.xlsx")
    if loaded_data_center:
        print(f"Tải thành công data_20_customers_center_depot.xlsx. Depot info: {loaded_data_center.get('depot_location_info')}")
