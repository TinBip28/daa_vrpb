# vrpb_ga/utils.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def calculate_distance(coord1, coord2):
    """Tính khoảng cách Euclidean giữa hai tọa độ."""
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def get_node_type_from_index(node_index, num_linehaul_customers_in_instance):
    """
    Xác định loại node dựa trên index trong mảng (0 là depot).
    """
    if node_index == 0:
        return "depot"
    if node_index <= num_linehaul_customers_in_instance:
        return "linehaul"
    else:
        return "backhaul"

def visualize_vrpb_solution(instance_data, all_tours_nodes, total_distance, title="VRPB Solution"):
    coords = np.array(instance_data["coords"])
    num_linehaul = instance_data["num_linehaul"]
    num_total_customers = instance_data["num_customers"]
    depot_idx = instance_data.get("depot_idx", 0) # Sử dụng get để tránh lỗi nếu thiếu

    plt.figure(figsize=(13, 10))

    # 1. Vẽ các điểm (nodes)
    if depot_idx < len(coords):
        plt.plot(coords[depot_idx, 0], coords[depot_idx, 1], 's', markersize=10, color='red', label=f'Depot ({depot_idx})')
        plt.text(coords[depot_idx, 0] + 0.5, coords[depot_idx, 1] + 0.5, str(depot_idx))

    linehaul_coords_x = []
    linehaul_coords_y = []
    for i in range(1, num_linehaul + 1):
        if i < len(coords): # Đảm bảo index không vượt quá kích thước mảng coords
            linehaul_coords_x.append(coords[i, 0])
            linehaul_coords_y.append(coords[i, 1])
            plt.text(coords[i, 0] + 0.5, coords[i, 1] + 0.5, str(i))
    if linehaul_coords_x:
        plt.plot(linehaul_coords_x, linehaul_coords_y, 'o', markersize=8, color='blue', label=f'Linehaul (1-{num_linehaul})')

    backhaul_coords_x = []
    backhaul_coords_y = []
    if num_total_customers > num_linehaul:
        for i in range(num_linehaul + 1, num_total_customers + 1):
            if i < len(coords): # Đảm bảo index không vượt quá kích thước mảng coords
                backhaul_coords_x.append(coords[i, 0])
                backhaul_coords_y.append(coords[i, 1])
                plt.text(coords[i, 0] + 0.5, coords[i, 1] + 0.5, str(i))
        if backhaul_coords_x:
            plt.plot(backhaul_coords_x, backhaul_coords_y, '^', markersize=8, color='green', label=f'Backhaul ({num_linehaul+1}-{num_total_customers})')

    # 2. Vẽ các lộ trình
    route_colors = list(mcolors.TABLEAU_COLORS.values())
    if all_tours_nodes and len(all_tours_nodes) > len(route_colors):
        try:
            cmap = plt.get_cmap('tab20')
            route_colors = [cmap(i) for i in np.linspace(0, 1, len(all_tours_nodes))]
        except:
             route_colors = route_colors * ( (len(all_tours_nodes) // len(route_colors)) + 1)

    if all_tours_nodes:
        for i, tour_nodes in enumerate(all_tours_nodes):
            if not tour_nodes or len(tour_nodes) < 2:
                continue
            
            tour_coords_x = [coords[node_idx, 0] for node_idx in tour_nodes if node_idx < len(coords)]
            tour_coords_y = [coords[node_idx, 1] for node_idx in tour_nodes if node_idx < len(coords)]
            
            if not tour_coords_x: continue # Bỏ qua nếu không có tọa độ hợp lệ

            color_idx = i % len(route_colors)
            color = route_colors[color_idx]
            
            plt.plot(tour_coords_x, tour_coords_y, '-', linewidth=1.5, color=color, label=f'Route {i+1}')
            for j in range(len(tour_nodes) - 1):
                start_node = tour_nodes[j]
                end_node = tour_nodes[j+1]
                if max(start_node, end_node) < len(coords):
                     plt.arrow(coords[start_node, 0], coords[start_node, 1],
                          coords[end_node, 0] - coords[start_node, 0],
                          coords[end_node, 1] - coords[start_node, 1],
                          head_width=1.5, head_length=2, fc=color, ec=color, length_includes_head=True,
                          alpha=0.7)

    plt.xlabel("Tọa độ X")
    plt.ylabel("Tọa độ Y")
    
    num_actual_tours = len(all_tours_nodes) if all_tours_nodes else 0
    full_title = f"{title}\nTổng quãng đường: {total_distance:.2f} | Số tours: {num_actual_tours}"
    plt.title(full_title)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()