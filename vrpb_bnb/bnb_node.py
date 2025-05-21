# vrpb_bnb/bnb_node.py
import numpy as np

class BnBNode:
    def __init__(self, tour, cost, visited_mask, linehaul_load, backhaul_load, linehaul_phase_active):
        """
        Khởi tạo một nút trong cây Nhánh và Cận.
        Args:
            tour (list): Danh sách các node_idx trong lộ trình hiện tại (bắt đầu từ depot).
            cost (float): Chi phí (quãng đường) của lộ trình hiện tại.
            visited_mask (np.array): Mảng boolean theo dõi các khách hàng đã được thăm.
            linehaul_load (float): Tổng nhu cầu linehaul đã được thêm vào kế hoạch của tour này.
            backhaul_load (float): Tổng nhu cầu backhaul đã được thu gom trong tour này.
            linehaul_phase_active (bool): True nếu vẫn đang trong pha phục vụ linehaul (cho VRPB truyền thống).
        """
        self.tour = tour
        self.cost = cost # Đây sẽ là cận dưới f(n) = g(n) + h(n), hiện tại g(n) là cost, h(n) = 0 (cần cải thiện)
        self.visited_mask = visited_mask
        self.linehaul_load = linehaul_load
        self.backhaul_load = backhaul_load
        self.linehaul_phase_active = linehaul_phase_active # Cho VRPB Truyền thống

    # Để sử dụng với hàng đợi ưu tiên (nếu dùng)
    def __lt__(self, other):
        return self.cost < other.cost # So sánh dựa trên cost (cận dưới)