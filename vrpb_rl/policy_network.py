# vrpb_rl_simple_pg/policy_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimplePolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_space_size, hidden_dim=128):
        super(SimplePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3_action = nn.Linear(hidden_dim // 2, action_space_size) 

    def forward(self, state_features):
        # Đảm bảo input là tensor và đúng device
        if not isinstance(state_features, torch.Tensor):
            # Lấy device từ một tham số của model (ví dụ fc1.weight.device)
            # để đảm bảo state_features được chuyển đến cùng device với model
            target_device = next(self.parameters()).device
            state_features = torch.FloatTensor(state_features).to(target_device)
        
        if state_features.dim() == 1: # Nếu là vector 1D, thêm batch dimension
            state_features = state_features.unsqueeze(0)

        x = F.relu(self.fc1(state_features))
        x = F.relu(self.fc2(x))
        action_scores = self.fc3_action(x) # Output là điểm số thô cho các hành động
        return action_scores

    def select_action(self, state_features, valid_actions_mask, device):
        """
        Chọn hành động dựa trên trạng thái và mask các hành động hợp lệ.
        QUAN TRỌNG: Không sử dụng torch.no_grad() ở đây nếu hàm này được gọi trong quá trình training.
        Việc có tính gradient hay không sẽ do người gọi quyết định.
        """
        # state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(device) # Đã chuyển vào forward
        
        # Bước 1: Lấy action_scores từ mạng policy.
        # self.forward sẽ xử lý việc chuyển state_features thành tensor và đưa lên đúng device.
        action_scores = self.forward(state_features) # Shape: (1, action_space_size)

        # Bước 2: Áp dụng masking để vô hiệu hóa các hành động không hợp lệ.
        # Đặt điểm số của các hành động không hợp lệ thành giá trị rất nhỏ (-infinity)
        # để xác suất của chúng sau softmax sẽ gần bằng 0.
        masked_action_scores = action_scores.clone() # Tạo bản sao để không ảnh hưởng action_scores gốc
        
        # Đảm bảo valid_actions_mask là boolean tensor trên đúng device
        if not isinstance(valid_actions_mask, torch.Tensor):
            valid_actions_mask_tensor = torch.BoolTensor(valid_actions_mask).to(device)
        else:
            valid_actions_mask_tensor = valid_actions_mask.to(device)

        if valid_actions_mask_tensor.dim() == 1: # Đảm bảo có batch dimension
             valid_actions_mask_tensor = valid_actions_mask_tensor.unsqueeze(0)

        # Chỉ áp dụng mask nếu có ít nhất một hành động hợp lệ được chỉ định trong mask
        # (Mặc dù lý tưởng là env luôn cung cấp ít nhất 1 hành động hợp lệ, ví dụ quay về depot)
        if torch.any(valid_actions_mask_tensor):
            masked_action_scores[~valid_actions_mask_tensor] = -float('inf')
        # else: Nếu không có hành động nào hợp lệ theo mask, softmax có thể ra NaN.
        #       Cần xử lý ở phần sau hoặc đảm bảo env luôn có lựa chọn.

        # Bước 3: Tính toán xác suất chọn hành động bằng softmax.
        action_probs = F.softmax(masked_action_scores, dim=1) # Shape: (1, action_space_size)
        
        # Xử lý trường hợp action_probs có thể là NaN (nếu tất cả masked_action_scores là -inf)
        if torch.any(torch.isnan(action_probs)):
            # print("PolicyNetwork: action_probs contains NaN. Masked scores might all be -inf.")
            # print(f"  Original scores: {action_scores.cpu().detach().numpy().flatten()}")
            # print(f"  Valid mask: {valid_actions_mask}")
            # print(f"  Masked scores: {masked_action_scores.cpu().detach().numpy().flatten()}")
            # Fallback: nếu NaN, thử phân phối đều trên các hành động hợp lệ (nếu có)
            valid_indices = np.where(valid_actions_mask)[0]
            if len(valid_indices) > 0:
                action_probs = torch.zeros_like(masked_action_scores)
                action_probs[0, valid_indices] = 1.0 / len(valid_indices)
            else: # Không có hành động hợp lệ nào, phân phối đều trên tất cả (trạng thái xấu)
                action_probs = torch.ones_like(masked_action_scores) / masked_action_scores.size(1)
            # print(f"  Fallback action_probs: {action_probs.cpu().detach().numpy().flatten()}")


        # Bước 4: Sample một hành động từ phân phối xác suất và tính log_prob.
        try:
            action_distribution = torch.distributions.Categorical(probs=action_probs)
            action = action_distribution.sample() # action là một tensor chứa index
            log_prob = action_distribution.log_prob(action) # log_prob này SẼ có grad_fn nếu action_scores có
            return action.item(), log_prob # Trả về index (int) và log_prob (tensor)
        except ValueError as e: 
            # print(f"PolicyNetwork: Error creating Categorical distribution: {e}")
            # print(f"  Action Probs causing error: {action_probs.cpu().detach().numpy().flatten()}")
            # Fallback an toàn, mặc dù điều này chỉ ra vấn đề sâu hơn với probs
            num_actions = masked_action_scores.size(1)
            action_idx = np.random.choice(num_actions) # Chọn hành động ngẫu nhiên
            # log_prob này không chính xác cho việc học nhưng để chương trình chạy tiếp
            pseudo_log_prob = torch.log(torch.tensor(1.0/num_actions + 1e-9, device=device, requires_grad=True)) # Cố gắng cho requires_grad
            return action_idx, pseudo_log_prob
