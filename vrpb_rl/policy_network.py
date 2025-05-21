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
        if not isinstance(state_features, torch.Tensor):
            target_device = next(self.parameters()).device
            state_features = torch.FloatTensor(state_features).to(target_device)
        
        if state_features.dim() == 1: 
            state_features = state_features.unsqueeze(0)

        x = F.relu(self.fc1(state_features))
        x = F.relu(self.fc2(x))
        action_scores = self.fc3_action(x) 
        return action_scores

    def select_action(self, state_features, valid_actions_mask, device):
        """
        Chọn hành động.
        - Nếu self.training là True (chế độ huấn luyện): sample từ phân phối xác suất.
        - Nếu self.training là False (chế độ đánh giá): chọn hành động có xác suất cao nhất (argmax).
        """
        action_scores = self.forward(state_features) 

        masked_action_scores = action_scores.clone()
        
        if not isinstance(valid_actions_mask, torch.Tensor):
            valid_actions_mask_tensor = torch.BoolTensor(valid_actions_mask).to(device)
        else:
            valid_actions_mask_tensor = valid_actions_mask.to(device)

        if valid_actions_mask_tensor.dim() == 1: 
             valid_actions_mask_tensor = valid_actions_mask_tensor.unsqueeze(0)

        if torch.any(valid_actions_mask_tensor):
            masked_action_scores[~valid_actions_mask_tensor] = -float('inf')
        
        action_probs = F.softmax(masked_action_scores, dim=1)
        
        if torch.any(torch.isnan(action_probs)):
            valid_indices = np.where(valid_actions_mask)[0]
            if len(valid_indices) > 0:
                action_probs = torch.zeros_like(masked_action_scores)
                action_probs[0, valid_indices] = 1.0 / len(valid_indices)
            else: 
                action_probs = torch.ones_like(masked_action_scores) / masked_action_scores.size(1)

        # --- THAY ĐỔI LOGIC CHỌN HÀNH ĐỘNG DỰA TRÊN self.training ---
        if self.training: # Chế độ huấn luyện: lấy mẫu
            try:
                action_distribution = torch.distributions.Categorical(probs=action_probs)
                action_tensor = action_distribution.sample() 
                log_prob = action_distribution.log_prob(action_tensor)
            except ValueError as e: 
                # print(f"PolicyNetwork (Training): Error creating Categorical distribution: {e}")
                # print(f"  Action Probs causing error: {action_probs.cpu().detach().numpy().flatten()}")
                num_actions = masked_action_scores.size(1)
                action_idx_fallback = np.random.choice(num_actions) 
                action_tensor = torch.tensor([action_idx_fallback], device=device)
                pseudo_log_prob = torch.log(torch.tensor(1.0/num_actions + 1e-9, device=device, requires_grad=True))
                log_prob = pseudo_log_prob
        else: # Chế độ đánh giá (self.training is False): chọn argmax
            # print("[Eval Mode] Selecting action with argmax")
            action_tensor = torch.argmax(action_probs, dim=1)
            # Tính log_prob cho hành động đã chọn (cần thiết nếu hàm gọi vẫn dùng nó)
            # Lấy xác suất của hành động được chọn bằng argmax
            # action_probs is shape [1, num_actions], action_tensor is shape [1]
            # log_prob = torch.log(action_probs[0, action_tensor.item()] + 1e-9) # Thêm epsilon nhỏ để tránh log(0)
            # Cách lấy log_prob chính xác hơn khi đã có distribution (dù chỉ dùng argmax)
            try:
                action_distribution = torch.distributions.Categorical(probs=action_probs)
                log_prob = action_distribution.log_prob(action_tensor)
            except ValueError: # Fallback nếu probs vẫn có vấn đề
                log_prob = torch.log(torch.tensor(1.0/action_probs.size(1) + 1e-9, device=device))


        return action_tensor.item(), log_prob 
