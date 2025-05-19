# vrpb_rl_simple_pg/reinforce_agent.py
import torch
import torch.optim as optim
import numpy as np
from policy_network import SimplePolicyNetwork

class REINFORCEAgent:
    def __init__(self, state_dim, action_space_size, hidden_dim=128, learning_rate=1e-3, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Agent sử dụng thiết bị: {self.device}")
        
        self.policy_network = SimplePolicyNetwork(state_dim, action_space_size, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma

        self.rewards = []
        self.log_probs = []

    def select_action(self, state_features, valid_actions_mask):
        action, log_prob = self.policy_network.select_action(state_features, valid_actions_mask, self.device)
        return action, log_prob

    def store_outcome(self, reward, log_prob):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def clear_memory(self):
        del self.rewards[:]
        del self.log_probs[:]
        
    def update_policy(self):
        if not self.log_probs or not self.rewards:
            # print("Debug (Agent): No log_probs or rewards in memory, skipping update. Returning Loss: 0.0")
            return 0.0

        discounted_returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_returns.insert(0, R)
        
        if not discounted_returns:
            # print("Debug (Agent): discounted_returns list is empty. Returning Loss: 0.0")
            return 0.0

        discounted_returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32, device=self.device)
        
        if len(discounted_returns_tensor) > 1:
            mean_val = discounted_returns_tensor.mean()
            std_val = discounted_returns_tensor.std()
            if std_val > 1e-9: 
                discounted_returns_tensor = (discounted_returns_tensor - mean_val) / (std_val + 1e-9)
            else: 
                discounted_returns_tensor = discounted_returns_tensor - mean_val 
        elif len(discounted_returns_tensor) == 0:
            return 0.0

        policy_loss_items = [] 
        for i, (log_prob, G_t) in enumerate(zip(self.log_probs, discounted_returns_tensor)):
            if not isinstance(log_prob, torch.Tensor):
                continue
            if torch.isnan(log_prob) or torch.isinf(log_prob) or \
               torch.isnan(G_t) or torch.isinf(G_t):
                continue
            loss_component = -log_prob * G_t
            policy_loss_items.append(loss_component)
        
        if not policy_loss_items:
            # print("Debug (Agent): No valid policy_loss_items to stack. Returning Loss: 0.0")
            return 0.0

        calculated_loss_value = -9999.99 # Sentinel value

        try:
            if not all(isinstance(item, torch.Tensor) for item in policy_loss_items):
                # print("Debug (Agent): Not all items in policy_loss_items are Tensors. Returning 0.0")
                return 0.0

            policy_loss_tensor = torch.stack(policy_loss_items).sum()
            calculated_loss_value = policy_loss_tensor.item() 
            
            # This was the print that showed non-zero values in your previous debug
            # print(f"Debug (Agent): SUMMED policy_loss_tensor value (before backward): {calculated_loss_value:.8e}")

            if torch.isnan(policy_loss_tensor) or torch.isinf(policy_loss_tensor):
                # print(f"Debug (Agent): SUMMED policy_loss_tensor is NaN or Inf: {calculated_loss_value:.8e}. Skipping backward. Returning Loss: 0.0")
                return 0.0 
            
            self.optimizer.zero_grad()
            policy_loss_tensor.backward()
            self.optimizer.step()
        
        except RuntimeError as e:
            # This block should ideally not be hit if the grad issue is fixed
            print(f"Debug (Agent): !!! update_policy: RuntimeError during backward/step: {e} !!!")
            print(f"Debug (Agent): update_policy: Returning Loss: 0.0 due to RuntimeError.")
            return 0.0 

        # --- UNCOMMENT AND MODIFY THIS PRINT ---
        # print(f"Debug (Agent): update_policy IS RETURNING THIS LOSS VALUE: {calculated_loss_value:.8e}")
        # --- END MODIFIED PRINT ---
        return calculated_loss_value


    def save_model(self, filepath):
        torch.save(self.policy_network.state_dict(), filepath)
        print(f"Đã lưu model vào {filepath}")

    def load_model(self, filepath):
        try:
            self.policy_network.load_state_dict(torch.load(filepath, map_location=self.device))
            self.policy_network.eval()
            print(f"Đã tải model từ {filepath}")
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file model {filepath}. Bỏ qua việc tải model.")
        except Exception as e:
            print(f"Lỗi khi tải model: {e}. Bỏ qua việc tải model.")
