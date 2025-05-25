import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.distributions import Normal # 确保导入 Normal

# 假设 EDP.py 文件与此脚本在同一目录或 PYTHONPATH 中
# 如果 EDP.py 在 toy_experiments 子目录中
from toy_experiments.EDP import EDP

# --- 模拟参数 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = 2
action_dim = 2
max_action = 1.0
discount = 0.0
tau = 0.005
lambda_policy_loss_weight = 1.0
beta_schedule_name = 'linear'
diffusion_timesteps_K = 50
time_embedding_dim_for_edp = 32
hidden_mlp_dim = 256
learning_rate = 3e-4
num_epochs = 200
iterations_per_epoch = 200
batch_size_data = 128
num_eval_samples = 1000
plot_axis_limit = 1.2

# --- 用户定义的 Data_Sampler 类 ---
class Data_Sampler:
    def __init__(self, state_data, action_data, reward_data, device_val):
        self.state = state_data.to(device_val)
        self.action = action_data.to(device_val)
        self.reward = reward_data.to(device_val)
        self.num_samples = self.state.shape[0]
        self.device = device_val
        self.next_state = state_data.clone().to(device_val)
        self.not_done = torch.ones(self.num_samples, 1, device=device_val)
        print(f"Data_Sampler initialized with {self.num_samples} samples on {self.device}.")
        if self.num_samples > 0:
            print(f"  Action mean: {self.action.mean(dim=0).cpu().numpy()}, std: {self.action.std(dim=0).cpu().numpy()}")
            print(f"  Reward mean: {self.reward.mean().item():.3f}, std: {self.reward.std().item():.3f}")
        else:
            print("Warning: Data_Sampler received 0 samples!")

    def sample(self, batch_size):
        if self.num_samples == 0:
            print("Warning: Attempting to sample from an empty Data_Sampler.")
            dummy_state = torch.empty((batch_size, self.state.shape[1] if self.state.ndim > 1 and self.state.shape[1] > 0 else state_dim), device=self.device)
            dummy_action = torch.empty((batch_size, self.action.shape[1] if self.action.ndim > 1 and self.action.shape[1] > 0 else action_dim), device=self.device)
            dummy_next_state = torch.empty((batch_size, self.next_state.shape[1] if self.next_state.ndim > 1 and self.next_state.shape[1] > 0 else state_dim), device=self.device)
            dummy_reward = torch.empty((batch_size, 1), device=self.device)
            dummy_not_done = torch.empty((batch_size, 1), device=self.device)
            return dummy_state, dummy_action, dummy_next_state, dummy_reward, dummy_not_done
        indices = np.random.choice(self.num_samples, batch_size, replace=self.num_samples < batch_size)
        return (
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            self.reward[indices],
            self.not_done[indices],
        )

# --- 用户提供的 generate_data 函数 ---
def generate_data(num_total_samples, device='cpu',
                  inner_ring_modes=16,
                  outer_ring_modes=24,
                  ring_std_dev=0.015,
                  corner_std_dev=0.03,
                  radius_inner=0.5,
                  radius_outer=0.75,
                  clip_val=1.0,
                  reward_noise_std=0.1):
    all_samples_list = []
    inner_ring_pct = 0.35
    outer_ring_pct = 0.45
    n_inner_total = round(num_total_samples * inner_ring_pct)
    n_outer_total = round(num_total_samples * outer_ring_pct)
    n_corners_total = max(0, num_total_samples - n_inner_total - n_outer_total)
    current_sum = n_inner_total + n_outer_total + n_corners_total
    diff = num_total_samples - current_sum
    if diff != 0:
        if n_outer_total >= abs(diff): n_outer_total += diff
        elif n_inner_total >= abs(diff): n_inner_total += diff
        else: n_corners_total += diff

    if inner_ring_modes > 0 and n_inner_total > 0:
        base_inner, rem_inner = divmod(n_inner_total, inner_ring_modes)
        samples_counts_inner = [base_inner + 1] * rem_inner + [base_inner] * (inner_ring_modes - rem_inner)
        if sum(samples_counts_inner) != n_inner_total and n_inner_total > 0 : samples_counts_inner[0]+= n_inner_total - sum(samples_counts_inner)
        np.random.shuffle(samples_counts_inner)
        for i in range(inner_ring_modes):
            if samples_counts_inner[i] == 0: continue
            angle = 2 * np.pi * i / inner_ring_modes
            center_x, center_y = radius_inner * np.cos(angle), radius_inner * np.sin(angle)
            loc = torch.tensor([center_x, center_y], dtype=torch.float32, device=device)
            scale = torch.tensor([ring_std_dev, ring_std_dev], dtype=torch.float32, device=device)
            all_samples_list.append(Normal(loc, scale).sample((samples_counts_inner[i],)).clip(-clip_val, clip_val))

    if outer_ring_modes > 0 and n_outer_total > 0:
        base_outer, rem_outer = divmod(n_outer_total, outer_ring_modes)
        samples_counts_outer = [base_outer + 1] * rem_outer + [base_outer] * (outer_ring_modes - rem_outer)
        if sum(samples_counts_outer) != n_outer_total and n_outer_total > 0 : samples_counts_outer[0]+= n_outer_total - sum(samples_counts_outer)
        np.random.shuffle(samples_counts_outer)
        for i in range(outer_ring_modes):
            if samples_counts_outer[i] == 0: continue
            angle = 2 * np.pi * i / outer_ring_modes + (np.pi / outer_ring_modes)
            center_x, center_y = radius_outer * np.cos(angle), radius_outer * np.sin(angle)
            loc = torch.tensor([center_x, center_y], dtype=torch.float32, device=device)
            scale = torch.tensor([ring_std_dev, ring_std_dev], dtype=torch.float32, device=device)
            all_samples_list.append(Normal(loc, scale).sample((samples_counts_outer[i],)).clip(-clip_val, clip_val))

    num_modes_corners, pos_corner = 4, 0.9
    if num_modes_corners > 0 and n_corners_total > 0:
        base_corners, rem_corners = divmod(n_corners_total, num_modes_corners)
        samples_counts_corners = [base_corners + 1] * rem_corners + [base_corners] * (num_modes_corners - rem_corners)
        if sum(samples_counts_corners) != n_corners_total and n_corners_total > 0 : samples_counts_corners[0]+= n_corners_total - sum(samples_counts_corners)
        np.random.shuffle(samples_counts_corners)
        corner_coords = [[-pos_corner, pos_corner], [-pos_corner, -pos_corner], [pos_corner, pos_corner], [pos_corner, -pos_corner]]
        for i in range(num_modes_corners):
            if samples_counts_corners[i] == 0: continue
            loc = torch.tensor(corner_coords[i], dtype=torch.float32, device=device)
            scale = torch.tensor([corner_std_dev, corner_std_dev], dtype=torch.float32, device=device)
            all_samples_list.append(Normal(loc, scale).sample((samples_counts_corners[i],)).clip(-clip_val, clip_val))

    if not all_samples_list:
        if num_total_samples > 0:
            data = Normal(torch.zeros(2, device=device), torch.ones(2, device=device)*0.1).sample((num_total_samples,)).clip(-clip_val, clip_val)
        else:
            return Data_Sampler(torch.empty((0,2),dtype=torch.float32,device=device), torch.empty((0,2),dtype=torch.float32,device=device), torch.empty((0,1),dtype=torch.float32,device=device), device)
    else:
        data = torch.cat(all_samples_list, dim=0)
        if data.shape[0] != num_total_samples and num_total_samples > 0:
            if data.shape[0] > num_total_samples: data = data[:num_total_samples]
            else:
                extra_data = Normal(torch.zeros(2,dtype=torch.float32,device=device), torch.ones(2,dtype=torch.float32,device=device)*0.1).sample((num_total_samples-data.shape[0],)).clip(-clip_val,clip_val)
                data = torch.cat([data, extra_data], dim=0)

    action = data.to(dtype=torch.float32, device=device)
    state = torch.zeros_like(action, dtype=torch.float32, device=device)
    reward_peaks = [(0.8,-0.8,6.0,0.15,0.15), (0.0,radius_outer,3.5,0.2,0.2), (-radius_inner,0.0,2.0,0.15,0.15), (-0.7,0.7,1.0,0.2,0.2)]
    current_rewards = torch.zeros((action.shape[0],1), dtype=torch.float32, device=device)
    for mux,muy,amp,sigx,sigy in reward_peaks:
        current_rewards += (amp * torch.exp(-((action[:,0]-mux)**2/(2*sigx**2)) - ((action[:,1]-muy)**2/(2*sigy**2)))).unsqueeze(1)
    reward = current_rewards + reward_noise_std * torch.randn_like(current_rewards, device=device)
    return Data_Sampler(state, action, reward, device)

# --- 定义 MLP 和 QNetwork ---
class EpsilonThetaMLP(nn.Module):
    def __init__(self, action_dim_in, time_emb_dim_in, state_dim_in, hidden_dim_in, output_dim_in):
        super().__init__()
        input_size = action_dim_in + time_emb_dim_in + state_dim_in
        self.fc1 = nn.Linear(input_size, hidden_dim_in)
        self.fc2 = nn.Linear(hidden_dim_in, hidden_dim_in)
        self.fc3 = nn.Linear(hidden_dim_in, output_dim_in)
    def forward(self, a_k, t_emb, state):
        x = torch.cat([a_k.float(), t_emb.float(), state.float()], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SimpleQNetwork(nn.Module):
    def __init__(self, state_dim_in, action_dim_in, hidden_dim_in):
        super().__init__()
        input_size = state_dim_in + action_dim_in
        self.fc1 = nn.Linear(input_size, hidden_dim_in)
        self.fc2 = nn.Linear(hidden_dim_in, hidden_dim_in)
        self.fc3 = nn.Linear(hidden_dim_in, 1)
    def forward(self, state, action):
        x = torch.cat([state.float(), action.float()], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 定义 EDP Agent ---
class EDP_Agent:
    def __init__(self, state_dim_val, action_dim_val, epsilon_theta_model_inst, critic_model_class, max_action_val, device_val,
                 discount_val=0.99, tau_val=0.005, lambda_pi_loss_weight=1.0,
                 beta_schedule_str='linear', n_timesteps_diff=100, time_emb_dim_actor=None,
                 lr_actor_val=3e-4, lr_critic_val=3e-4, hidden_dim_q_net=256,
                 is_policy_likelihood_based=False):
        self.device = device_val
        self.discount = discount_val
        self.tau = tau_val
        self.lambda_policy_loss = lambda_pi_loss_weight
        self.is_likelihood_based = is_policy_likelihood_based
        self.max_action = max_action_val
        self.action_dim = action_dim_val
        self.actor = EDP(state_dim_val, action_dim_val, epsilon_theta_model_inst.float(),
                         max_action_val, beta_schedule_str, n_timesteps_diff,
                         time_embedding_dim=time_emb_dim_actor, device=device_val).to(device_val)
        self.actor_optimizer = torch.optim.Adam(self.actor.epsilon_theta_network.parameters(), lr=lr_actor_val)
        self.critic1 = critic_model_class(state_dim_val, action_dim_val, hidden_dim_q_net).to(device_val).float()
        self.critic1_target = copy.deepcopy(self.critic1).float()
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic_val)
        self.critic2 = critic_model_class(state_dim_val, action_dim_val, hidden_dim_q_net).to(device_val).float()
        self.critic2_target = copy.deepcopy(self.critic2).float()
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic_val)

    def train(self, data_sampler_inst, iterations_val, batch_size_val_train):
        total_actor_loss_epoch, total_critic_loss_epoch = 0, 0
        self.actor.epsilon_theta_network.train()
        self.critic1.train(); self.critic2.train()
        for _ in range(iterations_val):
            state, action, next_state, reward, not_done = data_sampler_inst.sample(batch_size_val_train)
            state,action,next_state,reward,not_done = state.to(self.device).float(),action.to(self.device).float(),next_state.to(self.device).float(),reward.to(self.device).float(),not_done.to(self.device).float()
            with torch.no_grad():
                next_action_sampled = self.actor.forward(next_state, evaluation_sampler_type='dpm_solver')
                target_Q1, target_Q2 = self.critic1_target(next_state, next_action_sampled), self.critic2_target(next_state, next_action_sampled)
                target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)
            current_Q1, current_Q2 = self.critic1(state, action), self.critic2(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic1_optimizer.zero_grad(); self.critic2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic1_optimizer.step(); self.critic2_optimizer.step()
            total_critic_loss_epoch += critic_loss.item()
            diffusion_loss_val = self.actor.diffusion_loss(action, state)
            if self.is_likelihood_based:
                with torch.no_grad(): adv_weights_f_Q = torch.exp(self.critic1(state, action).detach() / 1.0)
                policy_loss_val = self.actor.edp_likelihood_based_policy_loss(state, action, self.critic1, adv_weights_f_Q)
            else:
                k_uniform = torch.randint(0, self.actor.n_timesteps, (state.shape[0],), device=self.device).long()
                a_hat_0, _, _ = self.actor.action_approximation(state, action, k_uniform)
                policy_loss_val = -self.critic1(state, a_hat_0).mean()
            actor_loss = diffusion_loss_val + self.lambda_policy_loss * policy_loss_val
            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
            total_actor_loss_epoch += actor_loss.item()
            self._update_target_networks()
        return total_actor_loss_epoch/iterations_val, total_critic_loss_epoch/iterations_val

    def _update_target_networks(self):
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()): target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()): target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def sample_actions_for_eval(self, state_val):
        self.actor.epsilon_theta_network.eval(); self.critic1.eval()
        with torch.no_grad():
            sampled_acts = self.actor.forward(state_val.to(self.device).float(), q_function_for_eas=self.critic1, eas_num_samples=10, evaluation_sampler_type='dpm_solver')
        return sampled_acts.clamp_(-self.max_action, self.max_action)

# --- 实例化组件 ---
epsilon_theta_net = EpsilonThetaMLP(action_dim, time_embedding_dim_for_edp, state_dim, hidden_mlp_dim, action_dim).to(device).float()
total_dataset_samples = 10000
print(f"Generating training data ({total_dataset_samples} samples)...")
custom_sampler = generate_data(num_total_samples=total_dataset_samples, device=device, inner_ring_modes=12, outer_ring_modes=16, ring_std_dev=0.02, corner_std_dev=0.04, radius_inner=0.45, radius_outer=0.85, clip_val=max_action, reward_noise_std=0.05)
edp_test_agent = EDP_Agent(state_dim,action_dim,epsilon_theta_net,SimpleQNetwork,max_action,device,discount,tau,lambda_policy_loss_weight,beta_schedule_name,diffusion_timesteps_K,time_embedding_dim_for_edp,learning_rate,learning_rate,hidden_mlp_dim,False)

# --- 训练循环 ---
print(f"Starting EDP training for {num_epochs} epochs (K={diffusion_timesteps_K})...")
actor_losses, critic_losses = [], []
if custom_sampler.num_samples > 0:
    for i in range(1, num_epochs + 1):
        avg_actor_loss, avg_critic_loss = edp_test_agent.train(custom_sampler, iterations_per_epoch, batch_size_data)
        actor_losses.append(avg_actor_loss); critic_losses.append(avg_critic_loss)
        if i % max(1, num_epochs // 20) == 0: print(f'Epoch: {i}/{num_epochs} | Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}')
    print("EDP training finished.")
else: print("Skipping training: no data.")

# --- 评估和绘图 ---
eval_state_for_custom_data = torch.zeros((num_eval_samples, state_dim), device=device).float()
final_actions_tensor = edp_test_agent.sample_actions_for_eval(eval_state_for_custom_data)
final_actions_numpy = final_actions_tensor.detach().cpu().numpy()

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor('white')

# 子图1: 训练数据的真实动作分布，根据奖励值着色
if custom_sampler.num_samples > 0:
    training_actions_plot = custom_sampler.action.detach().cpu().numpy()
    training_rewards_plot = custom_sampler.reward.detach().cpu().numpy().flatten()
    scatter_train = axs[0].scatter(training_actions_plot[:, 0], training_actions_plot[:, 1], c=training_rewards_plot, cmap='viridis', alpha=0.6, s=15 )
    cbar = fig.colorbar(scatter_train, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('True Reward Value', rotation=270, labelpad=15)
else:
    axs[0].text(0.5, 0.5, "No training data generated", ha='center', va='center')
axs[0].set_title('Training Data: Actions Colored by Reward')
axs[0].set_xlabel('Action Dimension 1')
axs[0].set_ylabel('Action Dimension 2')
axs[0].set_xlim(-plot_axis_limit, plot_axis_limit)
axs[0].set_ylim(-plot_axis_limit, plot_axis_limit)
# axs[0].legend(loc='upper right') # 颜色条已经解释了颜色含义，图例可能不需要
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].set_aspect('equal', adjustable='box')

# 子图2: EDP Agent 生成的动作
axs[1].scatter(final_actions_numpy[:, 0], final_actions_numpy[:, 1], alpha=0.3, color='#d62728', label='EDP Sampled Actions', s=15)
axs[1].set_title(f'EDP Sampled Actions (K={diffusion_timesteps_K}, EAS)')
axs[1].set_xlabel('Action Dimension 1')
axs[1].set_ylabel('Action Dimension 2')
axs[1].set_xlim(-plot_axis_limit, plot_axis_limit)
axs[1].set_ylim(-plot_axis_limit, plot_axis_limit)
axs[1].legend(loc='upper right')
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].set_aspect('equal', adjustable='box')

# 子图3: 损失曲线
if actor_losses and critic_losses:
    epochs_range = range(1, len(actor_losses) + 1)
    axs[2].plot(epochs_range, actor_losses, label='Actor Loss', color='purple', linewidth=2)
    axs[2].plot(epochs_range, critic_losses, label='Critic Loss', color='green', linewidth=2)
    axs[2].set_title('Training Losses')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.7)

    # 修改Y轴刻度设置逻辑
    # 仅当所有损失值（actor 和 critic）都严格为正时，才使用对数刻度
    all_loss_values = [val for val_list in [actor_losses, critic_losses] for val in val_list if val is not None]
    if all_loss_values and all(val > 1e-9 for val in all_loss_values): # 确保列表非空且所有值>0
        try:
            axs[2].set_yscale('log')
        except ValueError: # 以防万一（例如，如果所有值都非常接近0，某些matplotlib版本可能会有问题）
            axs[2].set_yscale('linear') # 出错则退回线性刻度
    else:
        axs[2].set_yscale('linear') # 如果有非正值，则使用线性刻度

else:
    axs[2].text(0.5, 0.5, "No training or no losses recorded", ha='center', va='center')

plt.suptitle('EDP Training and Evaluation on Custom Multi-Modal Data', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("EDP custom data test plotting complete.")