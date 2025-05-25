import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.distributions import Normal # Still needed if EDP or other parts use it internally

# Assuming EDP.py is in the toy_experiments subdirectory
from toy_experiments.EDP import EDP

# --- Simulation Parameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Based on new generate_data, state = torch.zeros_like(action), action is 2D
state_dim = 2
action_dim = 2
max_action = 1.0 # The new generate_data implicitly clips R to [0,1], so actions are within roughly a unit circle
discount = 0.0 # Bandit problem, single step reward matters
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
plot_axis_limit = 1.1 # Adjusted for unit circle data

# --- Data_Sampler Class ---
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
            print(f"  Action Min: {self.action.min(dim=0)[0].cpu().numpy()}, Max: {self.action.max(dim=0)[0].cpu().numpy()}")
            print(f"  Reward Min: {self.reward.min().item():.3f}, Max: {self.reward.max().item():.3f}, Mean: {self.reward.mean().item():.3f}")
        else:
            print("Warning: Data_Sampler received 0 samples!")

    def sample(self, batch_size):
        if self.num_samples == 0:
            print("Warning: Attempting to sample from an empty Data_Sampler.")
            # Return appropriately shaped empty tensors
            s_shape = self.state.shape[1] if self.state.ndim > 1 and self.state.shape[1] > 0 else state_dim
            a_shape = self.action.shape[1] if self.action.ndim > 1 and self.action.shape[1] > 0 else action_dim
            ns_shape = self.next_state.shape[1] if self.next_state.ndim > 1 and self.next_state.shape[1] > 0 else state_dim

            dummy_state = torch.empty((batch_size, s_shape), device=self.device)
            dummy_action = torch.empty((batch_size, a_shape), device=self.device)
            dummy_next_state = torch.empty((batch_size, ns_shape), device=self.device)
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

# --- New generate_data Function (Spiral Dataset) ---
def generate_data(num_total_samples, device='cpu',
                  theta_R_coefficient=None,
                  reward_sin_amplitude=0.5,
                  reward_sin_frequency_on_R=10.0 * np.pi, # Corrected to use np.pi
                  reward_sin_phase=0.0,
                  reward_sin_offset=0.5):

    if theta_R_coefficient is None:
        theta_R_coefficient = (6.0 * np.pi) # Corrected to use np.pi

    if num_total_samples <= 0:
        action_empty = torch.empty((0, 2), dtype=torch.float32, device=device)
        state_empty = torch.empty((0, 2), dtype=torch.float32, device=device)
        calculated_reward_empty = torch.empty((0, 1), dtype=torch.float32, device=device)
        return Data_Sampler(state=state_empty, action=action_empty, reward=calculated_reward_empty, device=device)

    R = torch.rand(num_total_samples, device=device, dtype=torch.float32) # R is in [0, 1]
    theta = theta_R_coefficient * R
    x = R * torch.cos(theta)
    y = R * torch.sin(theta)

    action = torch.stack((x, y), dim=1)

    state = torch.zeros_like(action, dtype=torch.float32, device=device)
    calculated_reward = reward_sin_amplitude * torch.sin(
        reward_sin_frequency_on_R * R + reward_sin_phase
    ) + reward_sin_offset

    calculated_reward = calculated_reward.unsqueeze(1)

    return Data_Sampler(state=state, action=action, reward=calculated_reward, device=device)


# --- Neural Network Definitions (EpsilonThetaMLP, SimpleQNetwork) ---
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

# --- EDP Agent Definition ---
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

# --- Instantiate Components ---
# 1. Epsilon-Theta Network
epsilon_theta_net = EpsilonThetaMLP(action_dim, time_embedding_dim_for_edp, state_dim, hidden_mlp_dim, action_dim).to(device).float()

# 2. Data Sampler using the new generate_data function
total_dataset_samples = 10000
print(f"Generating training data (Spiral Dataset - {total_dataset_samples} samples)...")
custom_sampler = generate_data(
    num_total_samples=total_dataset_samples,
    device=device,
    # Parameters for the spiral data and its reward
    theta_R_coefficient=8.0 * np.pi, # Makes a denser spiral
    reward_sin_amplitude=1.0,
    reward_sin_frequency_on_R=12.0 * np.pi,
    reward_sin_phase=np.pi/2,
    reward_sin_offset=1.0
)

# 3. EDP Agent
edp_test_agent = EDP_Agent(state_dim,action_dim,epsilon_theta_net,SimpleQNetwork,max_action,device,discount,tau,lambda_policy_loss_weight,beta_schedule_name,diffusion_timesteps_K,time_embedding_dim_for_edp,learning_rate,learning_rate,hidden_mlp_dim,False)

# --- Training Loop ---
print(f"Starting EDP training on Spiral Dataset for {num_epochs} epochs (K={diffusion_timesteps_K})...")
actor_losses, critic_losses = [], []
if custom_sampler.num_samples > 0:
    for i in range(1, num_epochs + 1):
        avg_actor_loss, avg_critic_loss = edp_test_agent.train(custom_sampler, iterations_per_epoch, batch_size_data)
        actor_losses.append(avg_actor_loss); critic_losses.append(avg_critic_loss)
        if i % max(1, num_epochs // 20) == 0: print(f'Epoch: {i}/{num_epochs} | Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}')
    print("EDP training on Spiral Dataset finished.")
else: print("Skipping training: no data generated by Data_Sampler.")

# --- Evaluation and Plotting ---
eval_state_for_custom_data = torch.zeros((num_eval_samples, state_dim), device=device).float()
final_actions_tensor = edp_test_agent.sample_actions_for_eval(eval_state_for_custom_data)
final_actions_numpy = final_actions_tensor.detach().cpu().numpy()

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor('white')

# Subplot 1: Training data actions colored by true reward
if custom_sampler.num_samples > 0:
    training_actions_plot = custom_sampler.action.detach().cpu().numpy()
    training_rewards_plot = custom_sampler.reward.detach().cpu().numpy().flatten()
    scatter_train = axs[0].scatter(training_actions_plot[:, 0], training_actions_plot[:, 1], c=training_rewards_plot, cmap='viridis', alpha=0.5, s=10 )
    cbar = fig.colorbar(scatter_train, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('True Reward Value', rotation=270, labelpad=15)
else:
    axs[0].text(0.5, 0.5, "No training data generated", ha='center', va='center')
axs[0].set_title('Training Data (Spiral): Actions Colored by Reward')
axs[0].set_xlabel('Action Dimension 1')
axs[0].set_ylabel('Action Dimension 2')
axs[0].set_xlim(-plot_axis_limit, plot_axis_limit)
axs[0].set_ylim(-plot_axis_limit, plot_axis_limit)
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].set_aspect('equal', adjustable='box')

# Subplot 2: EDP Agent generated actions
axs[1].scatter(final_actions_numpy[:, 0], final_actions_numpy[:, 1], alpha=0.3, color='#d62728', label='EDP Sampled Actions', s=15)
axs[1].set_title(f'EDP Sampled Actions (K={diffusion_timesteps_K}, EAS)')
axs[1].set_xlabel('Action Dimension 1')
axs[1].set_ylabel('Action Dimension 2')
axs[1].set_xlim(-plot_axis_limit, plot_axis_limit)
axs[1].set_ylim(-plot_axis_limit, plot_axis_limit)
axs[1].legend(loc='upper right')
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].set_aspect('equal', adjustable='box')

# Subplot 3: Loss curves
if actor_losses and critic_losses:
    epochs_range = range(1, len(actor_losses) + 1)
    axs[2].plot(epochs_range, actor_losses, label='Actor Loss', color='purple', linewidth=2)
    axs[2].plot(epochs_range, critic_losses, label='Critic Loss', color='green', linewidth=2)
    axs[2].set_title('Training Losses')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.7)
    all_loss_values = []
    if actor_losses: all_loss_values.extend([val for val in actor_losses if val is not None])
    if critic_losses: all_loss_values.extend([val for val in critic_losses if val is not None])
    if all_loss_values and all(val > 1e-9 for val in all_loss_values):
        try: axs[2].set_yscale('log')
        except ValueError: axs[2].set_yscale('linear')
    else: axs[2].set_yscale('linear')
else:
    axs[2].text(0.5, 0.5, "No training or no losses recorded", ha='center', va='center')

plt.suptitle('EDP Training on Spiral Dataset', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("EDP spiral data test plotting complete.")