import os
import torch
import numpy as np
from torch.distributions import Normal
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn # Added for nn.Module and nn.Functional
import torch.nn.functional as F # Added for F.relu etc.
import copy # Added for deepcopy
import math # Added for math.log etc.

# Assuming EDP class is in toy_experiments/EDP.py
from toy_experiments.EDP import EDP # This needs to be your EDP diffusion model class

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=2024, type=int) # Updated seed for new run
parser.add_argument("--exp_name", default='edp_spiral_test', type=str) # Experiment name
parser.add_argument("--eta", default=1.0, type=float) # For EDP, this is lambda_policy_loss_weight
parser.add_argument('--device_id', default=0, type=int) # For GPU selection if needed
parser.add_argument("--results_dir", default='toy_results_edp', type=str) # Directory for saving images
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument('--hidden_dim', default=256, type=int) # Increased hidden_dim

args = parser.parse_args()

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device_id}")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(args.device_id)}")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")
print(f"Selected device: {device}")

# --- Parameters ---
lambda_policy_loss_weight = args.eta
seed = args.seed
lr = args.lr
hidden_dim = args.hidden_dim

state_dim = 2
action_dim = 2
max_action = 1.0
discount = 0.0 # For Bandit problem
tau = 0.005
beta_schedule_name = 'linear' # Or 'vp' / 'cosine' as used in EDP
diffusion_timesteps_K = 50  # K for EDP

num_epochs = 200
num_data_points = 10000
batch_size = 128 # Adjusted batch size
iterations_per_epoch = int(num_data_points / batch_size)

img_dir = os.path.join(args.results_dir, args.exp_name)
os.makedirs(img_dir, exist_ok=True)

num_eval_plot_samples = 1000
plot_axis_limit = 1.1
time_embedding_dim_for_edp = 32


# --- Data_Sampler Class (as per your generate_data return type) ---
class Data_Sampler:
    def __init__(self, state, action, reward, device): # Matched to keyword args
        self.state = state.to(device).float()
        self.action = action.to(device).float()
        self.reward = reward.to(device).float()
        self.num_samples = self.state.shape[0]
        self.device = device
        self.next_state = state.clone().to(device).float() # For bandit, next_state can be same as state
        self.not_done = torch.ones(self.num_samples, 1, device=device).float()

        print(f"Data_Sampler initialized with {self.num_samples} samples on {self.device}.")
        if self.num_samples > 0:
            print(f"  Action Min: {self.action.min(dim=0)[0].cpu().numpy()}, Max: {self.action.max(dim=0)[0].cpu().numpy()}")
            print(f"  Action Mean: {self.action.mean(dim=0).cpu().numpy()}, Std: {self.action.std(dim=0).cpu().numpy()}")
            print(f"  Reward Min: {self.reward.min().item():.3f}, Max: {self.reward.max().item():.3f}, Mean: {self.reward.mean().item():.3f}")
        else:
            print("Warning: Data_Sampler received 0 samples!")

    def sample(self, current_batch_size):
        if self.num_samples == 0:
            print("Warning: Attempting to sample from an empty Data_Sampler.")
            s_shape = self.state.shape[1] if self.state.ndim > 1 and self.state.shape[1] > 0 else state_dim
            a_shape = self.action.shape[1] if self.action.ndim > 1 and self.action.shape[1] > 0 else action_dim
            ns_shape = self.next_state.shape[1] if self.next_state.ndim > 1 and self.next_state.shape[1] > 0 else state_dim
            return (torch.empty((current_batch_size, s_shape), device=self.device),
                    torch.empty((current_batch_size, a_shape), device=self.device),
                    torch.empty((current_batch_size, ns_shape), device=self.device),
                    torch.empty((current_batch_size, 1), device=self.device),
                    torch.empty((current_batch_size, 1), device=self.device))
        indices = np.random.choice(self.num_samples, current_batch_size, replace=self.num_samples < current_batch_size)
        return (self.state[indices], self.action[indices], self.next_state[indices],
                self.reward[indices], self.not_done[indices])


# --- generate_data Function (Spiral Dataset - User Provided) ---
def generate_data(num_total_samples, device='cpu',
                  theta_R_coefficient=None,
                  reward_sin_amplitude=0.5,
                  reward_sin_frequency_on_R=10.0 * np.pi,
                  reward_sin_phase=0.0,
                  reward_sin_offset=0.5):
    if theta_R_coefficient is None: theta_R_coefficient = (6.0 * np.pi)
    if num_total_samples <= 0:
        return Data_Sampler(state=torch.empty((0,2),d=device), action=torch.empty((0,2),d=device), reward=torch.empty((0,1),d=device), device=device)

    R = torch.rand(num_total_samples, device=device, dtype=torch.float32)
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


# --- Neural Network Definitions ---
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
                 discount_val, tau_val, lambda_pi_loss_weight_val,
                 beta_schedule_str, n_timesteps_diff_val, time_emb_dim_actor_val,
                 lr_actor_val, lr_critic_val, hidden_dim_q_net_val,
                 is_policy_likelihood_based_val=False):
        self.device = device_val
        self.discount = discount_val
        self.tau = tau_val
        self.lambda_policy_loss = lambda_pi_loss_weight_val
        self.is_likelihood_based = is_policy_likelihood_based_val
        self.max_action = max_action_val
        self.actor = EDP(state_dim_val, action_dim_val, epsilon_theta_model_inst.float(),
                         max_action_val, beta_schedule_str, n_timesteps_diff_val,
                         time_embedding_dim=time_emb_dim_actor_val, device=device_val).to(device_val)
        self.actor_optimizer = torch.optim.Adam(self.actor.epsilon_theta_network.parameters(), lr=lr_actor_val)
        self.critic1 = critic_model_class(state_dim_val, action_dim_val, hidden_dim_q_net_val).to(device_val).float()
        self.critic1_target = copy.deepcopy(self.critic1).float()
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic_val)
        self.critic2 = critic_model_class(state_dim_val, action_dim_val, hidden_dim_q_net_val).to(device_val).float()
        self.critic2_target = copy.deepcopy(self.critic2).float()
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic_val)

    def train(self, data_sampler_inst, current_iterations, current_batch_size):
        total_actor_loss_epoch, total_critic_loss_epoch = 0.0, 0.0
        self.actor.epsilon_theta_network.train()
        self.critic1.train(); self.critic2.train()
        for _ in range(current_iterations):
            state_b, action_b, next_state_b, reward_b, not_done_b = data_sampler_inst.sample(current_batch_size)
            state_b,action_b,next_state_b,reward_b,not_done_b = state_b.to(self.device).float(),action_b.to(self.device).float(),next_state_b.to(self.device).float(),reward_b.to(self.device).float(),not_done_b.to(self.device).float()
            with torch.no_grad():
                next_action_sampled = self.actor.forward(next_state_b, evaluation_sampler_type='dpm_solver')
                target_Q1, target_Q2 = self.critic1_target(next_state_b, next_action_sampled), self.critic2_target(next_state_b, next_action_sampled)
                target_Q = reward_b + not_done_b * self.discount * torch.min(target_Q1, target_Q2)
            current_Q1, current_Q2 = self.critic1(state_b, action_b), self.critic2(state_b, action_b)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic1_optimizer.zero_grad(); self.critic2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic1_optimizer.step(); self.critic2_optimizer.step()
            total_critic_loss_epoch += critic_loss.item()
            diffusion_loss_val = self.actor.diffusion_loss(action_b, state_b)
            if self.is_likelihood_based:
                with torch.no_grad(): adv_weights_f_Q = torch.exp(self.critic1(state_b, action_b).detach() / 1.0)
                policy_loss_val = self.actor.edp_likelihood_based_policy_loss(state_b, action_b, self.critic1, adv_weights_f_Q)
            else:
                k_uniform = torch.randint(0, self.actor.n_timesteps, (state_b.shape[0],), device=self.device).long()
                a_hat_0, _, _ = self.actor.action_approximation(state_b, action_b, k_uniform)
                policy_loss_val = -self.critic1(state_b, a_hat_0).mean()
            actor_loss = diffusion_loss_val + self.lambda_policy_loss * policy_loss_val
            self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
            total_actor_loss_epoch += actor_loss.item()
            self._update_target_networks()
        return total_actor_loss_epoch/current_iterations, total_critic_loss_epoch/current_iterations

    def _update_target_networks(self):
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()): target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()): target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def sample_actions_for_eval(self, state_val):
        self.actor.epsilon_theta_network.eval(); self.critic1.eval()
        with torch.no_grad():
            sampled_acts = self.actor.forward(state_val.to(self.device).float(), q_function_for_eas=self.critic1, eas_num_samples=10, evaluation_sampler_type='dpm_solver')
        return sampled_acts.clamp_(-self.max_action, self.max_action)

# --- Main Script Execution ---
torch.manual_seed(seed)
np.random.seed(seed)

# 1. Instantiate Epsilon-Theta Network
epsilon_theta_model = EpsilonThetaMLP(action_dim, time_embedding_dim_for_edp, state_dim, hidden_dim, action_dim).to(device).float()

# 2. Generate Data using the new function
print(f"Generating training data (Spiral Dataset - {num_data_points} samples)...")
data_sampler_instance = generate_data(
    num_total_samples=num_data_points,
    device=device,
    theta_R_coefficient=7.0 * np.pi, # Example: denser spiral
    reward_sin_amplitude=1.0,
    reward_sin_frequency_on_R=15.0 * np.pi, # More oscillations
    reward_sin_offset=1.0
)

# 3. Instantiate EDP Agent
edp_agent_instance = EDP_Agent(
    state_dim_val=state_dim, action_dim_val=action_dim,
    epsilon_theta_model_inst=epsilon_theta_model,
    critic_model_class=SimpleQNetwork, max_action_val=max_action, device_val=device,
    discount_val=discount, tau_val=tau, lambda_pi_loss_weight_val=lambda_policy_loss_weight,
    beta_schedule_str=beta_schedule_name, n_timesteps_diff_val=diffusion_timesteps_K,
    time_emb_dim_actor_val=time_embedding_dim_for_edp,
    lr_actor_val=lr, lr_critic_val=lr, hidden_dim_q_net_val=hidden_dim,
    is_policy_likelihood_based_val=False
)

# 4. Training Loop
print(f"Starting EDP training on Spiral Dataset for {num_epochs} epochs (K={diffusion_timesteps_K})...")
actor_loss_history, critic_loss_history = [], []
if data_sampler_instance.num_samples > 0:
    for epoch in range(1, num_epochs + 1):
        avg_actor_loss, avg_critic_loss = edp_agent_instance.train(data_sampler_instance, iterations_per_epoch, batch_size_data)
        actor_loss_history.append(avg_actor_loss); critic_loss_history.append(avg_critic_loss)
        if epoch % max(1, num_epochs // 20) == 0:
            print(f'Epoch: {epoch}/{num_epochs} | Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}')
    print("EDP training on Spiral Dataset finished.")
else:
    print("Skipping training: no data generated by Data_Sampler.")

# 5. Evaluation and Plotting
eval_states = torch.zeros((num_eval_plot_samples, state_dim), device=device).float()
sampled_actions_tensor = edp_agent_instance.sample_actions_for_eval(eval_states)
sampled_actions_numpy = sampled_actions_tensor.detach().cpu().numpy()

fig, axs = plt.subplots(1, 3, figsize=(21, 6.5)) # Adjusted figure size
fig.patch.set_facecolor('white')

# Plot 1: Ground Truth Data (Actions colored by Reward)
if data_sampler_instance.num_samples > 0:
    gt_actions_np = data_sampler_instance.action.cpu().numpy()
    gt_rewards_np = data_sampler_instance.reward.cpu().numpy().flatten()
    sc_gt = axs[0].scatter(gt_actions_np[:, 0], gt_actions_np[:, 1], c=gt_rewards_np, cmap='viridis', alpha=0.4, s=12)
    cb_gt = fig.colorbar(sc_gt, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04)
    cb_gt.set_label('True Reward Value', rotation=270, labelpad=15)
else:
    axs[0].text(0.5, 0.5, "No training data generated", ha='center', va='center', color='red', transform=axs[0].transAxes)
axs[0].set_title('Ground Truth: Actions Colored by Reward (Spiral)', fontsize=14)
axs[0].set_xlabel('Action Dim 1', fontsize=12)
axs[0].set_ylabel('Action Dim 2', fontsize=12)
axs[0].set_xlim(-plot_axis_limit, plot_axis_limit); axs[0].set_ylim(-plot_axis_limit, plot_axis_limit)
axs[0].grid(True, linestyle='--', alpha=0.6); axs[0].set_aspect('equal', adjustable='box')

# Plot 2: EDP Agent Sampled Actions
axs[1].scatter(sampled_actions_numpy[:, 0], sampled_actions_numpy[:, 1], alpha=0.3, color='#d62728', s=15, label='EDP Sampled Actions')
axs[1].set_title(f'EDP Sampled Actions (K={diffusion_timesteps_K}, EAS)', fontsize=14)
axs[1].set_xlabel('Action Dim 1', fontsize=12); axs[1].set_ylabel('Action Dim 2', fontsize=12)
axs[1].set_xlim(-plot_axis_limit, plot_axis_limit); axs[1].set_ylim(-plot_axis_limit, plot_axis_limit)
axs[1].grid(True, linestyle='--', alpha=0.6); axs[1].set_aspect('equal', adjustable='box'); axs[1].legend(loc='upper right')

# Plot 3: Loss Curves
if actor_loss_history and critic_loss_history:
    epochs_plot_range = range(1, len(actor_loss_history) + 1)
    axs[2].plot(epochs_plot_range, actor_loss_history, label='Actor Loss', color='purple', linewidth=1.5)
    axs[2].plot(epochs_plot_range, critic_loss_history, label='Critic Loss', color='green', linewidth=1.5)
    axs[2].set_title('Training Losses', fontsize=14)
    axs[2].set_xlabel('Epoch', fontsize=12); axs[2].set_ylabel('Loss', fontsize=12)
    axs[2].legend(); axs[2].grid(True, linestyle='--', alpha=0.6)
    all_plot_losses = [l for l_list in [actor_loss_history, critic_loss_history] for l in l_list if l is not None]
    if all_plot_losses and all(val > 1e-9 for val in all_plot_losses):
        try: axs[2].set_yscale('log')
        except ValueError: axs[2].set_yscale('linear')
    else: axs[2].set_yscale('linear')
else:
    axs[2].text(0.5, 0.5, "No losses recorded", ha='center', va='center', transform=axs[2].transAxes)

fig.suptitle(f'EDP Agent on Spiral Dataset (Seed: {seed}, K={diffusion_timesteps_K}, λ={lambda_policy_loss_weight})', fontsize=16, y=0.99)
plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # Adjust rect to make space for suptitle
file_name = f'edp_spiral_K{diffusion_timesteps_K}_lambda{lambda_policy_loss_weight}_seed{seed}.pdf'
plt.savefig(os.path.join(img_dir, file_name))
print(f"Plot saved to {os.path.join(img_dir, file_name)}")
plt.show()

print("EDP spiral data test script finished.")