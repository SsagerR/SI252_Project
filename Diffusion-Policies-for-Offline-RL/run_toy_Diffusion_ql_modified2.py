import os
import torch
import numpy as np
from torch.distributions import Normal
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from toy_experiments.toy_helpers import Data_Sampler

parser = argparse.ArgumentParser()
parser.add_argument("--ill", action='store_true')
parser.add_argument("--seed", default=2022, type=int)
parser.add_argument("--exp", default='exp_1', type=str)
parser.add_argument("--x", default=0., type=float)
parser.add_argument("--y", default=0., type=float)
parser.add_argument("--eta", default=2.5, type=float)
parser.add_argument('--device', default=0, type=int)
parser.add_argument("--dir", default='whole_grad', type=str)
parser.add_argument("--r_fun", default='no', type=str)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument("--mode", default='whole_grad', type=str)
args = parser.parse_args()

r_fun_std = 0.25
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")
print(f"Selected device: {device}")

eta = args.eta
seed = args.seed
lr = args.lr
hidden_dim = args.hidden_dim


def generate_data(num_total_samples, device='cpu',
                  theta_R_coefficient=None,
                  reward_sin_amplitude=0.5,
                  reward_sin_frequency_on_R=10.0 * np.pi,
                  reward_sin_phase=0.0,
                  reward_sin_offset=0.5):

    if theta_R_coefficient is None:
        theta_R_coefficient = (6.0 * np.pi)

    if num_total_samples <= 0:
        action = torch.empty((0, 2), dtype=torch.float32, device=device)
        state = torch.empty((0, 2), dtype=torch.float32, device=device)
        calculated_reward = torch.empty((0, 1), dtype=torch.float32, device=device)
        return Data_Sampler(state=state, action=action, reward=calculated_reward, device=device)

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


torch.manual_seed(seed)
np.random.seed(seed)

num_data = int(10000)
data_sampler = generate_data(num_data, device)

state_dim = 2
action_dim = 2
max_action = 1.0

discount = 0.99
tau = 0.005
model_type = 'MLP'

T = 50
beta_schedule = 'vp'
# hidden_dim = 64
# eta = 10.0
# lr = 3e-4

num_epochs = 200
batch_size = 100
iterations = int(num_data / batch_size)

img_dir = f'toy_imgs/{args.dir}'
os.makedirs(img_dir, exist_ok=True)

num_eval = 100

fig, axs = plt.subplots(1, 5, figsize=(5.5 * 5, 5))
axis_lim = 1.1

axs[0].clear()
num_samples_for_axs0_display = num_data
data_sampler_for_axs0 = data_sampler

if data_sampler_for_axs0 is not None and \
   hasattr(data_sampler_for_axs0, 'action') and data_sampler_for_axs0.action.nelement() > 0 and \
   hasattr(data_sampler_for_axs0, 'reward') and data_sampler_for_axs0.reward.nelement() > 0 and \
   data_sampler_for_axs0.action.shape[0] == data_sampler_for_axs0.reward.shape[0]:

    action_samples_np = data_sampler_for_axs0.action.cpu().numpy()
    reward_values_np = data_sampler_for_axs0.reward.cpu().numpy().flatten()

    scatter_plot_on_axs0 = axs[0].scatter(
        action_samples_np[:, 0],
        action_samples_np[:, 1],
        c=reward_values_np,
        cmap='viridis',
        alpha=0.5,
        s=10
    )
    fig.colorbar(scatter_plot_on_axs0, ax=axs[0], label='Reward Value')

else:
    axs[0].text(0.5, 0.5, "Data for Ground Truth plot (axs[0]) is invalid or empty.",
                ha='center', va='center', color='red', transform=axs[0].transAxes)

axs[0].set_title('Ground Truth: Actions Colored by Reward', fontsize=15)
axs[0].set_xlabel('Action_x', fontsize=12)
axs[0].set_ylabel('Action_y', fontsize=12)
axs[0].set_xlim(-axis_lim, axis_lim)
axs[0].set_ylim(-axis_lim, axis_lim)
axs[0].set_aspect('equal', adjustable='box')
axs[0].grid(True, linestyle='--', alpha=0.7)

# Plot QL-Diffusion
from toy_experiments.ql_diffusion import QL_Diffusion

agent = QL_Diffusion(state_dim=state_dim,
                     action_dim=action_dim,
                     max_action=max_action,
                     device=device,
                     discount=discount,
                     tau=tau,
                     eta=eta,
                     beta_schedule=beta_schedule,
                     n_timesteps=2,
                     model_type=model_type,
                     hidden_dim=hidden_dim,
                     lr=lr,
                     r_fun=None,
                     mode=args.mode)


for i in range(1, num_epochs+1):

    b_loss, q_loss = agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f'QL-Diffusion Epoch: {i} B_loss {b_loss} Q_loss {q_loss}')

# fig, ax = plt.subplots()
new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[1].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3, color='#d62728')
axs[1].set_xlim(-axis_lim, axis_lim)
axs[1].set_ylim(-axis_lim, axis_lim)
axs[1].set_xlabel('x', fontsize=20)
axs[1].set_ylabel('y', fontsize=20)
axs[1].set_title('Diffusion-QL N=2', fontsize=25)

# Plot QL-Diffusion
from toy_experiments.ql_diffusion import QL_Diffusion

agent = QL_Diffusion(state_dim=state_dim,
                     action_dim=action_dim,
                     max_action=max_action,
                     device=device,
                     discount=discount,
                     tau=tau,
                     eta=eta,
                     beta_schedule=beta_schedule,
                     n_timesteps=5,
                     model_type=model_type,
                     hidden_dim=hidden_dim,
                     lr=lr,
                     r_fun=None,
                     mode=args.mode)


for i in range(1, num_epochs+1):

    b_loss, q_loss = agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f'QL-Diffusion Epoch: {i} B_loss {b_loss} Q_loss {q_loss}')

# fig, ax = plt.subplots()
new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[2].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3, color='#d62728')
axs[2].set_xlim(-axis_lim, axis_lim)
axs[2].set_ylim(-axis_lim, axis_lim)
axs[2].set_xlabel('x', fontsize=20)
axs[2].set_ylabel('y', fontsize=20)
axs[2].set_title('Diffusion-QL N=5', fontsize=25)

# Plot QL-Diffusion
from toy_experiments.ql_diffusion import QL_Diffusion

agent = QL_Diffusion(state_dim=state_dim,
                     action_dim=action_dim,
                     max_action=max_action,
                     device=device,
                     discount=discount,
                     tau=tau,
                     eta=eta,
                     beta_schedule=beta_schedule,
                     n_timesteps=10,
                     model_type=model_type,
                     hidden_dim=hidden_dim,
                     lr=lr,
                     r_fun=None,
                     mode=args.mode)


for i in range(1, num_epochs+1):

    b_loss, q_loss = agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f'QL-Diffusion Epoch: {i} B_loss {b_loss} Q_loss {q_loss}')

# fig, ax = plt.subplots()
new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[3].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3, color='#d62728')
axs[3].set_xlim(-axis_lim, axis_lim)
axs[3].set_ylim(-axis_lim, axis_lim)
axs[3].set_xlabel('x', fontsize=20)
axs[3].set_ylabel('y', fontsize=20)
axs[3].set_title('Diffusion-QL N=10', fontsize=25)


# Plot QL-Diffusion
from toy_experiments.ql_diffusion import QL_Diffusion

agent = QL_Diffusion(state_dim=state_dim,
                     action_dim=action_dim,
                     max_action=max_action,
                     device=device,
                     discount=discount,
                     tau=tau,
                     eta=eta,
                     beta_schedule=beta_schedule,
                     n_timesteps=50,
                     model_type=model_type,
                     hidden_dim=hidden_dim,
                     lr=lr,
                     r_fun=None,
                     mode=args.mode)


for i in range(1, num_epochs+1):

    b_loss, q_loss = agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f'QL-Diffusion Epoch: {i} B_loss {b_loss} Q_loss {q_loss}')

# fig, ax = plt.subplots()
new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[4].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3, color='#d62728')
axs[4].set_xlim(-axis_lim, axis_lim)
axs[4].set_ylim(-axis_lim, axis_lim)
axs[4].set_xlabel('x', fontsize=20)
axs[4].set_ylabel('y', fontsize=20)
axs[4].set_title('Diffusion-QL N=50', fontsize=25)

file_name = f'ql_modify_N_variant2'
file_name += f'_sd{args.seed}.pdf'

fig.tight_layout()
fig.savefig(os.path.join(img_dir, file_name))

