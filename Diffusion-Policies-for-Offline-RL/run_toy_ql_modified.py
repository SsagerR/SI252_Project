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
        np.random.shuffle(samples_counts_inner)
        for i in range(inner_ring_modes):
            if samples_counts_inner[i] == 0: continue
            angle = 2 * np.pi * i / inner_ring_modes
            center_x = radius_inner * np.cos(angle)
            center_y = radius_inner * np.sin(angle)
            loc = torch.tensor([center_x, center_y], dtype=torch.float32)
            scale = torch.tensor([ring_std_dev, ring_std_dev], dtype=torch.float32)
            mode = Normal(loc, scale)
            samples = mode.sample((samples_counts_inner[i],)).clip(-clip_val, clip_val)
            all_samples_list.append(samples)

    if outer_ring_modes > 0 and n_outer_total > 0:
        base_outer, rem_outer = divmod(n_outer_total, outer_ring_modes)
        samples_counts_outer = [base_outer + 1] * rem_outer + [base_outer] * (outer_ring_modes - rem_outer)
        np.random.shuffle(samples_counts_outer)
        for i in range(outer_ring_modes):
            if samples_counts_outer[i] == 0: continue
            angle = 2 * np.pi * i / outer_ring_modes + (np.pi / outer_ring_modes)
            center_x = radius_outer * np.cos(angle)
            center_y = radius_outer * np.sin(angle)
            loc = torch.tensor([center_x, center_y], dtype=torch.float32)
            scale = torch.tensor([ring_std_dev, ring_std_dev], dtype=torch.float32)
            mode = Normal(loc, scale)
            samples = mode.sample((samples_counts_outer[i],)).clip(-clip_val, clip_val)
            all_samples_list.append(samples)
        
    num_modes_corners = 4 
    pos_corner = 0.9 
    if num_modes_corners > 0 and n_corners_total > 0:
        base_corners, rem_corners = divmod(n_corners_total, num_modes_corners)
        samples_counts_corners = [base_corners + 1] * rem_corners + [base_corners] * (num_modes_corners - rem_corners)
        np.random.shuffle(samples_counts_corners)
        corner_centers_coords = [
            [-pos_corner, pos_corner], [-pos_corner, -pos_corner],
            [pos_corner, pos_corner], [pos_corner, -pos_corner]
        ]
        for i in range(num_modes_corners):
            if samples_counts_corners[i] == 0: continue
            loc = torch.tensor(corner_centers_coords[i], dtype=torch.float32)
            scale = torch.tensor([corner_std_dev, corner_std_dev], dtype=torch.float32)
            mode = Normal(loc, scale)
            samples = mode.sample((samples_counts_corners[i],)).clip(-clip_val, clip_val)
            all_samples_list.append(samples)

    if not all_samples_list:
        if num_total_samples > 0:
            default_loc = torch.tensor([0.0, 0.0], dtype=torch.float32)
            default_scale = torch.tensor([0.1, 0.1], dtype=torch.float32)
            data = Normal(default_loc, default_scale).sample((num_total_samples,)).clip(-clip_val, clip_val)
        else: 
            data = torch.empty((0, 2), dtype=torch.float32)
    else:
        data = torch.cat(all_samples_list, dim=0)

    action = data.to(dtype=torch.float32)
    state = torch.zeros_like(action, dtype=torch.float32) 

    reward_peaks = [
        (0.8, -0.8, 6.0, 0.15, 0.15),
        (0.0, radius_outer, 3.5, 0.2, 0.2),
        (-radius_inner, 0.0, 2.0, 0.15, 0.15),
        (-0.7, 0.7, 1.0, 0.2, 0.2)
    ]
    
    current_rewards = torch.zeros((action.shape[0], 1), dtype=torch.float32, device=device)
    action_on_device = action.to(device)

    for mux, muy, amp, sigx, sigy in reward_peaks:
        term = amp * torch.exp(
            -((action_on_device[:, 0] - mux)**2 / (2 * sigx**2))
            -((action_on_device[:, 1] - muy)**2 / (2 * sigy**2))
        )
        current_rewards += term.unsqueeze(1)

    noise = reward_noise_std * torch.randn_like(current_rewards, device=device)
    final_reward = current_rewards + noise
    reward = final_reward 
    
    return Data_Sampler(state, action, reward, device)


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

num_epochs = 1000
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

# Plot QL-MLE
from toy_experiments.ql_mle import QL_MLE

agent = QL_MLE(state_dim=state_dim,
               action_dim=action_dim,
               max_action=max_action,
               device=device,
               discount=discount,
               tau=tau,
               eta=eta,
               hidden_dim=hidden_dim,
               lr=lr,
               r_fun=None)

for i in range(1, num_epochs + 1):

    agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f'QL-MLE Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[1].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3, color='#d62728')
axs[1].set_xlim(-axis_lim, axis_lim)
axs[1].set_ylim(-axis_lim, axis_lim)
axs[1].set_xlabel('x', fontsize=20)
axs[1].set_ylabel('y', fontsize=20)
axs[1].set_title('TD3+BC', fontsize=25)

# Plot QL-CVAE
from toy_experiments.ql_cvae import QL_CVAE

agent = QL_CVAE(state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                device=device,
                discount=discount,
                tau=tau,
                hidden_dim=hidden_dim,
                lr=lr,
                r_fun=None)

for i in range(1, num_epochs + 1):

    agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f'QL-CVAE Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.vae.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[2].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3, color='#d62728')
axs[2].set_xlim(-axis_lim, axis_lim)
axs[2].set_ylim(-axis_lim, axis_lim)
axs[2].set_xlabel('x', fontsize=20)
axs[2].set_ylabel('y', fontsize=20)
axs[2].set_title('BCQ', fontsize=25)

# Plot QL-MMD
from toy_experiments.ql_mmd import QL_MMD

agent = QL_MMD(state_dim=state_dim,
               action_dim=action_dim,
               max_action=max_action,
               device=device,
               discount=discount,
               tau=tau,
               hidden_dim=hidden_dim,
               lr=lr,
               r_fun=None)

for i in range(1, num_epochs + 1):

    agent.train(data_sampler, iterations=iterations, batch_size=batch_size)

    if i % 100 == 0:
        print(f'QL-MMD Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[3].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3, color='#d62728')
axs[3].set_xlim(-axis_lim, axis_lim)
axs[3].set_ylim(-axis_lim, axis_lim)
axs[3].set_xlabel('x', fontsize=20)
axs[3].set_ylabel('y', fontsize=20)
axs[3].set_title('BEAR-MMD', fontsize=25)


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
                     n_timesteps=T,
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
axs[4].set_title('Diffusion-QL', fontsize=25)

file_name = f'ql_all_T{T}_eta{eta}_r_fun{args.r_fun}_lr{lr}_hd{hidden_dim}_mode_{args.mode}'
file_name += f'_sd{args.seed}_circle_distribution.pdf'

fig.tight_layout()
fig.savefig(os.path.join(img_dir, file_name))

