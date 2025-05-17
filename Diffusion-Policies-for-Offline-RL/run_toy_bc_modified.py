import os
import torch
import numpy as np
from torch.distributions import Normal
import argparse
import matplotlib.pyplot as plt

from toy_experiments.toy_helpers import Data_Sampler

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=2022, type=int)
args = parser.parse_args()

seed = args.seed

def generate_data(num_total_samples, device='cpu',
                  inner_ring_modes=16, 
                  outer_ring_modes=24, 
                  ring_std_dev=0.015,  
                  corner_std_dev=0.03, 
                  radius_inner=0.5,    
                  radius_outer=0.75,   
                  clip_val=1.0):
    all_samples_list = []

    inner_ring_pct = 0.35 
    outer_ring_pct = 0.45 

    n_inner_total = round(num_total_samples * inner_ring_pct)
    n_outer_total = round(num_total_samples * outer_ring_pct)
    n_corners_total = max(0, num_total_samples - n_inner_total - n_outer_total)

    current_sum = n_inner_total + n_outer_total + n_corners_total
    diff = num_total_samples - current_sum
    if diff != 0:
        if n_outer_total >= abs(diff):
            n_outer_total += diff
        elif n_inner_total >= abs(diff):
            n_inner_total += diff
        else:
            n_corners_total += diff
            
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
        print("Warning: No samples were generated based on parameters. Returning empty data if num_total_samples is 0, otherwise default.")
        if num_total_samples > 0:
            default_loc = torch.tensor([0.0, 0.0], dtype=torch.float32)
            default_scale = torch.tensor([0.1, 0.1], dtype=torch.float32)
            data = Normal(default_loc, default_scale).sample((num_total_samples,)).clip(-clip_val, clip_val)
        else: 
            data = torch.empty((0, 2), dtype=torch.float32)
    else:
        data = torch.cat(all_samples_list, dim=0)
    if data.shape[0] != num_total_samples and num_total_samples > 0 and not all_samples_list :
         print(f"Error: Sample count mismatch. Generated {data.shape[0]}, requested {num_total_samples}. Check logic.")
         if data.nelement() == 0 :
            default_loc = torch.tensor([0.0, 0.0], dtype=torch.float32)
            default_scale = torch.tensor([0.1, 0.1], dtype=torch.float32)
            data = Normal(default_loc, default_scale).sample((num_total_samples,)).clip(-clip_val, clip_val)


    action = data.to(dtype=torch.float32)
    state = torch.zeros_like(action, dtype=torch.float32) 
    reward = torch.zeros((action.shape[0], 1), dtype=torch.float32) # 确保行数与action一致
    
    return Data_Sampler(state, action, reward, device)

torch.manual_seed(seed)
np.random.seed(seed)

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
hidden_dim = 128
lr = 3e-4

num_epochs = 1000
batch_size = 100
iterations = int(num_data / batch_size)

img_dir = 'toy_imgs/bc'
os.makedirs(img_dir, exist_ok=True)
fig, axs = plt.subplots(1, 5, figsize=(5.5 * 5, 5))
axis_lim = 1.1

# Plot the ground truth
num_eval = 1000
_, action_samples, _ = data_sampler.sample(num_eval)
action_samples = action_samples.cpu().numpy()
axs[0].scatter(action_samples[:, 0], action_samples[:, 1], alpha=0.3)
axs[0].set_xlim(-axis_lim, axis_lim)
axs[0].set_ylim(-axis_lim, axis_lim)
axs[0].set_xlabel('x', fontsize=20)
axs[0].set_ylabel('y', fontsize=20)
axs[0].set_title('Ground Truth', fontsize=25)


# Plot MLE BC
from toy_experiments.bc_mle import BC_MLE as MLE_Agent
mle_agent = MLE_Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=discount,
                      tau=tau,
                      lr=lr,
                      hidden_dim=hidden_dim)


for i in range(num_epochs):
    
    mle_agent.train(data_sampler,
                    iterations=iterations,
                    batch_size=batch_size)
    
    if i % 100 == 0:
        print(f'Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = mle_agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[1].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[1].set_xlim(-2.5, 2.5)
axs[1].set_ylim(-2.5, 2.5)
axs[1].set_xlabel('x', fontsize=20)
axs[1].set_ylabel('y', fontsize=20)
axs[1].set_title('BC-MLE', fontsize=25)


# Plot CVAE BC
from toy_experiments.bc_cvae import BC_CVAE as CVAE_Agent
cvae_agent = CVAE_Agent(state_dim=state_dim,
                        action_dim=action_dim,
                        max_action=max_action,
                        device=device,
                        discount=discount,
                        tau=tau,
                        lr=lr,
                        hidden_dim=hidden_dim)


for i in range(num_epochs):
    
    cvae_agent.train(data_sampler,
                     iterations=iterations,
                     batch_size=batch_size)
    
    if i % 100 == 0:
        print(f'Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = cvae_agent.vae.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[2].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[2].set_xlim(-axis_lim, axis_lim)
axs[2].set_ylim(-axis_lim, axis_lim)
axs[2].set_xlabel('x', fontsize=20)
axs[2].set_ylabel('y', fontsize=20)
axs[2].set_title('BC-CVAE', fontsize=25)


# Plot CVAE BC
from toy_experiments.bc_mmd import BC_MMD as MMD_Agent

mmd_agent =  MMD_Agent(state_dim=state_dim,
                       action_dim=action_dim,
                       max_action=max_action,
                       device=device,
                       discount=discount,
                       tau=tau,
                       lr=lr,
                       hidden_dim=hidden_dim)

for i in range(num_epochs):

    mmd_agent.train(data_sampler,
                    iterations=iterations,
                    batch_size=batch_size)

    if i % 100 == 0:
        print(f'Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = mmd_agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[3].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[3].set_xlim(-axis_lim, axis_lim)
axs[3].set_ylim(-axis_lim, axis_lim)
axs[3].set_xlabel('x', fontsize=20)
axs[3].set_ylabel('y', fontsize=20)
axs[3].set_title('BC-MMD', fontsize=25)


# Plot Diffusion BC
from toy_experiments.bc_diffusion import BC as Diffusion_Agent
diffusion_agent = Diffusion_Agent(state_dim=state_dim,
                                  action_dim=action_dim,
                                  max_action=max_action,
                                  device=device,
                                  discount=discount,
                                  tau=tau,
                                  beta_schedule=beta_schedule,
                                  n_timesteps=T,
                                  model_type=model_type,
                                  hidden_dim=hidden_dim,
                                  lr=lr)

for i in range(num_epochs):
    
    diffusion_agent.train(data_sampler,
                          iterations=iterations,
                          batch_size=batch_size)
    
    if i % 100 == 0:
        print(f'Epoch: {i}')

new_state = torch.zeros((num_eval, 2), device=device)
new_action = diffusion_agent.actor.sample(new_state)
new_action = new_action.detach().cpu().numpy()
axs[4].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[4].set_xlim(-axis_lim, axis_lim)
axs[4].set_ylim(-axis_lim, axis_lim)
axs[4].set_xlabel('x', fontsize=20)
axs[4].set_ylabel('y', fontsize=20)
axs[4].set_title('BC-Diffusion', fontsize=25)

fig.tight_layout()
fig.savefig(os.path.join(img_dir, f'bc_diffusion_circle_{T}_sd{seed}.pdf'))



