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

def generate_data(num_total_samples, device='cpu', theta_R_coefficient=None):
    if theta_R_coefficient is None:
        theta_R_coefficient = (6.0 * np.pi)

    if num_total_samples <= 0:
        action = torch.empty((0, 2), dtype=torch.float32, device=device)
        state = torch.empty((0, 2), dtype=torch.float32, device=device) 
        reward = torch.empty((0, 1), dtype=torch.float32, device=device)
        return Data_Sampler(state=state, action=action, reward=reward, device=device)
    
    R = torch.rand(num_total_samples, device=device, dtype=torch.float32)

    theta = theta_R_coefficient * R

    x = R * torch.cos(theta)
    y = R * torch.sin(theta)

    action = torch.stack((x, y), dim=1)
    state = torch.zeros_like(action, dtype=torch.float32, device=device)
    reward = torch.zeros((num_total_samples, 1), dtype=torch.float32, device=device)
    
    return Data_Sampler(state=state, action=action, reward=reward, device=device)

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

# Plot Diffusion BC
from toy_experiments.bc_diffusion import BC as Diffusion_Agent
diffusion_agent = Diffusion_Agent(state_dim=state_dim,
                                  action_dim=action_dim,
                                  max_action=max_action,
                                  device=device,
                                  discount=discount,
                                  tau=tau,
                                  beta_schedule=beta_schedule,
                                  n_timesteps=2,
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
axs[1].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[1].set_xlim(-axis_lim, axis_lim)
axs[1].set_ylim(-axis_lim, axis_lim)
axs[1].set_xlabel('x', fontsize=20)
axs[1].set_ylabel('y', fontsize=20)
axs[1].set_title('BC-Diffusion N=2', fontsize=25)

# Plot Diffusion BC
from toy_experiments.bc_diffusion import BC as Diffusion_Agent
diffusion_agent = Diffusion_Agent(state_dim=state_dim,
                                  action_dim=action_dim,
                                  max_action=max_action,
                                  device=device,
                                  discount=discount,
                                  tau=tau,
                                  beta_schedule=beta_schedule,
                                  n_timesteps=5,
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
axs[2].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[2].set_xlim(-axis_lim, axis_lim)
axs[2].set_ylim(-axis_lim, axis_lim)
axs[2].set_xlabel('x', fontsize=20)
axs[2].set_ylabel('y', fontsize=20)
axs[2].set_title('BC-Diffusion N=5', fontsize=25)

# Plot Diffusion BC
from toy_experiments.bc_diffusion import BC as Diffusion_Agent
diffusion_agent = Diffusion_Agent(state_dim=state_dim,
                                  action_dim=action_dim,
                                  max_action=max_action,
                                  device=device,
                                  discount=discount,
                                  tau=tau,
                                  beta_schedule=beta_schedule,
                                  n_timesteps=10,
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
axs[3].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[3].set_xlim(-axis_lim, axis_lim)
axs[3].set_ylim(-axis_lim, axis_lim)
axs[3].set_xlabel('x', fontsize=20)
axs[3].set_ylabel('y', fontsize=20)
axs[3].set_title('BC-Diffusion N=10', fontsize=25)

# Plot Diffusion BC
from toy_experiments.bc_diffusion import BC as Diffusion_Agent
diffusion_agent = Diffusion_Agent(state_dim=state_dim,
                                  action_dim=action_dim,
                                  max_action=max_action,
                                  device=device,
                                  discount=discount,
                                  tau=tau,
                                  beta_schedule=beta_schedule,
                                  n_timesteps=50,
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
axs[4].set_title('BC-Diffusion N=50', fontsize=25)

fig.tight_layout()
fig.savefig(os.path.join(img_dir, f'bc_diffusion_modified2_N_{seed}.pdf'))



