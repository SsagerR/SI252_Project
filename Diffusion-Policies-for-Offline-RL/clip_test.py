import os
import torch
import numpy as np
# from torch.distributions import Normal # Not used in this script snippet
import argparse
import matplotlib.pyplot as plt

# Assuming toy_helpers.py and bc_diffusion.py are in a directory named 'toy_experiments'
# and this script is in the parent directory of 'toy_experiments'
# or 'toy_experiments' is in the Python path.
from toy_experiments.toy_helpers import Data_Sampler
from toy_experiments.bc_diffusion import BC as Diffusion_Agent

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
elif torch.backends.mps.is_available() and torch.backends.mps.is_built(): # For Apple Silicon GPU
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

num_epochs = 3 # Reduced for quicker testing if needed, original was 1000
batch_size = 100
iterations = num_data // batch_size

img_dir = 'toy_imgs/bc'
os.makedirs(img_dir, exist_ok=True)
fig, axs = plt.subplots(1, 5, figsize=(6 * 5, 5.5)) # Adjusted figsize slightly for text and legends
axis_lim = 1.1

# Define colors and labels for plotting
color_in_bounds = 'blue'
color_out_of_bounds = 'red'
alpha_scatter = 0.3
label_in_bounds = 'In [-1,1]x[-1,1]'
label_out_of_bounds = 'Outside [-1,1]x[-1,1]' # For visible out-of-bounds points

def plot_points_with_bounds_check(ax, points_data, title):
    """Helper function to plot points with bounds checking, legend, and count."""
    x_coords = points_data[:, 0]
    y_coords = points_data[:, 1]

    # Create a mask for points strictly outside [-1, 1] x [-1, 1]
    out_of_bounds_strict_mask = (x_coords < -1.0) | (x_coords > 1.0) | \
                                (y_coords < -1.0) | (y_coords > 1.0)
    in_bounds_strict_mask = ~out_of_bounds_strict_mask

    # Count the number of points strictly outside [-1, 1] x [-1, 1]
    num_out_of_bounds_strict = np.sum(out_of_bounds_strict_mask)

    # Plot in-bounds points (those strictly within [-1,1]x[-1,1])
    ax.scatter(x_coords[in_bounds_strict_mask], y_coords[in_bounds_strict_mask],
               alpha=alpha_scatter, color=color_in_bounds, label=label_in_bounds)

    # Plot out-of-bounds points (those strictly outside [-1,1]x[-1,1] but potentially still visible within axis_lim)
    # These points will be colored red if they are visible within the plot's xlim/ylim.
    ax.scatter(x_coords[out_of_bounds_strict_mask], y_coords[out_of_bounds_strict_mask],
               alpha=alpha_scatter, color=color_out_of_bounds, label=label_out_of_bounds)

    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_title(title, fontsize=25)
    ax.legend(fontsize='small', loc='lower right') # Adjust legend location if needed

    # Add text annotation for the count of points strictly outside [-1, 1] x [-1, 1]
    # You can change the text (e.g., to Chinese) or position as needed.
    annotation_text = f'Count outside [-1,1]x[-1,1]: {num_out_of_bounds_strict}'
    ax.text(0.03, 0.97, annotation_text, transform=ax.transAxes, # Position: top-left
            fontsize='medium', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))


# Plot the ground truth
num_eval = 1000
_, action_samples, _ = data_sampler.sample(num_eval)
action_samples_np = action_samples.cpu().numpy()
plot_points_with_bounds_check(axs[0], action_samples_np, 'Ground Truth')


# --- BC-Diffusion Agents ---
n_timesteps_list = [2, 5, 10, 50]

for idx, n_steps in enumerate(n_timesteps_list):
    print(f"\nTraining and evaluating BC-Diffusion with N={n_steps}")
    diffusion_agent = Diffusion_Agent(state_dim=state_dim,
                                      action_dim=action_dim,
                                      max_action=max_action,
                                      device=device,
                                      discount=discount,
                                      tau=tau,
                                      beta_schedule=beta_schedule,
                                      n_timesteps=n_steps,
                                      model_type=model_type,
                                      hidden_dim=hidden_dim,
                                      lr=lr)

    for i in range(num_epochs):
        diffusion_agent.train(data_sampler,
                              iterations=iterations,
                              batch_size=batch_size)
        if (i + 1) % 100 == 0:
            print(f'N={n_steps}, Epoch: {i + 1}/{num_epochs}')

    new_state_eval = torch.zeros((num_eval, state_dim), device=device)
    sampled_actions = diffusion_agent.actor.sample(new_state_eval)
    sampled_actions_np = sampled_actions.detach().cpu().numpy()

    plot_points_with_bounds_check(axs[idx + 1], sampled_actions_np, f'BC-Diffusion N={n_steps}')


fig.tight_layout(pad=0.5) # Adjust layout
fig.savefig(os.path.join(img_dir, f'bc_diffusion_N_highlighted_counted_{seed}.pdf'))
plt.show()

print("Script finished.")