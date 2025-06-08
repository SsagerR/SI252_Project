import os
import torch
import numpy as np
from torch.distributions import Normal
import argparse
import matplotlib.pyplot as plt

# Assuming toy_helpers.Data_Sampler is defined as in the previous context
# If not, you'll need to provide its definition.
# For now, I'll assume it's a simple class/namedtuple like:
# from collections import namedtuple
# Data_Sampler = namedtuple('Data_Sampler', ['state', 'action', 'reward', 'device'])
# OR a class that can be instantiated as Data_Sampler(state, action, reward, device)
# and has a .sample() method.

# Placeholder for Data_Sampler if not available (replace with actual import)
try:
    from toy_experiments.toy_helpers import Data_Sampler
except ImportError:
    print("Warning: 'toy_experiments.toy_helpers.Data_Sampler' not found. Using a placeholder.")
    from collections import namedtuple
    class Placeholder_Data_Sampler:
        def __init__(self, state, action, reward, device):
            self.state = state
            self.action = action
            self.reward = reward
            self.device = device
            self._data_size = state.shape[0]
            self._idx = np.arange(self._data_size)

        def sample(self, batch_size):
            sampled_indices = np.random.choice(self._idx, size=batch_size, replace=False)
            return (
                self.state[sampled_indices],
                self.action[sampled_indices],
                self.reward[sampled_indices]
            )
    Data_Sampler = Placeholder_Data_Sampler


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=2022, type=int)
# Add other arguments from the QL script if they are needed by agents here
# For example, hidden_dim, lr, etc. if they are not hardcoded later.
# parser.add_argument("--lr", default=3e-4, type=float)
# parser.add_argument('--hidden_dim', default=128, type=int)
args = parser.parse_args()

seed = args.seed
# lr = args.lr # Example if added
# hidden_dim = args.hidden_dim # Example if added

# Device selection (more robust, similar to QL script)
if torch.cuda.is_available():
    # device_str = "cuda:0" # Or use a specific device if multiple GPUs
    # For simplicity, using the first available CUDA device.
    # If you have args.device from parser, you can use f"cuda:{args.device}"
    device_str = "cuda:0"
    torch.cuda.set_device(int(device_str.split(':')[1])) # Ensure correct device context
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device_str = "mps"
    print("Using MPS (Apple Silicon GPU)")
else:
    device_str = "cpu"
    print("Using CPU")
device = torch.device(device_str)


def get_char_local_points(char_symbol):
    points = {}
    points['S'] = [
        (0.4, 0.9), (0.1, 1.0), (-0.2, 0.9), (-0.4, 0.7),
        (-0.3, 0.4), (-0.1, 0.1), (0.1, -0.2), (0.3, -0.5),
        (0.4, -0.7), (0.1, -0.9), (-0.2, -1.0), (-0.4, -0.9)
    ]
    # Define which local points of 'I' are considered "middle"
    # These are the indices from the list below.
    # Stem: (0.0, 1.0)_0, (0.0, 0.6)_1, (0.0, 0.2)_2, (0.0, -0.2)_3, (0.0, -0.6)_4, (0.0, -1.0)_5
    # Middle points for 'I' could be (0.0, 0.2) and (0.0, -0.2)
    points['I_middle_indices'] = [2, 3] # Indices of the middle stem points
    points['I'] = [
        (0.0, 1.0), (0.0, 0.6), (0.0, 0.2), (0.0, -0.2), (0.0, -0.6), (0.0, -1.0), # Stem
        (-0.3, 1.0), (0.3, 1.0),  # Top bar
        (-0.3, -1.0), (0.3, -1.0) # Bottom bar
    ]
    points['2'] = [
        (-0.4, 0.9), (-0.1, 1.0), (0.2, 0.9), (0.4, 0.6),
        (0.3, 0.2), (0.0, -0.1), (-0.2, -0.4), (-0.4, -0.7),
        (-0.4, -1.0), (-0.1, -1.0), (0.2, -1.0), (0.4, -1.0)
    ]
    points['5'] = [
        (0.4, 1.0), (0.0, 1.0), (-0.4, 1.0),
        (-0.4, 0.8), (-0.4, 0.3),
        (-0.3, 0.0), (0.0, -0.2), (0.3, -0.4), (0.4, -0.7),
        (0.3, -1.0), (0.0, -1.0), (-0.3, -1.0)
    ]
    if char_symbol == 'I_middle_indices': # Special key to get indices
        return points['I_middle_indices']
    return points.get(char_symbol, [])


def generate_data(num_total_samples, device='cpu',
                  clip_val=1.0,
                  # reward_noise_std is not used as std is part of reward N(mean, std)
                  # radius_inner, radius_outer are also not used for this reward logic
                  **kwargs): # Absorb unused kwargs like radius_inner, etc.
    all_samples_list = []
    all_rewards_list = [] # New list to store rewards per mode

    character_sequence = ['S', 'I', '2', '5', '2']
    num_characters = len(character_sequence)

    total_plot_width_for_chars = 1.8
    character_cell_width = total_plot_width_for_chars / num_characters
    start_x_position = -total_plot_width_for_chars / 2.0

    char_x_scale = character_cell_width * 0.7
    char_y_scale = 0.35
    char_mode_std_dev = 0.02

    samples_per_char_base, samples_rem = divmod(num_total_samples, num_characters)
    char_sample_counts = [samples_per_char_base + 1] * samples_rem + \
                         [samples_per_char_base] * (num_characters - samples_rem)

    middle_I_mode_indices = get_char_local_points('I_middle_indices')
    low_reward_means = [1.0, 2.0, 3.0]
    reward_std = 0.5

    for char_idx, char_sym in enumerate(character_sequence):
        samples_for_this_char = char_sample_counts[char_idx]
        if samples_for_this_char <= 0:
            continue

        char_local_modes = get_char_local_points(char_sym)
        num_modes_for_char = len(char_local_modes)
        if num_modes_for_char == 0:
            continue

        base_s_p_m, rem_s_p_m = divmod(samples_for_this_char, num_modes_for_char)
        mode_sample_counts_for_char = [base_s_p_m + 1] * rem_s_p_m + \
                                      [base_s_p_m] * (num_modes_for_char - rem_s_p_m)

        char_cell_center_x = start_x_position + (char_idx * character_cell_width) + (character_cell_width / 2)

        for mode_idx, (local_x, local_y) in enumerate(char_local_modes):
            num_samples_this_mode = mode_sample_counts_for_char[mode_idx]
            if num_samples_this_mode == 0:
                continue

            center_x = char_cell_center_x + (local_x * char_x_scale)
            center_y = local_y * char_y_scale
            loc = torch.tensor([center_x, center_y], dtype=torch.float32)
            scale_tensor = torch.tensor([char_mode_std_dev, char_mode_std_dev], dtype=torch.float32) # Renamed to avoid conflict
            mode_dist = Normal(loc, scale_tensor)
            samples = mode_dist.sample((num_samples_this_mode,)).clip(-clip_val, clip_val)
            all_samples_list.append(samples)

            # --- New Reward Generation Logic ---
            current_reward_mean = 0.0
            if char_sym == 'I' and mode_idx in middle_I_mode_indices:
                current_reward_mean = 5.0
            else:
                current_reward_mean = np.random.choice(low_reward_means)
            
            # Generate rewards for samples from this mode
            # Ensure reward tensors are on the correct device
            mean_tensor = torch.full((num_samples_this_mode, 1), current_reward_mean, dtype=torch.float32, device=device)
            std_tensor = torch.full((num_samples_this_mode, 1), reward_std, dtype=torch.float32, device=device)
            
            current_mode_rewards =torch.ones((num_samples_this_mode, 1))
            all_rewards_list.append(current_mode_rewards)

    if not all_samples_list: # Fallback if no samples generated
        if num_total_samples > 0:
            print("Warning: No samples generated. Generating default central data and rewards.")
            default_loc = torch.tensor([0.0, 0.0], dtype=torch.float32)
            default_scale = torch.tensor([0.1, 0.1], dtype=torch.float32)
            data = Normal(default_loc, default_scale).sample((num_total_samples,)).clip(-clip_val, clip_val)
            # Default rewards if no specific points generated
            reward_mean = np.random.choice(low_reward_means)
            mean_tensor = torch.full((num_total_samples, 1), reward_mean, dtype=torch.float32, device=device)
            std_tensor = torch.full((num_total_samples, 1), reward_std, dtype=torch.float32, device=device)
            reward = Normal(mean_tensor, std_tensor).sample()
        else:
            data = torch.empty((0, 2), dtype=torch.float32)
            reward = torch.empty((0, 1), dtype=torch.float32) # Ensure reward is also empty
    else:
        data = torch.cat(all_samples_list, dim=0)
        reward = torch.cat(all_rewards_list, dim=0)


    # Adjust if generated sample count doesn't exactly match
    if data.shape[0] != num_total_samples and num_total_samples > 0:
        print(f"Notice: Generated data points ({data.shape[0]}) vs requested ({num_total_samples}). Adjusting...")
        if data.shape[0] > num_total_samples:
            perm = torch.randperm(data.shape[0])[:num_total_samples]
            data = data[perm]
            reward = reward[perm]
        elif data.shape[0] < num_total_samples and data.shape[0] > 0:
            num_to_add = num_total_samples - data.shape[0]
            if data.shape[0] > 0:
                indices_to_add = torch.randint(0, data.shape[0], (num_to_add,))
                extra_samples = data[indices_to_add]
                extra_rewards = reward[indices_to_add]
                data = torch.cat([data, extra_samples], dim=0)
                reward = torch.cat([reward, extra_rewards], dim=0)
    
    action = data.to(dtype=torch.float32) # Already on device from generation or cat
    state = torch.zeros_like(action, dtype=torch.float32) # Create on same device as action

    # Ensure all outputs are on the specified device
    return Data_Sampler(state.to(device), action.to(device), reward.to(device), device)

torch.manual_seed(seed)
np.random.seed(seed)

num_data = int(10000)
data_sampler = generate_data(num_data, device=device, clip_val=1.0) # Using clip_val consistent with original script

state_dim = 2
action_dim = 2
max_action = 1.0 # Corresponds to clip_val in generate_data

discount = 0.99
tau = 0.005 # Often used in actor-critic updates, may or may not be used by all BC agents
model_type = 'MLP' # For Diffusion BC

T = 50 # For Diffusion BC
beta_schedule = 'vp' # For Diffusion BC
hidden_dim = 128 # Default hidden dimension for networks
lr = 3e-4 # Default learning rate

num_epochs = 1000
batch_size = 100
if batch_size > 0 and num_data > 0:
    iterations = int(num_data / batch_size)
else:
    iterations = 0
    print("Warning: batch_size or num_data is 0, setting iterations to 0.")


#img_dir = 'toy_imgs/bc_5cluster_reward1' # Changed directory name for clarity
img_dir = 'toy_imgs'
os.makedirs(img_dir, exist_ok=True)
fig, axs = plt.subplots(1, 5, figsize=(5.5 * 5, 5))
axis_lim = 1.1

# Plot the ground truth
num_eval_plot = num_data # Use a portion of data for plotting if num_data is very large
# Sample directly from the generated actions for ground truth plot
# The Data_Sampler's sample method might shuffle or pick a subset.
# For a true ground truth of the generated distribution, we use data_sampler.action
if hasattr(data_sampler, 'action') and data_sampler.action.nelement() > 0:
    action_samples_gt = data_sampler.action.cpu().numpy()
    # If rewards are constant 1, all points will have the same color.
    # We can add reward to the plot if needed, but it won't be very informative here.
    # reward_samples_gt = data_sampler.reward.cpu().numpy().flatten()
    axs[0].scatter(action_samples_gt[:num_eval_plot, 0], action_samples_gt[:num_eval_plot, 1], alpha=0.3, s=10) # c=reward_samples_gt if desired
else:
    axs[0].text(0.5, 0.5, "No ground truth data to plot.", ha='center', va='center', color='red')

axs[0].set_xlim(-axis_lim, axis_lim)
axs[0].set_ylim(-axis_lim, axis_lim)
axs[0].set_xlabel('x', fontsize=20)
axs[0].set_ylabel('y', fontsize=20)
axs[0].set_title('Ground Truth (SI252, Reward=1)', fontsize=20) # Updated title
axs[0].set_aspect('equal', adjustable='box')
axs[0].grid(True, linestyle='--', alpha=0.7)


# Plot MLE BC
try:
    from toy_experiments.bc_mle import BC_MLE as MLE_Agent
    mle_agent = MLE_Agent(state_dim=state_dim,
                          action_dim=action_dim,
                          max_action=max_action,
                          device=device,
                          discount=discount, # BC_MLE might not use discount/tau
                          tau=tau,
                          lr=lr,
                          hidden_dim=hidden_dim)

    if iterations > 0:
        for i in range(1, num_epochs + 1):
            mle_agent.train(data_sampler,
                            iterations=iterations,
                            batch_size=batch_size)
            if i % 100 == 0:
                print(f'BC-MLE Epoch: {i}')
    else:
        print("Skipping BC-MLE training as iterations is 0.")

    new_state_mle = torch.zeros((num_eval_plot, state_dim), device=device)
    new_action_mle = mle_agent.actor.sample(new_state_mle) # BC_MLE might have .sample() or .act()
    new_action_mle = new_action_mle.detach().cpu().numpy()
    axs[1].scatter(new_action_mle[:, 0], new_action_mle[:, 1], alpha=0.3, color='#d62728')
except ImportError:
    axs[1].text(0.5, 0.5, "BC_MLE not found.", ha='center', va='center', color='orange')
    print("Skipping BC-MLE: toy_experiments.bc_mle.BC_MLE not found.")

axs[1].set_xlim(-axis_lim, axis_lim) # Changed from -2.5, 2.5 to be consistent
axs[1].set_ylim(-axis_lim, axis_lim) # Changed from -2.5, 2.5 to be consistent
axs[1].set_xlabel('x', fontsize=20)
axs[1].set_ylabel('y', fontsize=20)
axs[1].set_title('BC-MLE', fontsize=25)
axs[1].set_aspect('equal', adjustable='box')
axs[1].grid(True, linestyle='--', alpha=0.7)


# Plot CVAE BC
try:
    from toy_experiments.bc_cvae import BC_CVAE as CVAE_Agent
    cvae_agent = CVAE_Agent(state_dim=state_dim,
                            action_dim=action_dim,
                            max_action=max_action, # BC_CVAE might also use latent_dim
                            device=device,
                            discount=discount, # BC_CVAE might not use discount/tau
                            tau=tau,
                            lr=lr,
                            hidden_dim=hidden_dim) # Ensure BC_CVAE takes hidden_dim

    if iterations > 0:
        for i in range(1, num_epochs + 1):
            cvae_agent.train(data_sampler,
                             iterations=iterations,
                             batch_size=batch_size)
            if i % 100 == 0:
                print(f'BC-CVAE Epoch: {i}')
    else:
        print("Skipping BC-CVAE training as iterations is 0.")

    new_state_cvae = torch.zeros((num_eval_plot, state_dim), device=device)
    new_action_cvae = cvae_agent.vae.sample(new_state_cvae) # BC_CVAE uses vae.sample
    new_action_cvae = new_action_cvae.detach().cpu().numpy()
    axs[2].scatter(new_action_cvae[:, 0], new_action_cvae[:, 1], alpha=0.3, color='#d62728')
except ImportError:
    axs[2].text(0.5, 0.5, "BC_CVAE not found.", ha='center', va='center', color='orange')
    print("Skipping BC-CVAE: toy_experiments.bc_cvae.BC_CVAE not found.")

axs[2].set_xlim(-axis_lim, axis_lim)
axs[2].set_ylim(-axis_lim, axis_lim)
axs[2].set_xlabel('x', fontsize=20)
axs[2].set_ylabel('y', fontsize=20)
axs[2].set_title('BC-CVAE', fontsize=25)
axs[2].set_aspect('equal', adjustable='box')
axs[2].grid(True, linestyle='--', alpha=0.7)


# Plot MMD BC
try:
    from toy_experiments.bc_mmd import BC_MMD as MMD_Agent
    mmd_agent =  MMD_Agent(state_dim=state_dim,
                           action_dim=action_dim,
                           max_action=max_action,
                           device=device,
                           discount=discount, # BC_MMD might not use discount/tau
                           tau=tau,
                           lr=lr,
                           hidden_dim=hidden_dim)
    if iterations > 0:
        for i in range(1, num_epochs + 1):
            mmd_agent.train(data_sampler,
                            iterations=iterations,
                            batch_size=batch_size)
            if i % 100 == 0:
                print(f'BC-MMD Epoch: {i}')
    else:
        print("Skipping BC-MMD training as iterations is 0.")

    new_state_mmd = torch.zeros((num_eval_plot, state_dim), device=device)
    new_action_mmd = mmd_agent.actor.sample(new_state_mmd) # BC_MMD might have .sample() or .act()
    new_action_mmd = new_action_mmd.detach().cpu().numpy()
    axs[3].scatter(new_action_mmd[:, 0], new_action_mmd[:, 1], alpha=0.3, color='#d62728')
except ImportError:
    axs[3].text(0.5, 0.5, "BC_MMD not found.", ha='center', va='center', color='orange')
    print("Skipping BC-MMD: toy_experiments.bc_mmd.BC_MMD not found.")

axs[3].set_xlim(-axis_lim, axis_lim)
axs[3].set_ylim(-axis_lim, axis_lim)
axs[3].set_xlabel('x', fontsize=20)
axs[3].set_ylabel('y', fontsize=20)
axs[3].set_title('BC-MMD', fontsize=25)
axs[3].set_aspect('equal', adjustable='box')
axs[3].grid(True, linestyle='--', alpha=0.7)


# Plot Diffusion BC
try:
    from toy_experiments.bc_diffusion import BC as Diffusion_Agent # Original script had 'BC'
    diffusion_agent = Diffusion_Agent(state_dim=state_dim,
                                      action_dim=action_dim,
                                      max_action=max_action,
                                      device=device,
                                      discount=discount, # BC_Diffusion might not use discount/tau
                                      tau=tau,
                                      beta_schedule=beta_schedule,
                                      n_timesteps=T,
                                      model_type=model_type, # Ensure BC_Diffusion uses this
                                      hidden_dim=hidden_dim,
                                      lr=lr)
    if iterations > 0:
        for i in range(1, num_epochs + 1):
            # Diffusion BC train might return loss, adjust if needed
            diffusion_agent.train(data_sampler,
                                  iterations=iterations,
                                  batch_size=batch_size)
            if i % 100 == 0:
                print(f'BC-Diffusion Epoch: {i}')
    else:
        print("Skipping BC-Diffusion training as iterations is 0.")

    new_state_diffusion = torch.zeros((num_eval_plot, state_dim), device=device)
    new_action_diffusion = diffusion_agent.actor.sample(new_state_diffusion) # Diffusion BC uses actor.sample
    new_action_diffusion = new_action_diffusion.detach().cpu().numpy()
    axs[4].scatter(new_action_diffusion[:, 0], new_action_diffusion[:, 1], alpha=0.3, color='#d62728')
except ImportError:
    axs[4].text(0.5, 0.5, "BC_Diffusion not found.", ha='center', va='center', color='orange')
    print("Skipping BC-Diffusion: toy_experiments.bc_diffusion.BC not found.")

axs[4].set_xlim(-axis_lim, axis_lim)
axs[4].set_ylim(-axis_lim, axis_lim)
axs[4].set_xlabel('x', fontsize=20)
axs[4].set_ylabel('y', fontsize=20)
axs[4].set_title('BC-Diffusion', fontsize=25)
axs[4].set_aspect('equal', adjustable='box')
axs[4].grid(True, linestyle='--', alpha=0.7)


fig.tight_layout()
file_name = f'bc_5cluster_reward1_T{T}_sd{seed}_modified3.pdf' # Updated filename
fig.savefig(os.path.join(img_dir, file_name))
print(f"Saved plot to {os.path.join(img_dir, file_name)}")