import os
import torch
import numpy as np
from torch.distributions import Normal
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# Assuming toy_experiments.toy_helpers.Data_Sampler is defined elsewhere
# For this example, let's create a placeholder if it's not available
try:
    from toy_experiments.toy_helpers import Data_Sampler
except ImportError:
    print("Warning: toy_experiments.toy_helpers.Data_Sampler not found. Using a placeholder.")
    class Data_Sampler:
        def __init__(self, state, action, reward, device):
            self.state = state
            self.action = action
            self.reward = reward
            self.device = device
        def sample(self, batch_size):
            indices = np.random.randint(0, self.state.shape[0], batch_size)
            return (
                self.state[indices],
                self.action[indices],
                self.reward[indices],
                self.state[indices], # Assuming next_state is same as state
                torch.zeros(batch_size, 1, device=self.device) # Assuming not_done
            )

parser = argparse.ArgumentParser()
parser.add_argument("--ill", action='store_true')
parser.add_argument("--seed", default=2022, type=int)
parser.add_argument("--exp", default='exp_1', type=str)
parser.add_argument("--x", default=0., type=float)
parser.add_argument("--y", default=0., type=float)
parser.add_argument("--eta", default=2.5, type=float)
parser.add_argument('--device', default=0, type=int)
parser.add_argument("--dir", default='whole_grad', type=str)
parser.add_argument("--r_fun", default='no', type=str) # This will be less relevant now
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument("--mode", default='whole_grad', type=str)
args = parser.parse_args()

# r_fun_std is no longer used for reward generation with the new logic
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device}")
    print(f"Using CUDA GPU: cuda:{args.device}")
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
            
            current_mode_rewards = Normal(mean_tensor, std_tensor).sample()
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
np.random.seed(seed) # For np.random.choice

num_data = int(10000)
# Call generate_data without radius_inner/outer as they are not used for new reward logic
data_sampler = generate_data(num_data, device=device, clip_val=1.0)


state_dim = 2
action_dim = 2
max_action = 1.0 # For agent action clipping, not reward clipping

discount = 0.99
tau = 0.005
model_type = 'MLP'
T = 50
beta_schedule = 'vp'

num_epochs = 1000
batch_size = 100
iterations = 0
if batch_size > 0 and num_data > 0:
    iterations = num_data // batch_size
else:
    print("Warning: batch_size or num_data is 0, setting iterations to 0.")


#img_dir = f'toy_imgs/{args.dir}_SI252_targeted_rewards' # Updated dir name
img_dir = 'toy_imgs'
os.makedirs(img_dir, exist_ok=True)

num_eval = 100

fig, axs = plt.subplots(1, 5, figsize=(5.5 * 5, 5))
axis_lim = 1.1

axs[0].clear()
data_sampler_for_axs0 = data_sampler

if data_sampler_for_axs0 is not None and \
   hasattr(data_sampler_for_axs0, 'action') and data_sampler_for_axs0.action.nelement() > 0 and \
   hasattr(data_sampler_for_axs0, 'reward') and data_sampler_for_axs0.reward.nelement() > 0 and \
   data_sampler_for_axs0.action.shape[0] == data_sampler_for_axs0.reward.shape[0]:

    action_samples_np = data_sampler_for_axs0.action.cpu().numpy()
    reward_values_np = data_sampler_for_axs0.reward.cpu().numpy().flatten()

    # Determine color map limits based on expected reward range for better visualization
    # Expected range roughly from (1-3*0.5) to (5+3*0.5) -> -0.5 to 6.5
    vmin_reward = min(1.0 - 3*0.5, np.min(reward_values_np)) # Allow for actual min if lower
    vmax_reward = max(5.0 + 3*0.5, np.max(reward_values_np)) # Allow for actual max if higher


    scatter_plot_on_axs0 = axs[0].scatter(
        action_samples_np[:, 0],
        action_samples_np[:, 1],
        c=reward_values_np,
        cmap='viridis', # Or 'coolwarm' might show high/low well
        alpha=0.6, # slightly more opaque
        s=5,
        vmin=vmin_reward, # Set color limits
        vmax=vmax_reward
    )
    fig.colorbar(scatter_plot_on_axs0, ax=axs[0], label='Reward Value (Targeted)')

else:
    error_message = "Data for Ground Truth plot (axs[0]) is invalid or empty.\n"
    # ... (error message details as before)
    axs[0].text(0.5, 0.5, error_message, ha='center', va='center', color='red', transform=axs[0].transAxes, fontsize=8)

axs[0].set_title('Ground Truth: "SI252" (Targeted Rewards)', fontsize=13) # Adjusted title
axs[0].set_xlabel('Action_x', fontsize=12)
axs[0].set_ylabel('Action_y', fontsize=12)
axs[0].set_xlim(-axis_lim, axis_lim)
axs[0].set_ylim(-axis_lim, axis_lim)
axs[0].set_aspect('equal', adjustable='box')
axs[0].grid(True, linestyle='--', alpha=0.7)


# --- Agent Training and Plotting Sections (Copied from previous, ensure they run) ---
titles = ['TD3+BC (QL-MLE)', 'BCQ (QL-CVAE)', 'BEAR-MMD (QL-MMD)', 'Diffusion-QL']

# QL-MLE
try:
    from toy_experiments.ql_mle import QL_MLE
    agent_mle = QL_MLE(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                       device=device, discount=discount, tau=tau, eta=eta,
                       hidden_dim=hidden_dim, lr=lr, r_fun=None)
    if iterations > 0:
        print("Training QL-MLE...")
        for i in range(1, num_epochs + 1):
            agent_mle.train(data_sampler, iterations=iterations, batch_size=batch_size)
            if i % (num_epochs // 10 if num_epochs >=10 else 1) == 0 : print(f'QL-MLE Epoch: {i}')
    else: print("Skipping QL-MLE training as iterations is 0.")
    new_state_mle = torch.zeros((num_eval, 2), device=device)
    new_action_mle = agent_mle.actor.sample(new_state_mle).detach().cpu().numpy()
    axs[1].clear()
    axs[1].scatter(new_action_mle[:, 0], new_action_mle[:, 1], alpha=0.3, color='#d62728')
    axs[1].set_title(titles[0], fontsize=15); axs[1].set_xlim(-axis_lim, axis_lim); axs[1].set_ylim(-axis_lim, axis_lim)
    axs[1].set_xlabel('x', fontsize=12); axs[1].set_ylabel('y', fontsize=12)
    axs[1].set_aspect('equal', adjustable='box'); axs[1].grid(True, linestyle='--', alpha=0.7)
except ImportError: print("QL_MLE not found, skipping plot.")
except Exception as e:
    print(f"Error QL-MLE: {e}")
    axs[1].text(0.5, 0.5, f"{titles[0]}\n(Error)", ha='center', va='center', transform=axs[1].transAxes, color='red')

# QL-CVAE
try:
    from toy_experiments.ql_cvae import QL_CVAE
    agent_cvae = QL_CVAE(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                         device=device, discount=discount, tau=tau,
                         hidden_dim=hidden_dim, lr=lr, r_fun=None)
    if iterations > 0:
        print("Training QL-CVAE...")
        for i in range(1, num_epochs + 1):
            agent_cvae.train(data_sampler, iterations=iterations, batch_size=batch_size)
            if i % (num_epochs // 10 if num_epochs >=10 else 1) == 0: print(f'QL-CVAE Epoch: {i}')
    else: print("Skipping QL-CVAE training.")
    new_state_cvae = torch.zeros((num_eval, 2), device=device)
    new_action_cvae = agent_cvae.vae.sample(new_state_cvae).detach().cpu().numpy()
    axs[2].clear()
    axs[2].scatter(new_action_cvae[:, 0], new_action_cvae[:, 1], alpha=0.3, color='#d62728')
    axs[2].set_title(titles[1], fontsize=15); axs[2].set_xlim(-axis_lim, axis_lim); axs[2].set_ylim(-axis_lim, axis_lim)
    axs[2].set_xlabel('x', fontsize=12); axs[2].set_ylabel('y', fontsize=12)
    axs[2].set_aspect('equal', adjustable='box'); axs[2].grid(True, linestyle='--', alpha=0.7)
except ImportError: print("QL_CVAE not found, skipping plot.")
except Exception as e:
    print(f"Error QL-CVAE: {e}")
    axs[2].text(0.5, 0.5, f"{titles[1]}\n(Error)", ha='center', va='center', transform=axs[2].transAxes, color='red')

# QL-MMD
try:
    from toy_experiments.ql_mmd import QL_MMD
    agent_mmd = QL_MMD(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                       device=device, discount=discount, tau=tau,
                       hidden_dim=hidden_dim, lr=lr, r_fun=None)
    if iterations > 0:
        print("Training QL-MMD...")
        for i in range(1, num_epochs + 1):
            agent_mmd.train(data_sampler, iterations=iterations, batch_size=batch_size)
            if i % (num_epochs // 10 if num_epochs >=10 else 1) == 0: print(f'QL-MMD Epoch: {i}')
    else: print("Skipping QL-MMD training.")
    new_state_mmd = torch.zeros((num_eval, 2), device=device)
    new_action_mmd = agent_mmd.actor.sample(new_state_mmd).detach().cpu().numpy()
    axs[3].clear()
    axs[3].scatter(new_action_mmd[:, 0], new_action_mmd[:, 1], alpha=0.3, color='#d62728')
    axs[3].set_title(titles[2], fontsize=15); axs[3].set_xlim(-axis_lim, axis_lim); axs[3].set_ylim(-axis_lim, axis_lim)
    axs[3].set_xlabel('x', fontsize=12); axs[3].set_ylabel('y', fontsize=12)
    axs[3].set_aspect('equal', adjustable='box'); axs[3].grid(True, linestyle='--', alpha=0.7)
except ImportError: print("QL_MMD not found, skipping plot.")
except Exception as e:
    print(f"Error QL-MMD: {e}")
    axs[3].text(0.5, 0.5, f"{titles[2]}\n(Error)", ha='center', va='center', transform=axs[3].transAxes, color='red')

# QL-Diffusion
try:
    from toy_experiments.ql_diffusion import QL_Diffusion
    agent_diffusion = QL_Diffusion(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
                                   device=device, discount=discount, tau=tau, eta=eta,
                                   beta_schedule=beta_schedule, n_timesteps=T, model_type=model_type,
                                   hidden_dim=hidden_dim, lr=lr, r_fun=None, mode=args.mode)
    if iterations > 0:
        print("Training QL-Diffusion...")
        for i in range(1, num_epochs + 1):
            b_loss, q_loss = agent_diffusion.train(data_sampler, iterations=iterations, batch_size=batch_size)
            if i % (num_epochs // 10 if num_epochs >=10 else 1) == 0:
                 print(f'QL-Diffusion Epoch: {i} B_loss {b_loss:.3f} Q_loss {q_loss:.3f}')
    else: print("Skipping QL-Diffusion training.")
    new_state_diffusion = torch.zeros((num_eval, 2), device=device)
    new_action_diffusion = agent_diffusion.actor.sample(new_state_diffusion).detach().cpu().numpy()
    axs[4].clear()
    axs[4].scatter(new_action_diffusion[:, 0], new_action_diffusion[:, 1], alpha=0.3, color='#d62728')
    axs[4].set_title(titles[3], fontsize=15); axs[4].set_xlim(-axis_lim, axis_lim); axs[4].set_ylim(-axis_lim, axis_lim)
    axs[4].set_xlabel('x', fontsize=12); axs[4].set_ylabel('y', fontsize=12)
    axs[4].set_aspect('equal', adjustable='box'); axs[4].grid(True, linestyle='--', alpha=0.7)
except ImportError: print("QL_Diffusion not found, skipping plot.")
except Exception as e:
    print(f"Error QL-Diffusion: {e}")
    axs[4].text(0.5, 0.5, f"{titles[3]}\n(Error)", ha='center', va='center', transform=axs[4].transAxes, color='red')
# --- End Agent Plotting ---


file_name = f'ql_all_T{T}_eta{eta}_lr{lr}_hd{hidden_dim}_mode_{args.mode}' # Shorter r_fun
file_name += f'_sd{args.seed}_modified3.pdf'

fig.tight_layout()
fig.savefig(os.path.join(img_dir, file_name))
print(f"Saved plot to {os.path.join(img_dir, file_name)}")