import argparse
import gym
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import d4rl

from agents.model import MLP
from agents.diffusion_modified import Diffusion
from utils import utils

#    'K_denoising_steps': 5,   

DPPO_DEFAULT_HPARAMS = {
    'gamma_ENV': 0.99,  
    'gamma_DENOISE': 0.99,  
    'K_denoising_steps': 5,  
    'K_prime_fine_tune': 10,  
    'ppo_epsilon': 0.01,  
    'gae_lambda': 0.95, 
    'num_ppo_updates': 5,  
    'ppo_batch_size': 50000,  
    'actor_lr': 1e-4,  
    'critic_lr': 1e-3,  
    'actor_mlp_dims': [512, 512, 512],  
    'critic_mlp_dims': [256, 256, 256], 
    'beta_schedule': 'vp', 
    'min_sigma_exp': 0.1, 
    'min_sigma_prob': 0.1,
    't_dim': 16, 
}

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=[256, 256, 256]):
        super(ValueNetwork, self).__init__()
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Mish()) 
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)

class DPPOAgent:
    def __init__(self, state_dim, action_dim, max_action, device, pre_trained_actor_path, hparams):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.hparams = hparams

        self.actor_model_mlp = MLP(state_dim=state_dim, action_dim=action_dim, device=device, t_dim=hparams['t_dim'])
        self.actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=self.actor_model_mlp,
            max_action=max_action,
            beta_schedule=hparams['beta_schedule'],
            n_timesteps=hparams['K_denoising_steps'],
        ).to(device)

        self.actor.set_dppo_sigmas(hparams['min_sigma_exp'], hparams['min_sigma_prob'])

        print(f"Loading pre-trained actor from: {pre_trained_actor_path}")
        self.actor.load_state_dict(torch.load(pre_trained_actor_path, map_location=device), strict=False)

        self.actor_old_model_mlp = MLP(state_dim=state_dim, action_dim=action_dim, device=device, t_dim=hparams['t_dim'])
        self.actor_old = Diffusion(
            state_dim=state_dim, action_dim=action_dim, model=self.actor_old_model_mlp,
            max_action=max_action, beta_schedule=hparams['beta_schedule'], n_timesteps=hparams['K_denoising_steps']
        ).to(device)
        self.actor_old.set_dppo_sigmas(hparams['min_sigma_exp'], hparams['min_sigma_prob'])
        self.actor_old.load_state_dict(self.actor.state_dict()) 

        self.value_net = ValueNetwork(state_dim, hparams['critic_mlp_dims']).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=hparams['actor_lr'])
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=hparams['critic_lr'])

        self.gamma_env = hparams['gamma_ENV']
        self.gamma_denoise = hparams['gamma_DENOISE']
        self.gae_lambda = hparams['gae_lambda']
        self.ppo_epsilon = hparams['ppo_epsilon']
        self.K_total = hparams['K_denoising_steps']
        self.K_prime = hparams['K_prime_fine_tune'] 

        if self.K_prime < self.K_total:
            print(f"Note: DPPO will fine-tune only last K' steps ({self.K_prime}/{self.K_total}).")
            print("This is handled by summing policy gradient only over the last K_prime steps.")

    def collect_experience(self, env, num_env_steps_to_collect):
        """
        在环境中进行交互，收集经验。
        这会模拟 DPPO 论文中 Algorithm 1 的第 6-13 行。
        现在会存储每个去噪步骤的中间动作及其旧策略的对数似然。
        """
        
        traj_data = collections.defaultdict(list)
        episode_rewards = []

        current_env_steps = 0
        while current_env_steps < num_env_steps_to_collect:
            obs=env.reset()
            #obs, _ = env.reset()
            episode_reward = 0
            done = False
            env_step_count = 0

            while not done and env_step_count < env.spec.max_episode_steps:
                state_tensor = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
                
                with torch.no_grad(): 
                    action_final_denoised, actions_denoised_chain, log_probs_denoising_chain_old = \
                        self.actor_old.sample(state_tensor)
                    
                action_final_denoised_np = action_final_denoised.cpu().numpy().flatten()
                
                #next_obs, reward, terminated, truncated, _ = env.step(action_final_denoised_np)
                #done = terminated or truncated
                next_obs, reward, done, _ = env.step(action_final_denoised_np)

                episode_reward += reward
                current_env_steps += 1
                env_step_count += 1

                traj_data['env_states'].append(obs)
                traj_data['actions_final'].append(action_final_denoised_np)
                traj_data['env_rewards'].append(reward)
                traj_data['env_next_states'].append(next_obs)
                traj_data['env_dones'].append(done)
                traj_data['denoising_actions_chains'].append(actions_denoised_chain.squeeze(0).cpu().numpy())
                traj_data['denoising_log_probs_old_chains'].append(log_probs_denoising_chain_old.squeeze(0).cpu().numpy())

                obs = next_obs
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            if current_env_steps >= num_env_steps_to_collect:
                break
        
        return traj_data, episode_rewards

    def compute_advantages(self, traj_data):
        """
        计算 Advantage estimates using GAE.
        对应 DPPO 论文 Algorithm 1 的第 14 行。
        DPPO (A2) 提到只计算 k=0 的 Advantage，然后对去噪步进行折扣。
        这里的 Value Network 只依赖于环境状态 s_t。
        """
        env_states = torch.FloatTensor(np.array(traj_data['env_states'])).to(self.device)
        env_rewards = torch.FloatTensor(np.array(traj_data['env_rewards'])).to(self.device).unsqueeze(1)
        env_dones = torch.FloatTensor(np.array(traj_data['env_dones'])).to(self.device).unsqueeze(1)
        env_next_states = torch.FloatTensor(np.array(traj_data['env_next_states'])).to(self.device)

        with torch.no_grad():
            values = self.value_net(env_states)
            next_values = self.value_net(env_next_states)

        advantages_env = torch.zeros_like(env_rewards, device=self.device)
        last_gae_lambda = 0

        for t in reversed(range(len(env_rewards))):
            if env_dones[t]:
                next_value_for_td = 0
                last_gae_lambda = 0.0
            else:
                next_value_for_td = next_values[t]
            
            delta = env_rewards[t] + self.gamma_env * next_value_for_td - values[t]
            advantages_env[t] = delta + self.gamma_env * self.gae_lambda * last_gae_lambda * (1 - env_dones[t])
            last_gae_lambda = advantages_env[t]

        return advantages_env, values

    def update_policy(self, traj_data, advantages_env, values_env, iter):
        """
        根据收集到的经验和优势函数更新策略和价值函数。
        对应 DPPO 论文 Algorithm 1 的第 15-21 行。
        实现 DPPO 的两层 MDP 策略梯度。
        """
        env_states_all = torch.FloatTensor(np.array(traj_data['env_states'])).to(self.device) 
        actions_final_all = torch.FloatTensor(np.array(traj_data['actions_final'])).to(self.device) 
        denoising_actions_chains_all = torch.FloatTensor(np.array(traj_data['denoising_actions_chains'])).to(self.device)
        denoising_log_probs_old_chains_all = torch.FloatTensor(np.array(traj_data['denoising_log_probs_old_chains'])).to(self.device)

        advantages_env_detached = advantages_env.detach()
        values_env_detached = values_env.detach()

        for _ in range(self.hparams['num_ppo_updates']):
            indices = torch.randperm(len(env_states_all)).to(self.device)
            num_samples = len(env_states_all)
            batch_size = self.hparams['ppo_batch_size']

            if batch_size > num_samples:
                batch_size = num_samples

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_env_states = env_states_all[batch_indices]
                batch_actions_final = actions_final_all[batch_indices] # a_t^0
                batch_denoising_actions_chains = denoising_actions_chains_all[batch_indices] # [a_t^{K-1}, ..., a_t^0]
                batch_denoising_log_probs_old_chains = denoising_log_probs_old_chains_all[batch_indices]
                batch_advantages_env = advantages_env_detached[batch_indices]
                batch_values_env = values_env_detached[batch_indices]

                total_actor_loss = 0
                
                fine_tune_start_idx_in_chain = self.K_total - self.K_prime

                for k_idx_in_chain in range(self.K_total):

                    if self.K_prime < self.K_total and k_idx_in_chain < fine_tune_start_idx_in_chain:
                        continue 
                    
                    if k_idx_in_chain == 0: 
                        x_t_noisy_for_current_step = torch.randn_like(batch_actions_final)
                        x_t_prev_sampled_for_current_step = batch_denoising_actions_chains[:, 0, :]
                        # t_step_for_model = torch.full((batch_size,), self.K_total - 1, device=self.device, dtype=torch.long)
                        t_step_for_model = torch.full((batch_size,), self.K_total - 1 - k_idx_in_chain, device=self.device, dtype=torch.long)
                    else:
                        x_t_noisy_for_current_step = batch_denoising_actions_chains[:, k_idx_in_chain - 1, :]
                        x_t_prev_sampled_for_current_step = batch_denoising_actions_chains[:, k_idx_in_chain, :]
                        # t_step_for_model = torch.full((batch_size,), self.K_total - k_idx_in_chain, device=self.device, dtype=torch.long)
                        t_step_for_model = torch.full((batch_size,), self.K_total - 1 - k_idx_in_chain, device=self.device, dtype=torch.long)
                    log_prob_new = self.actor.get_log_prob(
                        x_t_prev_sampled=x_t_prev_sampled_for_current_step,
                        x_t_noisy=x_t_noisy_for_current_step,
                        t_step=t_step_for_model,
                        state=batch_env_states
                    )

                    log_prob_old = batch_denoising_log_probs_old_chains[:, k_idx_in_chain, :]

                    ratio = torch.exp(log_prob_new - log_prob_old)

                    denoising_step_k = self.K_total - 1 - k_idx_in_chain
                    denoise_discounted_advantage = batch_advantages_env * (self.gamma_denoise ** denoising_step_k)

                    surr1 = ratio * denoise_discounted_advantage
                    surr2 = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * denoise_discounted_advantage

                    total_actor_loss += (-torch.min(surr1, surr2)).mean()

                actor_loss = total_actor_loss / self.K_prime if self.K_prime < self.K_total else total_actor_loss / self.K_total
                
                if iter >= 20:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                
                value_loss = F.mse_loss(self.value_net(batch_env_states), batch_advantages_env + batch_values_env) # Target is Return
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

        self.actor_old.load_state_dict(self.actor.state_dict())


    def train_online(self, env_name, pre_trained_model_path, num_online_iterations, steps_per_iteration, eval_episodes=10):
        env = gym.make(env_name)
        env.seed(self.hparams['seed'])
        episode_rewards_history = []

        for iteration in range(num_online_iterations):
            print(f"\n--- Online Fine-tuning Iteration {iteration+1}/{num_online_iterations} ---")
            traj_data, current_episode_rewards = self.collect_experience(env, num_env_steps_to_collect=steps_per_iteration)
            
            episode_rewards_history.extend(current_episode_rewards)
            print(f"Collected {len(traj_data['env_states'])} environment steps. Average reward: {np.mean(current_episode_rewards):.2f}")

            advantages, values = self.compute_advantages(traj_data)

            self.update_policy(traj_data, advantages, values, iteration)

            eval_env = gym.make(env_name)
            eval_env.seed(self.hparams['seed'] + 100)
            eval_scores = []
            total_success_rate = 0 

            original_min_sigma_exp = self.actor.min_sigma_exp
            self.actor.set_dppo_sigmas(0.0, self.hparams['min_sigma_prob'])
            
            for _ in range(eval_episodes):
                #obs, _ = eval_env.reset()
                obs = eval_env.reset()
                done = False
                total_eval_reward = 0
                while not done:
                    state_tensor = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
                    with torch.no_grad():
                        action, _, _ = self.actor.sample(state_tensor, apply_sigma_exp=False)
                    action_np = action.cpu().numpy().flatten()

                    # next_obs, reward, terminated, truncated, info = eval_env.step(action_np)
                    # done = terminated or truncated
                    next_obs, reward, done, info = eval_env.step(action_np)
                    
                    total_eval_reward += reward
                    obs = next_obs
                    
                    if 'is_success' in info:
                        total_success_rate += info['is_success']
            
                eval_scores.append(total_eval_reward)
            self.actor.set_dppo_sigmas(original_min_sigma_exp, self.hparams['min_sigma_prob'])
            
            avg_eval_reward = np.mean(eval_scores)

            normalized_avg_score = None
            if hasattr(eval_env, 'get_normalized_score'):
                normalized_avg_score = eval_env.get_normalized_score(avg_eval_reward)
                print(f"Evaluation after iteration {iteration+1}: Average Reward = {avg_eval_reward:.2f}, Normalized Score = {normalized_avg_score:.2f}")
            else:
                print(f"Evaluation after iteration {iteration+1}: Average Reward = {avg_eval_reward:.2f}")
            
            if total_success_rate > 0:
                print(f"Success Rate: {total_success_rate / eval_episodes * 100:.2f}%")


        return episode_rewards_history, self.actor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="halfcheetah-medium-v2", type=str)
    parser.add_argument("--pre_trained_model_dir", default="custom_results/halfcheetah-medium-v2/custom_run", type=str)
    #parser.add_argument("--online_iterations", default=100, type=int)
    parser.add_argument("--online_iterations", default=50, type=int)
    parser.add_argument("--steps_per_iteration", default=2000, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output_dir", default="dppo_fine_tune_results", type=str) 

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env_temp = gym.make(args.env_name)
    state_dim = env_temp.observation_space.shape[0]
    action_dim = env_temp.action_space.shape[0]
    max_action = float(env_temp.action_space.high[0])
    env_temp.close()

    print(f"Environment: {args.env_name}, State Dim: {state_dim}, Action Dim: {action_dim}, Max Action: {max_action}")

    hparams_dppo = DPPO_DEFAULT_HPARAMS.copy()
    hparams_dppo['seed'] = args.seed 

    if "halfcheetah" in args.env_name.lower() or "hopper" in args.env_name.lower() or "walker2d" in args.env_name.lower():
        #hparams_dppo['K_denoising_steps'] = 20
        hparams_dppo['K_denoising_steps'] = 5
        hparams_dppo['K_prime_fine_tune'] = 10

    pre_trained_actor_path = os.path.join(args.pre_trained_model_dir, "actor_final_bc_model.pth") 
    if not os.path.exists(pre_trained_actor_path):
        print(f"Error: Pre-trained model not found at {pre_trained_actor_path}")
        print("Please ensure 'pre_trained_model_dir' is correct and the model file exists.")
        exit()

    dppo_agent = DPPOAgent(state_dim, action_dim, max_action, device, pre_trained_actor_path, hparams_dppo)

    results_dir = os.path.join(args.output_dir, f"{args.env_name}_seed{args.seed}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    all_rewards, final_policy_model = dppo_agent.train_online(
        args.env_name, pre_trained_actor_path,
        num_online_iterations=args.online_iterations,
        steps_per_iteration=args.steps_per_iteration,
        eval_episodes=args.eval_episodes
    )

    rewards_file_path = os.path.join(results_dir, "dppo_online_rewards_bc.txt")
    with open(rewards_file_path, "w") as f:
        for reward in all_rewards:
            f.write(f"{reward}\n")
    print(f"\nOnline rewards saved to: {rewards_file_path}")

    final_policy_model_path = os.path.join(results_dir, "dppo_fine_tuned_actor_bc.pth")
    torch.save(final_policy_model.state_dict(), final_policy_model_path)
    print(f"Fine-tuned DPPO actor saved to: {final_policy_model_path}")