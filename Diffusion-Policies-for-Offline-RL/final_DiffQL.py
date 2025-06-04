import argparse
import gym
import numpy as np
import os
import torch
import json
import collections 

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter 

from main import hyperparameters as default_hyperparameters

def train_agent_and_collect_rewards(env, state_dim, action_dim, max_action, device, output_dir, args):
    dataset = d4rl.qlearning_dataset(env)
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    utils.print_banner('Loaded buffer')

    from agents.ql_diffusion import Diffusion_QL as Agent
    agent = Agent(state_dim=state_dim,
                  action_dim=action_dim,
                  max_action=max_action,
                  device=device,
                  discount=args.discount,
                  tau=args.tau,
                  max_q_backup=args.max_q_backup,
                  beta_schedule=args.beta_schedule,
                  n_timesteps=args.T,
                  eta=args.eta,
                  lr=args.lr,
                  lr_decay=args.lr_decay,
                  lr_maxt=args.num_epochs,
                  grad_norm=args.gn)

    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch 
    utils.print_banner(f"Training Start", separator="*", num_star=90)

    while training_iters < max_timesteps:
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(data_sampler,
                                  iterations=iterations,
                                  batch_size=args.batch_size,
                                  log_writer=None)
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        print(f"--- Epoch {curr_epoch}/{args.num_epochs} ---")
        print(f"Training Iterations: {training_iters}")
        print(f"BC Loss: {np.mean(loss_metric['bc_loss']):.4f}")
        print(f"QL Loss: {np.mean(loss_metric['ql_loss']):.4f}")
        print(f"Actor Loss: {np.mean(loss_metric['actor_loss']):.4f}")
        print(f"Critic Loss: {np.mean(loss_metric['critic_loss']):.4f}")

        raw_episode_rewards, _, normalized_avg_score, _ = eval_policy_custom(agent, args.env_name, args.seed,
                                                                              eval_episodes=args.eval_episodes)
        print(f"Online Evaluation: Avg Normalized Reward = {normalized_avg_score:.2f}")

    print("Offline training complete.")
    return agent, raw_episode_rewards

def eval_policy_custom(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100) 

    scores = [] 
    for _ in range(eval_episodes):
        traj_return = 0.
        state, done = eval_env.reset(), False 
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward  
        scores.append(traj_return) 

    avg_reward = np.mean(scores) 
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores] 
    avg_norm_score = eval_env.get_normalized_score(avg_reward) 
    std_norm_score = np.std(normalized_scores) 

    utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    return scores, std_reward, avg_norm_score, std_norm_score 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default='custom_run', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument("--env_name", default="halfcheetah-medium-v2", type=str)
    parser.add_argument("--dir", default="custom_results", type=str)
    parser.add_argument("--seed", default=0, type=int) 
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

    parser.add_argument("--batch_size", default=256, type=int) 
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true') 

    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)

    parser.add_argument("--algo", default="ql", type=str)
    parser.add_argument("--ms", default='offline', type=str, help="['online', 'offline']")
    parser.add_argument("--eval_episodes", default=10, type=int) 

    args = parser.parse_args()

    if args.env_name in default_hyperparameters:
        hparams = default_hyperparameters[args.env_name]
        args.num_epochs = hparams['num_epochs'] 
        args.eval_freq = hparams['eval_freq'] 
        args.lr = hparams['lr'] 
        args.eta = hparams['eta'] 
        args.max_q_backup = hparams['max_q_backup'] 
        args.reward_tune = hparams['reward_tune'] 
        args.gn = hparams['gn'] 
        args.top_k = hparams['top_k'] 
        print(f"Loaded hyperparameters for {args.env_name}: {hparams}")
    else:
        print(f"Warning: No specific hyperparameters found for {args.env_name}. Using default argparse values.")

    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        args.device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        args.device = torch.device("cpu") 
        print("Using CPU")
    print(f"Selected device: {args.device}")

    args.output_dir = f'{args.dir}/{args.env_name}/{args.exp}'
    os.makedirs(args.output_dir, exist_ok=True)
    utils.print_banner(f"Saving location: {args.output_dir}")

    env = gym.make(args.env_name) 
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}, max_action: {max_action}")

    if args.algo == 'ql':
        trained_agent, final_eval_rewards = train_agent_and_collect_rewards(
            env, state_dim, action_dim, max_action, args.device, args.output_dir, args
        )
    else:
        raise ValueError(f"Algorithm '{args.algo}' is not supported in this custom script. Only 'ql' is implemented.")

    rewards_file_path = os.path.join(args.output_dir, "DiffQL_results.txt")
    with open(rewards_file_path, "w") as f:
        for reward in final_eval_rewards:
            f.write(f"{reward}\n")
    print(f"Online episode rewards saved to {rewards_file_path}")

    trained_agent.save_model(args.output_dir, "final_trained_model")
    print(f"Final trained model saved to {args.output_dir}/final_trained_model.pth")