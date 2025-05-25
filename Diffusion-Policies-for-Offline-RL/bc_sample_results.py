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

import numpy as np
import torch


def generate_data(num_total_samples, device='cpu',
                  theta_R_coefficient=None,
                  # 正弦奖励参数
                  reward_sin_amplitude=0.5,
                  reward_sin_frequency_on_R=10.0 * np.pi, # R在[0,1]范围内时对应的频率
                  reward_sin_phase=0.0,
                  reward_sin_offset=0.5):
    """
    根据极坐标采样方法生成数据点，并计算基于半径R的正弦函数奖励。
    - 半径 R 在 [0, 1) 之间均匀分布。
    - 角度 theta 与 R 线性相关: theta = theta_R_coefficient * R。
    - Reward 是 R 的正弦函数: reward = A * sin(freq_R * R + phase) + offset。
    
    默认的 theta_R_coefficient 为 6.0 * np.pi (产生3圈螺旋)。
    默认的奖励参数使得 reward 在 [0,1] 范围内振荡一个周期。

    参数:
        num_total_samples (int): 要生成的总数据点数。
        device (str, optional): 创建张量的设备 ('cpu' 或 'cuda')。
                                默认为 'cpu'。
        theta_R_coefficient (float, optional): 线性关系 theta = coeff * R 的系数。
                                            如果为 None，则默认为 6.0 * np.pi。
        reward_sin_amplitude (float, optional): 正弦奖励的振幅 (A)。默认为 0.5。
        reward_sin_frequency_on_R (float, optional): R在[0,1]区间内，正弦波的频率 (freq_R)。
                                                 默认为 2.0 * np.pi (一个完整周期)。
        reward_sin_phase (float, optional): 正弦奖励的相位 (phase)。默认为 0.0。
        reward_sin_offset (float, optional): 正弦奖励的垂直偏移 (offset)。默认为 0.5。

    返回:
        Data_Sampler: 一个具名元组，包含 state、action (采样的x,y点)、
                      reward 张量以及设备信息。
    """

    if theta_R_coefficient is None:
        theta_R_coefficient = (6.0 * np.pi) # 保持用户指定的默认螺旋参数

    if num_total_samples <= 0:
        action = torch.empty((0, 2), dtype=torch.float32, device=device)
        state = torch.empty((0, 2), dtype=torch.float32, device=device)
        calculated_reward = torch.empty((0, 1), dtype=torch.float32, device=device)
        return Data_Sampler(state=state, action=action, reward=calculated_reward, device=device)

    # 1. 生成 R 和 theta
    R = torch.rand(num_total_samples, device=device, dtype=torch.float32) # Shape: (N,)
    theta = theta_R_coefficient * R

    # 2. 转换为笛卡尔坐标
    x = R * torch.cos(theta)
    y = R * torch.sin(theta)

    action = torch.stack((x, y), dim=1) # Shape: (N, 2)

    # 3. State 张量 (默认为零)
    state = torch.zeros_like(action, dtype=torch.float32, device=device) # Shape: (N, 2)

    # 4. 计算基于 R 的正弦奖励
    # reward = amplitude * sin(frequency_R * R + phase) + offset
    # R 的形状是 (num_total_samples,)
    calculated_reward = reward_sin_amplitude * torch.sin(
        reward_sin_frequency_on_R * R + reward_sin_phase
    ) + reward_sin_offset  # Shape: (N,)

    # 将奖励张量调整为 (num_total_samples, 1)
    calculated_reward = calculated_reward.unsqueeze(1) # Shape: (N, 1)

    return Data_Sampler(state=state, action=action, reward=calculated_reward, device=device)

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

# (在您的主脚本中，当 data_sampler 已经被 generate_complex_data 创建之后)

# 2. 立刻显示Ground Truth图像 (修改后的版本)
print("Displaying generated ground truth data distribution with rewards...")
plt.figure(figsize=(8, 7)) # 可以调整图像大小，为colorbar留出空间
axis_lim_gt = 1.1

# 正确检查 data_sampler 是否有效并且包含数据
if data_sampler is not None and \
   hasattr(data_sampler, 'action') and data_sampler.action.nelement() > 0 and \
   hasattr(data_sampler, 'reward') and data_sampler.reward.nelement() > 0 and \
   data_sampler.action.shape[0] == data_sampler.reward.shape[0]: # 确保动作和奖励数量匹配

    action_samples_to_plot = data_sampler.action 
    reward_values_to_plot = data_sampler.reward

    action_samples_to_plot_np = action_samples_to_plot.cpu().numpy()
    reward_values_to_plot_np = reward_values_to_plot.cpu().numpy().flatten() # flatten rewards for c
    
    print(f"Plotting {action_samples_to_plot_np.shape[0]} generated ground truth samples, colored by reward.")
    
    # 使用 reward_values_to_plot_np作为颜色参数 c
    # cmap='viridis' 是一个常用的颜色映射表
    scatter_plot = plt.scatter(
        action_samples_to_plot_np[:, 0], 
        action_samples_to_plot_np[:, 1], 
        c=reward_values_to_plot_np,  # 按奖励值着色
        cmap='viridis',              # 选择颜色映射
        alpha=0.5,                   # 透明度调高一点，更容易看清重叠点
        s=10                         # 点的大小可以适当调整
    )
    
    # 添加颜色条
    plt.colorbar(scatter_plot, label='Reward Value')
    
else:
    plt.text(0.5, 0.5, "Data sampler is empty, invalid, or action/reward mismatch for GT display", 
             ha='center', va='center', color='red')

plt.title('Ground Truth: Actions Colored by Reward', fontsize=15) # 修改标题
plt.xlabel('Action_x', fontsize=12)
plt.ylabel('Action_y', fontsize=12)
plt.xlim(-axis_lim_gt, axis_lim_gt)
plt.ylim(-axis_lim_gt, axis_lim_gt)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show() 
print("Ground truth display closed. Proceeding with model training...")
# --- Ground Truth 即时显示结束 --- 