import matplotlib.pyplot as plt
import numpy as np
import re

def parse_rewards_from_file(filepath, file_type='simple'):
    """
    从文件中解析奖励数据。
    file_type='simple' 表示文件只包含每行的奖励数值。
    file_type='edp' 表示文件包含 'Raw Return = X' 格式的行。
    """
    rewards = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if file_type == 'simple':
                try:
                    rewards.append(float(line))
                except ValueError:
                    # Skip lines that are not float numbers
                    continue
            elif file_type == 'edp':
                match = re.search(r'Raw Return = ([\d.]+)', line)
                if match:
                    try:
                        rewards.append(float(match.group(1)))
                    except ValueError:
                        continue
    return rewards

def calculate_cumulative_rewards(rewards):
    """
    计算累积奖励。
    """
    return np.cumsum(rewards)

# 定义文件路径和对应的类型
file_data = {
    'BC-DPPO': {'path': 'BC-DPPO.txt', 'type': 'simple'},
    'DiffQL': {'path': 'DiffQL.txt', 'type': 'simple'},
    'EDP-DPPO': {'path': 'EDP-DPPO.txt', 'type': 'simple'},
    'EDP': {'path': 'EDP.txt', 'type': 'edp'}
}

# 存储所有方法的累积奖励
cumulative_rewards_data = {}

# 解析并计算累积奖励
for name, info in file_data.items():
    rewards = parse_rewards_from_file(info['path'], info['type'])
    cumulative_rewards = calculate_cumulative_rewards(rewards)
    cumulative_rewards_data[name] = cumulative_rewards

# 绘制图像
plt.figure(figsize=(12, 7))

# 修改绘图循环，只绘制前100个点
for name, cum_rewards in cumulative_rewards_data.items():
    # 限制为前100个点
    episodes_to_plot = min(100, len(cum_rewards))
    plt.plot(np.arange(1, episodes_to_plot + 1), cum_rewards[:episodes_to_plot], label=name)

plt.xlabel('Number of Episodes')
plt.ylabel('Cumulative Reward')
plt.title('HalfCheetah-v2 Cumulative Rewards Comparison')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()