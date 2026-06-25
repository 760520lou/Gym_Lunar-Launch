import time
import os
import random
from collections import deque, namedtuple

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

import utils  # 请确保确保 utils.py 存在且包含必要函数

# ---------------------------
# GPU 配置（按需分配 GPU 内存）
# ---------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_memory_growth(gpu, True)
        print("检测到 GPU，已配置按需分配 GPU 内存。")
    except RuntimeError as e:
        print("设置 GPU 时出错：", e)
else:
    print("未检测到 GPU，将使用 CPU。")

# ---------------------------
# 设置随机种子
# ---------------------------
SEED = utils.SEED
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------
# 超参数定义
# ---------------------------
MEMORY_SIZE = 100_000      # 经验回放缓冲区大小
GAMMA = 0.99               # 折扣因子
ALPHA = 1e-3               # 学习率
TAU = 0.01                 # 目标网络软更新系数
NUM_STEPS_FOR_UPDATE = 4   # 每隔多少步更新网络
EPSILON_DECAY = 0.995      # ε 衰减系数
EPSILON_MIN = 0.01         # 最小探索率

# ---------------------------
# Gym 环境创建与 API 兼容处理
# ---------------------------
env = gym.make('LunarLander-v2')

def reset_env(env):
    """
    重置环境，兼容新旧 API：
    - 如果返回值为 tuple，则认为是新版 Gymnasium API 格式 (observation, info)；
    - 否则直接返回 observation。
    """
    result = env.reset()
    if isinstance(result, tuple):
        observation, _ = result
        return observation
    return result

def step_env(env, action):
    """
    执行一步环境转移，兼容新旧 API：
    - 新 Gymnasium API 返回 (observation, reward, terminated, truncated, info)
      此时 done = terminated or truncated；
    - 老 API 返回 (observation, reward, done, info)；
    返回统一格式：(next_state, reward, done, info)
    """
    result = env.step(action)
    if isinstance(result, tuple):
        if len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
            return next_state, reward, done, info
        elif len(result) == 4:
            next_state, reward, done, info = result
            return next_state, reward, done, info
    raise ValueError("Unexpected return format from env.step()")

# 使用 reset_env 获取初始状态并确定状态维度、动作数
state = reset_env(env)
state_size = env.observation_space.shape[0]
num_actions = env.action_space.n

# ---------------------------
# 经验数据结构定义
# ---------------------------
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

# ---------------------------
# 模型加载或创建
# ---------------------------
model_path = 'lunar_lander_model.h5'
if os.path.exists(model_path):
    print("加载已有模型...")
    q_network = load_model(model_path)
else:
    q_network = Sequential([
        Input(shape=(state_size,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')
    ])

# 构造与 q_network 结构相同的目标 Q 网络，并复制初始权重
target_q_network = Sequential([
    Input(shape=(state_size,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_actions, activation='linear')
])
target_q_network.set_weights(q_network.get_weights())

# 优化器定义
optimizer = Adam(learning_rate=ALPHA)

# ---------------------------
# 定义 Q 网络训练相关函数
# ---------------------------
def compute_loss(experiences, gamma, q_network, target_q_network):
    states, actions, rewards, next_states, done_vals = experiences

    # 计算目标 Q 值：y = r + gamma * max_a' Q(next_state, a') * (1 - done)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    y_targets = rewards + gamma * max_qsa * (1 - done_vals)

    # 计算当前 Q 网络对应的 Q 值
    q_values = q_network(states)
    idx = tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1)
    q_values = tf.gather_nd(q_values, idx)

    # Mean Squared Error 计算损失
    loss = MSE(y_targets, q_values)
    return loss

def agent_learn(experiences, gamma):
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)
    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # 软更新目标网络权重
    for target_param, q_param in zip(target_q_network.trainable_variables, q_network.trainable_variables):
        target_param.assign(TAU * q_param + (1 - TAU) * target_param)

# ---------------------------
# 主训练循环
# ---------------------------
def main():
    start = time.time()

    num_episodes = 2000
    max_num_timesteps = 1000

    total_point_history = []
    num_p_av = 100  # 用于计算平均奖励的窗口大小
    epsilon = 0.1 if os.path.exists(model_path) else 1  # 若使用预训练模型，则降低探索率

    memory_buffer = deque(maxlen=MEMORY_SIZE)
    os.makedirs('videos', exist_ok=True)

    for i in range(num_episodes):
        state = reset_env(env)  # 使用兼容重构的 reset 方法
        total_points = 0

        for t in range(max_num_timesteps):
            state_qn = np.expand_dims(state, axis=0)
            q_values = q_network(state_qn)
            action = utils.get_action(q_values, epsilon)

            # 使用兼容重构的 step 方法
            next_state, reward, done, _ = step_env(env, action)
            memory_buffer.append(Experience(state, action, reward, next_state, done))

            if utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer):
                experiences = utils.get_experiences(memory_buffer)
                agent_learn(experiences, GAMMA)

            state = next_state  # 直接更新状态
            total_points += reward
            if done:
                break

        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        print(f"Episode {i + 1} | Avg Reward (last {num_p_av} episodes): {av_latest_points:.2f} | Epsilon: {epsilon:.3f}")

        # 每 50 轮保存视频
        if (i + 1) % 500 == 0:
            video_filename = os.path.join("videos", f"lunar_lander_ep{i + 1}.mp4")
            print(f"Creating video for episode {i + 1}...")
            utils.create_video(video_filename, env, q_network)
            print(f"Video saved to {video_filename}")

        # 如果平均奖励达到阈值，保存模型并退出训练
        if av_latest_points >= 270:
            print(f"Environment solved in {i + 1} episodes!")
            q_network.save(model_path)
            break

    tot_time = time.time() - start
    print(f"Total Training Time: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")
    utils.plot_history(total_point_history)

    final_video = os.path.join("videos", "lunar_lander_final.mp4")
    print("Creating final video...")
    utils.create_video(final_video, env, q_network)
    print(f"Final video saved to {final_video}")

if __name__ == '__main__':
    main()