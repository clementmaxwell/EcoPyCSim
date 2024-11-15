import os
import matplotlib.pyplot as plt
import numpy as np
from schedulers.marl.maddpg.MADDPG import MADDPG

from env import cloud_scheduling_v0

def set_env(num_jobs, num_server_farms, num_servers):
  env = cloud_scheduling_v0.CloudSchedulingEnv(
    num_jobs, num_server_farms, num_servers)

  env.reset()

  _dim_info = {}
  for agent_id in env.agents:
    obs_space = env.observation_space(agent_id)
    #print(f"obs space {agent_id}: {obs_space}")
    obs_shape = {key: space.shape for key, space in obs_space.spaces.items()}
    action_dim = env.action_space(agent_id).n
    
    _dim_info[agent_id] = {
      'obs_shape': obs_shape,
      'action_dim': action_dim
    }
  
  return env, _dim_info

num_jobs = 300
num_server_farms = 30
num_servers = 210

episode_num = 10
random_steps = num_jobs * 0.1
learn_iterval = 5
capacity = int(1e6)
batch_size = 1024
actor_lr = 0.0005
critic_lr = 0.0005
gamma = 0.9
tau = 0.1

env_dir = os.path.join('./results', 'maddpg')
if not os.path.exists(env_dir):
  os.makedirs(env_dir)

reward_file_path = os.path.join(env_dir, 'reward.txt')
env, dim_info = set_env(num_jobs=num_jobs, num_server_farms=num_server_farms, num_servers=num_servers)

project_dir = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(project_dir, "results")
model_file_path = os.path.join(save_folder, "model.pt")

maddpg = MADDPG(
  dim_info,
  capacity=capacity,
  batch_size=batch_size,
  actor_lr=actor_lr,
  critic_lr=critic_lr,
  res_dir=env_dir)

agent_num = env.num_agents
episode_rewards = {agent_id: np.zeros(episode_num) for agent_id in env.agents}

for episode in range(episode_num):
  obs, info = env.reset()
  agent_reward = {agent_id: 0 for agent_id in env.agents}

  step = 0
  while env.agents:
    step += 1
    if episode == 0 and step < random_steps:
      action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
    else:
      action = maddpg.select_action(obs)
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env.agents}
    
    maddpg.add(obs, action, reward, next_obs, done)
    
    for agent_id, r in reward.items():
      agent_reward[agent_id] += r
    
    obs = next_obs
    
    if step >= random_steps and step % learn_iterval == 0:
      maddpg.learn(batch_size, gamma)
      maddpg.update_target(tau)
    
    # Check if all agents are done
    if all(done.values()):
      break

  with open(reward_file_path, "a") as file:
    for agent_id, r in agent_reward.items():
      episode_rewards[agent_id][episode] = r
    server_farm_reward = agent_reward['server_farm']
    server_reward = agent_reward['server']
    file.write(f"episode {episode + 1}, server_farm reward: {server_farm_reward}, server reward: {server_reward}\n")

  message = f'episode {episode + 1}, '
  sum_reward = 0
  for agent_id, r in agent_reward.items():
    message += f'{agent_id}: {r:.4f}; '
    sum_reward += r
  message += f'sum reward: {sum_reward:.4f}'
  print(f"server farm reward: {reward['server_farm']}, server reward: {reward['server']}")
  print(message)

  #if (episode+1) % 50 == 0:
  maddpg.save(episode_rewards)

def get_running_reward(arr: np.ndarray, window=100):
  """calculate the running reward, i.e., average of last 'window' elements from rewards"""
  running_reward = np.zeros_like(arr)
  for i in range(len(arr)):
    start_index = max(0, i - window + 1)
    running_reward[i] = np.mean(arr[start_index:i + 1])
  return running_reward

fig, ax = plt.subplots()
x = range(1, episode_num + 1)
for agent_id, reward in episode_rewards.items():
  ax.plot(x, reward, label=f'{agent_id} reward')
  ax.plot(x, get_running_reward(reward), linestyle='--', label=f'{agent_id} running reward')
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
title = 'MADDPG performance'
ax.set_title(title)
plt.savefig(os.path.join(env_dir, title))