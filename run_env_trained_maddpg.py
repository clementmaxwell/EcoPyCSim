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

capacity = int(1e6)
batch_size = 1024 #1024
actor_lr = 0.0005
critic_lr = 0.0005

env, dim_info = set_env(num_jobs=num_jobs, num_server_farms=num_server_farms, num_servers=210)

project_dir = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(project_dir, "results")
env_dir = os.path.join(save_folder, 'maddpg')
model_file_path = os.path.join(env_dir, "model.pt")
maddpg = MADDPG.load(dim_info, model_file_path, capacity, batch_size, actor_lr, critic_lr)

# Testing MADDPG
print("MADDPG testing: ")
obs, info = env.reset(seed=42069)

prices = []
wall_times = []

step = 0
while env.agents:
    action = maddpg.select_action(obs)
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env.agents}
    
    wall_times.append(info["server_farm"]["wall_time"])
    prices.append(info["server_farm"]["price"])
    
    step += 1
    obs = next_obs
    
    if all(done.values()):
      print("Number of rejected jobs: ", len(info['server_farm']['rejected_job_ids']))
      env.close()
      break

sum_price = np.sum(prices)
average_price = np.mean(prices)
end_wall_time = wall_times[-1] if wall_times else None

print("Steps taken as MADDPG agents: ", step)
print("Total tasks rejected by MADDPG agents: ", env.rejected_tasks_count)
print(f"Average Energy Price (MADDPG): {average_price:.2f}")
print(f"Total Energy Price (MADDPG): {sum_price:.2f}")
print(f"Wall-time end (MADDPG): {end_wall_time:.2f}")

# Plotting metrics for MADDPG
steps = [s for s in range(step)]
plt.figure(figsize=(10, 5))
  
# Energy cost vs price at each step
plt.subplot(2, 1, 1)
plt.plot(steps, prices, label='Energy Cost')
plt.xlabel('Step (Env Step)')
plt.ylabel('Energy Cost')
plt.title('Energy Cost vs Price at each step (MADDPG)')
plt.legend()

# Energy cost vs time
plt.subplot(2, 1, 2)
plt.plot(wall_times, prices, label='Energy Cost')
plt.xlabel('Time (Wall Time)')
plt.ylabel('Energy Cost')
plt.title('Energy Cost vs Time (MADDPG)')
plt.legend()

plt.tight_layout()
plt.show()

# Testing Random actions
print("Random testing: ")
obs, info = env.reset(seed=42069)

prices = []
wall_times = []

step = 0
while env.agents:
    action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env.agents}
    
    wall_times.append(info["server_farm"]["wall_time"])
    prices.append(info["server_farm"]["price"])
    
    step += 1
    obs = next_obs
    
    if all(done.values()):
      print("Number of rejected jobs: ", len(info['server_farm']['rejected_job_ids']))
      env.close()
      break

sum_price = np.sum(prices)
average_price = np.mean(prices)
end_wall_time = wall_times[-1] if wall_times else None

print("Steps taken as Random actions: ", step)
print("Total tasks rejected by Random actions: ", env.rejected_tasks_count)
print(f"Average Energy Price (Random): {average_price:.2f}")
print(f"Total Energy Price (Random): {sum_price:.2f}")
print(f"Wall-time end (Random): {end_wall_time:.2f}")

# Plotting metrics for Random actions
steps = [s for s in range(step)]
plt.figure(figsize=(10, 5))
  
# Energy cost vs price at each step
plt.subplot(2, 1, 1)
plt.plot(steps, prices, label='Energy Cost')
plt.xlabel('Step (Env Step)')
plt.ylabel('Energy Cost')
plt.title('Energy Cost vs Price at each step (Random)')
plt.legend()

# Energy cost vs time
plt.subplot(2, 1, 2)
plt.plot(wall_times, prices, label='Energy Cost')
plt.xlabel('Time (Wall Time)')
plt.ylabel('Energy Cost')
plt.title('Energy Cost vs Time (Random)')
plt.legend()

plt.tight_layout()
plt.show()
