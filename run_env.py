import pprint
import matplotlib.pyplot as plt

from env import cloud_scheduling_v0

def main():
  num_jobs = 2
  num_server_farms = 5
  num_servers = 5
  env = cloud_scheduling_v0.CloudSchedulingEnv(num_jobs, num_server_farms, num_servers)
  
  obs, infos = env.reset()
  
  while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    print('observations: ', end='')
    pprint.pprint(obs, sort_dicts=False)
    print('rewards: ', end='')
    pprint.pprint(rewards, sort_dicts=False)
    print('terminations: ', end='')
    pprint.pprint(terminations, sort_dicts=False)
    #pprint.pprint(truncations, sort_dicts=False)
    print('infos: ', end='')
    pprint.pprint(infos, sort_dicts=False)
    
    print("\n")
    if all(terminations.values()):
      print("terminated using env termination")
      env.close()
      break
  
  
  prices = list()
  wall_times = list()
  average_rewards_per_episode = {
    "server_farm_agent": [],
    "server_agent": []
  }
  rejected_tasks = 0
  steps = 0
  
  obs, infos = env.reset()
  
  while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    
    wall_times.append(infos["server_farm"]["wall_time"])
    prices.append(infos["server_farm"]["price"])
    
    steps += 1
    
    if all(terminations.values()):
      #print("terminated using env termination")
      env.close()
      break
  
  steps = [s for s in range(steps)]
  
  plt.figure(figsize=(10, 5))
    
  # energy cost vs price at each step
  plt.subplot(2, 1, 1)
  plt.plot(steps, prices, label='Energy Cost')
  plt.xlabel('Step (Env Step)')
  plt.ylabel('Energy Cost')
  plt.title('Energy Cost vs Price at each step')
  plt.legend()
  
  # energy cost vs time
  plt.subplot(2, 1, 2)
  plt.plot(wall_times, prices, label='Energy Cost')
  plt.xlabel('Time (Wall Time)')
  plt.ylabel('Energy Cost')
  plt.title('Energy Cost vs Time')
  plt.legend()
  
  plt.tight_layout()
  plt.show()
  
if __name__ == "__main__":
  main()