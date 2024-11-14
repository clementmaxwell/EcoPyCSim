# code is from: https://github.com/wild-firefox/maddpg-pettingzoo-pytorch/tree/master
import logging
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from schedulers.marl.maddpg.Agent import Agent
from schedulers.marl.maddpg.Buffer import Buffer


def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""
    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir, device=None):
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        # sum all the dims of each agent to get input dim for critic
        global_obs_dim = sum(
            sum(np.prod(obs_shape) for obs_shape in val['obs_shape'].values())
            for val in dim_info.values()
        )
        global_act_dim = sum(val['action_dim'] for val in dim_info.values())
        
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        for agent_id, info in dim_info.items():
            obs_dim = sum(np.prod(obs_shape) for obs_shape in info['obs_shape'].values())
            act_dim = info['action_dim']
            #print(f"Initializing agent {agent_id}: obs_dim={obs_dim}, act_dim={act_dim}")
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_dim + global_act_dim, actor_lr, critic_lr, self.device)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, self.device)
        self.dim_info = dim_info

        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))

    def flatten_obs(self, obs_dict):
        #print("obs dict: ")
        #print(obs_dict)
        flattened_obs = []
        for key in sorted(obs_dict.keys()):
            obs_array = obs_dict[key]
            if isinstance(obs_array, np.ndarray):
                flattened_obs.extend(obs_array.flatten())
            else:
                flattened_obs.append(obs_array)  # Append scalar directly
        #print(f"Flattened obs: {flattened_obs}")
        return np.array(flattened_obs)
    
    def add(self, obs, action, reward, next_obs, done):
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            flat_o = self.flatten_obs(obs[agent_id])
            flat_next_o = self.flatten_obs(next_obs[agent_id])

            # Ensure the flattened observation matches the expected shape in the buffer
            #print(f"Agent: {agent_id}, Flat obs shape: {flat_o.shape}, Buffer obs shape: {self.buffers[agent_id].obs.shape[1]}")
            assert flat_o.shape[0] == self.buffers[agent_id].obs.shape[1], \
                f"Shape mismatch: {flat_o.shape[0]} != {self.buffers[agent_id].obs.shape[1]}"

            # Add to the buffer
            self.buffers[agent_id].add(flat_o, action[agent_id], reward[agent_id], flat_next_o, done[agent_id])

    def sample(self, batch_size):
      """Sample experience from all the agents' buffers and collect data for network input"""
      # Get the total number of transitions; these buffers should have the same number of transitions
      total_num = len(next(iter(self.buffers.values())))
      if total_num < batch_size:
          batch_size = total_num

      indices = np.random.choice(total_num, size=batch_size, replace=False)

      # NOTE that in MADDPG, we need the obs and actions of all agents
      # but only the reward and done of the current agent is needed in the calculation
      obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}
      for agent_id in self.buffers.keys():
          o, a, r, n_o, d = self.buffers[agent_id].sample(indices)
          obs[agent_id] = o
          act[agent_id] = a
          reward[agent_id] = r
          next_obs[agent_id] = n_o
          done[agent_id] = d
          # calculate next_action using target_network and next_state
          next_act[agent_id] = self.agents[agent_id].target_action(n_o)

      return obs, act, reward, next_obs, done, next_act

    def select_action(self, obs):
        actions = {}
        for agent, o in obs.items():
            flat_o = self.flatten_obs(o)  # Flatten the observation
            flat_o = torch.from_numpy(flat_o).unsqueeze(0).float().to(self.device)
            a = self.agents[agent].action(flat_o)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            actions[agent] = a.squeeze(0).argmax().item()
            self.logger.info(f'{agent} action: {actions[agent]}')
        return actions

    def learn(self, batch_size, gamma):
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act = self.sample(batch_size)
            # update critic
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()),
                                                                 list(next_act.values()))
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs[agent_id], model_out=True)
            act[agent_id] = action
            actor_loss = -agent.critic_value(list(obs.values()), list(act.values())).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            # self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def save(self, reward):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            os.path.join(self.res_dir, 'model.pt')
        )
        #with open(os.path.join(self.res_dir, 'rewards.txt'), 'w') as f:  # save training data
        #    pickle.dump({'rewards': reward}, f)

    @classmethod
    def load(cls, dim_info, file, capacity, batch_size, actor_lr, critic_lr, device=None):
        """init maddpg using the model saved in `file`"""
        # Initialize the instance with proper parameters
        instance = cls(dim_info, capacity, batch_size, actor_lr, critic_lr, os.path.dirname(file), device=device)
        
        # Load the saved model state
        data = torch.load(file)
        # data = torch.load(file, map_location=device)  # 确保加载到适当的设备上
        # Load the actor parameters
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        
        return instance
