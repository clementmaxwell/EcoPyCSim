import numpy as np

from helper.create_jobs import initialize_user_requests_queue
from helper.create_server_farm import initialize_server_farms
from components.timeline import Timeline, TimelineEvent

from gymnasium import spaces
from pettingzoo import ParallelEnv

class CloudSchedulingEnv(ParallelEnv):
  def __init__(
    self,
    num_jobs,
    num_server_farms,
    num_servers,
    render_mode=None
  ):
    
    self.num_jobs = num_jobs
    self.num_server_farms = num_server_farms
    self.num_servers = num_servers
    assert self.num_servers / self.num_server_farms >= 1, "Server number must be possible to be divided among server farm number."
    self.server_farm_id = 0
    self.server_id = 0
    
    self.wall_time = 0
    self.timeline = Timeline()
    
    self.jobs = {}
    self.server_farms = {}
    
    self.handle_event = {
      TimelineEvent.Type.JOB_ARRIVAL: self._handle_job_arrival,
      TimelineEvent.Type.TASK_ARRIVAL: self._handle_task_arrival,
      TimelineEvent.Type.TASK_DEPARTURE: self._handle_task_departure,
    }
    
    self.agents = ["server_farm", "server"]
    
    self.possible_agents = self.agents[:]
    
    self.observation_spaces = {}
    for agent in self.agents:
      self.observation_spaces[agent] = None
    
    self.action_spaces = {}
    for agent in self.agents:
      self.action_spaces[agent] = None
    
    self.render_mode = render_mode
  
  def observation_space(self, agent):
    if agent == "server_farm":
      arr_list = np.array([[0.0 for _ in range(server_farm.num_servers)] for server_farm in self.server_farms.values()])
      shape = arr_list.shape
      obs = spaces.Dict({
        "cpus_utilization": spaces.Box(low=0, high=1.0, shape=shape, dtype=float),
        "task_cpu": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
        "task_ram": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
        "task_deadline": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float),
        "wall_time": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float)
      })
      return obs
    elif agent == "server":
      first_key = next(iter(self.server_farms.keys()))
      num_servers = self.server_farms[first_key].num_servers
      arr_list = np.array([0.0] * num_servers)
      shape = arr_list.shape
      obs = spaces.Dict({
        "cpus_utilization": spaces.Box(low=0, high=1, shape=shape, dtype=float),
        "task_cpu": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
        "task_ram": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
        "task_deadline": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float),
        "wall_time": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float)
      })
      return obs
  
  def action_space(self, agent):
    if agent == "server_farm":
      return spaces.Discrete(self.num_server_farms)
    elif agent == "server":
      return spaces.Discrete(
        self.server_farms[self.server_farm_id].num_servers)
  
  def render(self):
    pass
  
  def reset(self, seed=None, options=None):
    self.wall_time = 0
    self.time_limit = 0
    self.timeline.reset()
    
    self.jobs.clear()
    job_sequence = initialize_user_requests_queue(self.num_jobs, seed)
    for timeline, job in job_sequence:
      self.timeline.push(
        timeline, TimelineEvent(TimelineEvent.Type.JOB_ARRIVAL, data={"job": job})
      )
      self.jobs[job.id] = job

    server_farms = initialize_server_farms(self.num_servers, self.num_server_farms, seed)
    for server_farm in server_farms:
      self.server_farms[server_farm.id] = server_farm
    
    self.server_farm_id = 0
    self.server_id = 0
    
    self.observation_spaces = {
      agent: self.observation_space(agent)
      for agent in self.agents
    }
    
    self.action_spaces = {
      agent: self.action_space(agent)
      for agent in self.agents
    }
    
    self.prev_server_farm_reward = 0
    self.prev_server_reward = 0
    
    self.active_job_ids = []
    self.completed_job_ids = set()
    
    self.rejected_job_ids = set()
    self.rejected_tasks_count = 0
    self.task_rejected_status = False
    
    self.schedulable_tasks = False
    self.scheduled_tasks = set()
    
    self.scheduled_task_cpu = 0
    self.scheduled_task_ram = 0
    self.scheduled_task_deadline = 0
    
    self._load_initial_jobs()
    
    infos = {agent: self.info(agent) for agent in self.agents}
    obs = {agent: self._get_observation(agent) for agent in self.agents}
    return obs, infos
  
  def step(self, actions):
    if self.schedulable_tasks:
      self._take_action(actions)
      
      obs = {agent: self._get_observation(agent) for agent in self.agents}
      reward = {agent: self._get_reward(agent) for agent in self.agents}
      terminated = {agent: False for agent in self.agents}
      truncated = {agent: False for agent in self.agents}
      infos = {agent: self.info(agent) for agent in self.agents}
      
      self.scheduled_task_deadline = 0
      self.task_rejected_status = False
      return obs, reward, terminated, truncated, infos
    
    # step through timeline until next scheduling event
    self._resume_simulation()
    
    obs = {agent: self._get_observation(agent) for agent in self.agents}
    reward = {agent: 0 for agent in self.agents}
    terminated = {agent: False for agent in self.agents}
    terminate = self.all_jobs_complete
    if terminate == True:
      terminated = {agent: True for agent in self.agents}
    truncated = {agent: False for agent in self.agents}
    infos = {agent: self.info(agent) for agent in self.agents}
    
    return obs, reward, terminated, truncated, infos
  
  @property
  def all_jobs_complete(self):
    return self.num_completed_jobs == len(self.jobs.keys())
  
  @property
  def num_completed_jobs(self):
    return len(self.completed_job_ids)
  
  @property
  def num_active_jobs(self):
    return len(self.active_job_ids)
  
  @property
  def num_rejected_jobs(self):
    return len(self.rejected_job_ids)
  
  def info(self, agent):
    if agent == "server_farm":
      info = {
      "active_job_ids": self.active_job_ids,
      "completed_job_ids": self.completed_job_ids,
      "rejected_job_ids": self.rejected_job_ids,
      "rejected_tasks_count": self.rejected_tasks_count,
      "wall_time": self.wall_time,
      "price": round(sum(server_farm.get_price for server_farm in self.server_farms.values()), 2),
      }
      return info
    elif agent == "server":
      return {}
  
  def _load_initial_jobs(self):
    # job arrival (find the schedulable ready tasks)
    arrived_jobs = []
    while not self.timeline.empty:
      wall_time, event = self.timeline.peek()
      
      try:
        job = event.data["job"]
        arrived_jobs.append((wall_time, job))
        self.timeline.pop()
      except KeyError:
        raise Exception("initial timeline must only contain jobs")
    
    for wall_time, job in arrived_jobs:
      self._handle_job_arrival(job, wall_time)
    self.wall_time = arrived_jobs[0][0]

    arrived_jobs.clear()
    self.schedulable_tasks = True
  
  def _take_action(self, actions):
    # agents take action of choosing which server farm and server to place the task
    self.server_farm_id = actions["server_farm"]
    self.server_id = actions["server"]
    
    server_farm = self.server_farms[self.server_farm_id]
    server = server_farm.servers[str(self.server_id)]
    
    # pseudocode:
    # pop task arrival event from Timeline
    # schedule the task to the chosen Server Farm in chosen Server
    # perform bookkeeping on task status in job in active job
    # add the task departure event after it has been scheduled
    # accept/reject task happens here if chosen server in server farm is valid   
    self.wall_time, event = self.timeline.pop()
    
    try:
      task = event.data["task_arrival"]
    except KeyError:
      raise Exception("scheduling action timeline must only contain task arrival events")
    
    if task.job_id in self.rejected_job_ids:
      self._process_task_rejection(task)
      return
    
    self._handle_task_arrival(task)
    
    if server.is_available:
      scheduled = server.host_task_in_server(task)
      if scheduled:
        self.scheduled_task_cpu = task.cpu
        self.scheduled_task_ram = task.ram
        self.scheduled_task_deadline = self.wall_time + task.runtime
        self._process_task_scheduling(task)
        self._insert_task_departure_event(task)
        self.scheduled_tasks.add(task)
        self.schedulable_tasks = False
      else:
        self._process_task_rejection(task)
    else:
      # reject task, drop the subsequent tasks from the job from the system.
      self._process_task_rejection(task)
  
  def _resume_simulation(self):
    """resumes the simulation until either there are new scheduling
    decisions to be made, or it's done.
    """
    assert not self.timeline.empty, print(
      print("rejected job ids: ", self.rejected_job_ids),
      print("rejected tasks count: ", self.rejected_tasks_count),
      print("current active job ids: ", self.active_job_ids),
      print("status of tasks in the current active job ids: "),
      print([task.status for active_job_id in self.active_job_ids for task in self.jobs[active_job_id].tasks.values()]),
      print("current jobs in the simulation: ", [key for key in self.jobs.keys()]),
      print("completed job ids:", self.completed_job_ids),
      self.timeline.print_queue()
    )
    
    while not self.timeline.empty:
      # handling only two types of event here:
      # task arrival (go to scheduling mode)
      # task departure (remove task from the cloud system)
      self.wall_time, event = self.timeline.peek()
      #print("wall time: ", self.wall_time, "event: ", event)
      try:
        task = event.data["task_arrival"]
        self.schedulable_tasks = True
      except KeyError:
        self.handle_event[event.type](**event.data)
        self.timeline.pop()
      
      schedulable_tasks = self.schedulable_tasks
      if schedulable_tasks:
        break

  def _get_observation(self, agent):
    if agent == "server_farm":
      cpus_util = np.array([self.server_farms[key].curr_cpus_util for key in self.server_farms.keys()])
      task_cpu = np.array(self.scheduled_task_cpu).flatten()
      task_ram = np.array(self.scheduled_task_ram).flatten()
      task_deadline = np.array(self.scheduled_task_deadline).flatten()

      obs = {
        "cpus_utilization": cpus_util,
        "task_cpu": task_cpu,
        "task_ram": task_ram,
        "task_deadline": task_deadline,
        "wall_time": np.array([self.wall_time])
        }
      return obs
    elif agent == "server":
      server_farm = self.server_farms[self.server_farm_id]
      cpus_util = np.array(server_farm.curr_cpus_util)
      task_cpu = np.array(self.scheduled_task_cpu).flatten()
      task_ram = np.array(self.scheduled_task_ram).flatten()
      task_deadline = np.array(self.scheduled_task_deadline).flatten()

      obs = {
        "cpus_utilization": cpus_util,
        "task_cpu": task_cpu,
        "task_ram": task_ram,
        "task_deadline": task_deadline,
        "wall_time": np.array([self.wall_time])
        }
      return obs

  def _get_reward(self, agent):
    # Shared reward for overall system efficiency
    curr_energy_cost = sum(self.server_farms[key].get_price for key in self.server_farms.keys())
    energy_saving_reward = curr_energy_cost - self.prev_server_farm_reward

    # Encourage cooperation by rewarding task success shared across agents
    if not self.task_rejected_status:
      task_success_reward = 1  # Positive reward for successful task scheduling
    else:
      task_success_reward = -2  # Penalty for task rejection

    # Combine global and individual rewards
    if agent == "server_farm":
      # Server farm reward includes both individual and global components
      combined_reward = energy_saving_reward + (task_success_reward * 0.5)
    elif agent == "server":
      # Server reward focuses on cooperating to complete tasks
      combined_reward = task_success_reward + (energy_saving_reward * 0.5)

    self.prev_server_farm_reward = curr_energy_cost
    return round(combined_reward, 2)
  
  # event handlers
  def _handle_job_arrival(self, job, wall_time=None):
    # get the schedulable ready tasks from the arrived job,
    # and put the ready tasks as task arrival events in the timeline.
    self.active_job_ids += [job.id]
    ready_tasks = self._find_schedulable_tasks(job_ids=[job.id])
    for task in ready_tasks:
      self.timeline.push(
        wall_time, TimelineEvent(TimelineEvent.Type.TASK_ARRIVAL, data={"task_arrival": task})
      )
  
  def _handle_task_arrival(self, task_arrival):
    task = task_arrival
    task.arrival_time = self.wall_time
  
  def _handle_task_departure(self, task_departure):
    # perform bookkeeping by keeping track of its departure time, and mark task as finished,
    # perform task departure from cloud system,
    # push the next ready task as task arrival events if there is any from the current active job.
    # or, remove the job in the system if all of its task has been completed,
    task = task_departure
    self.scheduled_task_cpu = task.cpu
    self.scheduled_task_ram = task.ram
    
    try:
      # remove task from cloud system code here:
      server_farm = self.server_farms[task.server_farm_id]
      server = server_farm.servers[task.server_id]

      server.clear_completed_task_in_server(task)
      job = self.jobs[task.job_id]
    except Exception:
      if task.job_id in self.rejected_job_ids:
        return
      return

    task.departure_time = self.wall_time
    self._process_task_completion(task)
    
    if job.completed:
      self._process_job_completion(job)
    else:
      ready_tasks = self._find_schedulable_tasks()

      self._insert_ready_tasks_events(ready_tasks)
  
  def _find_schedulable_tasks(self, job_ids=None):
    if job_ids is None:
      job_ids = list(self.active_job_ids)
    
    schedulable_tasks = [
      task
      for job_id in iter(job_ids)
      for task in iter(self.jobs[job_id].get_ready_tasks())
      if task not in self.scheduled_tasks and self._is_task_ready(task)
    ]
    
    return schedulable_tasks
  
  def _is_task_ready(self, task):
    """a task is ready if:
    - its status is in ready state, not rejected, not in running state,
    - and all of its parent dependencies has been satisfied"""
    if task.status != 1 or task.status == -1 or task.status == 2 or task.status == 3:
      return False
    
    job = self.jobs[task.job_id]
    for task in job.get_parent_of_task(task.id):
      if task.status != 0 and task.status != -1:
        return False
    return True
  
  def _insert_ready_tasks_events(self, ready_tasks):
    for task in ready_tasks:
      self.timeline.push(
        self.wall_time, TimelineEvent(TimelineEvent.Type.TASK_ARRIVAL, data={"task_arrival": task})
      )
      self.scheduled_tasks.add(task)
  
  def _insert_task_departure_event(self, task):
    departure_time = round(self.wall_time + task.runtime, 2)
    self.timeline.push(
      departure_time, TimelineEvent(TimelineEvent.Type.TASK_DEPARTURE, data={"task_departure": task})
    )
   
  def _process_task_scheduling(self, task):
    # mark task as running in job
    job = self.jobs[task.job_id]
    job.modify_task_status(task.id, 2)
  
  def _process_task_completion(self, task):
    # mark task as finished in job
    job = self.jobs[task.job_id]
    job.modify_task_status(task.id, 0)
  
  def _process_job_completion(self, job):
    assert job.id in self.jobs
    self.active_job_ids.remove(job.id)
    self.completed_job_ids.add(job.id)

    job.time_completed = self.wall_time
  
  def _process_task_rejection(self, task):
    self.task_rejected_status = True
    self.schedulable_tasks = False
    try:
      job = self.jobs[task.job_id]
      job.reject_task_and_cascade(task.id)
      self.rejected_tasks_count += job.number_of_rejected_tasks
      self.active_job_ids.remove(job.id)
      self.rejected_job_ids.add(task.job_id)
      self.jobs.pop(job.id)
    except Exception:
      return