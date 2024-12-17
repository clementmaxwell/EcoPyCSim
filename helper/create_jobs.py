import os
import numpy as np

from components.model_scripts.make_user_workloads import create_dags
from components.models import task
from components.models import job

def generate_arrival_times(total_dags, lambda_):
  arrival_times = np.random.exponential(scale=1/lambda_, size=total_dags)
  arrival_times = np.cumsum(arrival_times)
  return arrival_times

"""
Create user specified number of jobs (workflows)
"""
def initialize_user_requests_queue(total_jobs, seed=None):
  if seed != None:
    np.random.seed(seed)
  
  base_dir = os.path.dirname(__file__)
  
  jobs_file_path = os.path.join(base_dir, "jobs_dataset", "google_cluster_trace.csv")
  jobs_file_path = os.path.abspath(jobs_file_path)

  if not os.path.isfile(jobs_file_path):
    raise FileNotFoundError(f"File not found: {jobs_file_path}")
  
  dags = create_dags(jobs_file_path, total_jobs, seed)
  
  # parameters: lambda is job arrival rate,
  # mu and sigma is for normal distribution on modelling task service time.
  # to visualize the normal distribution graph,
  # user can go to learning_codes -> statistics -> run normal_distribution_test.py
  total_dags = len(dags)
  lambda_ = 0.5 # can be set by user.
  mu = 5
  sigma = np.sqrt(1.6)
  
  # generate jobs arrival times using poisson process
  jobs_arrival_times = generate_arrival_times(total_dags, lambda_)
  
  user_requests_queue = []

  for idx, dag in enumerate(dags):
    tasks_list = []
    for vertex in dag.vs:
      task_name = vertex['name']
      cpu_mem_attr = vertex.attributes().get('Required_CPU_and_MEM', set())

      id = task_name
      cpu = next(iter(cpu_mem_attr))[0]
      ram = next(iter(cpu_mem_attr))[1]
      status = 3
      # set execution time of task using normal distribution
      vm_runtime = round(np.random.default_rng().normal(mu, sigma), 2)
      #print(f"Task {id}, run time: {vm_runtime}")
      a_task = task.Task(id, idx, cpu, ram, status, vm_runtime)

      tasks_list.append(a_task)

    num_tasks = dag.vcount()
    # set job arrival time
    #print(f"job: {idx}, arrival time: {round(jobs_arrival_times[idx], 2)}\n")
    job_arrival_time = round(jobs_arrival_times[idx], 2)
    a_job = job.Job(idx, dag, tasks_list, num_tasks, job_arrival_time)
    user_requests_queue.append((job_arrival_time, a_job))

  return user_requests_queue