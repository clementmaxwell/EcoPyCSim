import igraph as ig

class Job:
  def __init__(
    self,
    id: int,
    dag: ig.Graph(directed=True), # type: ignore
    tasks: list,
    num_tasks: int,
    time_arrived: float,
  ):
    # unique identifier of this job
    self.id = id
    
    # igraph dag storing the task dependencies
    self.dag = dag
    
    # tasks in the job
    self.tasks = self.populate_tasks(tasks)
    
    # number of tasks in this job
    self.num_tasks = num_tasks
    
    # time that this job arrived
    self.time_arrived = time_arrived
    
    # time that this job completed
    self.time_completed = None
    
    self.get_first_ready_task_flag = False
    
    self.deadline = self.find_critical_path_length()
  
  @property
  def completed(self):
    return all(task.status == 0 for task in self.tasks.values())
  
  @property
  def number_of_rejected_tasks(self):
    return sum(1 for task in self.tasks.values() if task.status == -1)
  
  def populate_tasks(self, tasks):
    return {task.id: task for task in tasks}
  
  def get_ready_tasks(self):
    if self.get_first_ready_task_flag == False:
      ready_tasks = self.get_first_ready_tasks()
      self.get_first_ready_task_flag = True
      return ready_tasks
    
    elif self.get_first_ready_task_flag == True:
      ready_tasks = set()
      
      for task_id, task in self.tasks.items():
        if task.status == 0:
          #print(f"task id:{str(task_id)}")
          child_tasks = self.get_children_of_task(task_id)
          #print(f"Job: get_ready_tasks: child_tasks: {child_tasks}")
          ready_tasks.update(child_tasks)
      
      return list(ready_tasks)
  
  def get_first_ready_tasks(self):
    """
    Get the first ready task (first parent node without dependencies)
    in the job
    """
    ready_task = []
    for task_id, task in self.tasks.items():
      if len(self.dag.neighbors(task_id, mode="in")) == 0:
        # Found the first ready tasks, set it to ready status
        task.status = 1
        ready_task.append(task)
    
    return ready_task
  
  def get_children_of_task(self, task_id):
    """
    Get the children tasks of a specific task within this job.
    """
    children = self.dag.neighbors(task_id, mode="out")
    
    return [
      self.tasks[str(child_id)]
      for child_id in children
      if str(child_id) in self.tasks and
      self.tasks[str(child_id)].status != 0 and
      not setattr(self.tasks[str(child_id)], 'status', 1)
    ]
  
  def get_parent_of_task(self, task_id):
    """
    Get the parent tasks of a specific task within this job.
    """
    parent = self.dag.neighbors(task_id, mode="in")
    #print(f"Job: get_parent_of_task: {parent}")
    return [self.tasks[str(parent_id)] for parent_id in parent if str(parent_id) in self.tasks]
  
  def reject_task_and_cascade(self, task_id):
    """
    Reject a task and cascade rejection to dependent tasks.
    """
    if task_id not in self.tasks or self.tasks[task_id].status == -1:
      return
    
    stack = [task_id]
    visited = set()
    
    while stack:
      current_id = stack.pop()
      if current_id not in visited:
        visited.add(current_id)
        self.tasks[str(current_id)].status = -1
        stack.extend(self.get_future_dependent_tasks(current_id))
  
  def get_future_dependent_tasks(self, task_id):
    """
    Get dependent future tasks based on task dependencies.
    """
    future_tasks = []
    for neighbor_id in self.dag.neighbors(task_id, mode="out"):
      if (self.tasks[str(neighbor_id)].status == 1 or
          self.tasks[str(neighbor_id)].status == 3):
        future_tasks.append(neighbor_id)
        future_tasks.extend(self.get_future_dependent_tasks(neighbor_id))
    return future_tasks
  
  def find_critical_path_length(self):
    """
    Find the execution time of longest sequential sequence of tasks to be scheduled.
    """
    topological_sort = self.dag.topological_sorting()
    
    # dp technique to find cpl
    max_weight = [0] * len(topological_sort)
    
    source_vertex_id = topological_sort[0]
    source_vertex_weight = self.tasks[str(source_vertex_id)].runtime
    max_weight[topological_sort[0]] = source_vertex_weight
    
    # fill the max_weight arr to find cpl
    # for example, let dag: 0->1->2->3 after topological sort.
    # each vertex has its own weight. let weight = [2, 3, 4, 5] 
    # when max_weight arr is being filled, the first ith is:
    # max_weight = [0, 0, 0, 0],
    # and its next and subsequent ith be:
    # topological sorted dag:   0 -- > 1 --> 2 --> 3
    # max_weight =             [2      5     6     11]
    # cpl = max(max_weight)
    
    for vertex in topological_sort:
      for neighbor in self.dag.neighbors(vertex, mode="out"):
        neighbor_vertex_weight = self.tasks[str(neighbor)].runtime
        weight = max_weight[vertex] + neighbor_vertex_weight
        if weight > max_weight[neighbor]:
          max_weight[neighbor] = weight
    
    critical_path_length = max(max_weight)
    return critical_path_length
  
  def modify_task_status(self, task_id, status):
    self.tasks[str(task_id)].status = status