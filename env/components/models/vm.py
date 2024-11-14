class Vm:
  def __init__(
    self,
    id: int,
    cpu: float,
    ram: float
  ):
    
    self.id = id
    
    self.cpu = cpu
    
    self.ram = ram
    
    self.status = 0 # 0: off, 1: occupied
    
    self.hosted_task = None
  
  def host_task(self, task):
    self.hosted_task = task
    self.status = 1  # Set VM status to 1: occupied
    self.cpu += task.cpu
    self.ram += task.ram

  def release_task(self):
    assert self.hosted_task is not None, "VM has not hosted task, expect a hosted task"
    if self.status == 1 and self.hosted_task:
      self.cpu -= self.hosted_task.cpu
      self.ram -= self.hosted_task.ram
      
      self.status = 0
      hosted_task_id = self.hosted_task.id
      vm_id = self.id
      self.hosted_task = None
      
    return hosted_task_id, vm_id