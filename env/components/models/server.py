import math

class Server:
  def __init__(
    self,
    id: int,
    server_farm_id: int,
    vms: list,
    c_cpu: float,
    c_ram: float,
    alpha: float,
    beta: float
  ):
    
    self.id = id
    
    self.server_farm_id = server_farm_id
    
    self.vms = self.populate_vm(vms)
    
    self.vm_numbers = len(self.vms)
    
    self.c_cpu = c_cpu
    
    self.c_ram = c_ram
    
    self.alpha = alpha
    
    self.beta = beta
    
    self.current_cpu_usage = 0.0
    
    self.current_ram_usage = 0.0
    
  # Energy Consumption Model on Server
  # Inspired from:
  # "DRL-Cloud: Deep Reinforcement Learning-Based Resource Provisioning
  # and Task Scheduling for Cloud Service Providers".
  @property
  def cpu_utilization_rate(self):
    total_vms_cpu = sum(vm.cpu for vm in self.vms.values() if vm.status == 1)
    cpu_utilization_rate = total_vms_cpu / self.c_cpu
    return round(cpu_utilization_rate, 2)

  @property
  def static_power(self):
    return 0.035 if self.cpu_utilization_rate > 0 else 0

  @property
  def dynamic_power(self):
    cpu_utilization_rate = self.cpu_utilization_rate
    optimal_utilization_rate = 0.7
    if cpu_utilization_rate < optimal_utilization_rate:
      return round(cpu_utilization_rate * self.alpha, 2)
    return round(
      (optimal_utilization_rate * self.alpha) +
      (math.pow(cpu_utilization_rate - optimal_utilization_rate, 2)
      * self.beta), 2)

  @property
  def total_power(self):
    return round((self.static_power + self.dynamic_power), 2)
  
  @property
  def is_available(self):
    # Check if any VM in the server is available
    return any(vm.status == 0 for vm in self.vms.values())
  
  def populate_vm(self, vms):
    return {vm.id: vm for vm in vms}
  
  def host_task_in_server(self, task):
    available_vm_id = next((vm.id for vm in self.vms.values() if vm.status == 0), None)
    if available_vm_id: # take the available vm id
      verdict = self.check_cpu_mem_constraint(task, available_vm_id) # check task CPU and MEM constraint
      if verdict: # if task can be hosted without violating VM and MEM limits, host the task on VM
        self.vms[available_vm_id].host_task(task)
        task.vm_id = available_vm_id
        task.server_id = self.id
        task.server_farm_id = self.server_farm_id
        
        self.current_cpu_usage += self.vms[available_vm_id].cpu
        self.current_ram_usage += self.vms[available_vm_id].ram
        return True, task # True for successful task hosting
      return False # False for rejected task
  
  # check if the VM has sufficient CPU or RAM to host the task without overburdening VM resource constraint
  def check_cpu_mem_constraint(self, task, vm_id):
    vm = self.vms[vm_id]
    if (task.cpu + vm.cpu > 1) or (task.ram + vm.ram > 1):
      return False
    return True
  
  def clear_completed_task_in_server(self, task):
    vm = self.vms[task.vm_id]
    task_id, vm_id = vm.release_task()
    
    self.current_cpu_usage -= self.vms[vm_id].cpu
    self.current_ram_usage -= self.vms[vm_id].ram
    
    return task_id, vm_id
  
  def check_active_vms(self):
    for vm in self.vms.values():
      if vm.status == 1:
        print(f"VM {vm.status} is active on server {self.id} hosting task {vm.hosted_task.id}")