class Task:
  def __init__(
    self,
    id: int,
    job_id: int,
    cpu: float,
    ram: float,
    status: int,
    runtime: float,
    ):
    
    self.id = id
    self.job_id = job_id
    self.cpu = cpu
    self.ram = ram
    self.server_farm_id = None
    self.server_id = None
    self.vm_id = None
    self.status = status # -1: rejected, 0: finished, 1: ready, 2: running, 3: initialized.
    self.runtime = runtime
    self.arrival_time = None
    self.departure_time = None
    #self.deadline = self.start_time + self.runtime