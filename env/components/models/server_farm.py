import igraph as ig

class Server_Farm:
  def __init__(
    self,
    id: int,
    graph: ig.Graph(), # type: ignore
    servers: list,
    num_servers: int
  ):
    
    self.id = id
    
    self.graph = graph
    
    self.servers = self.populate_servers(servers)
    
    self.num_servers = num_servers
    
    self.all_cpus = sum([server.c_cpu for server in self.servers.values()])
    
    self.all_rams = sum([server.c_ram for server in self.servers.values()])
    
  @property
  def curr_cpus_util(self):
    return [server.cpu_utilization_rate for server in self.servers.values()]
  
  @property
  def curr_pwrs(self):
    return [server.total_power for server in self.servers.values()]
  
  @property
  def get_price(self):
    total_power = round(sum(server.total_power for server in self.servers.values()), 2)
    # Threshold from "Impact of dynamic energy pricing schemes
    # on a novel multi-user home energy management system".
    threshold = 1.5
    # Price from "Optimal residential load control with price
    # prediction in real-time electricity pricing environments".
    real_time_pricing_low = 5.91
    real_time_pricing_high = 8.27
    if total_power <= threshold:
      price = total_power * real_time_pricing_low
    else:
      price = total_power * real_time_pricing_high
    return round(price, 2)
  
  def populate_servers(self, servers):
    return {server.id: server for server in servers}