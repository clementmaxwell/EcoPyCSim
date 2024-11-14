import igraph as ig
import matplotlib.pyplot as plt
import random

# Set vertices and edges attributes
def add_graph_vertices_and_edges_attributes(g, seed=None):
  if seed is not None:
    random.seed(seed)
  else:
    random.seed(42)
  
  # Create a new graph with manually set vertex IDs
  new_indices = list(range(g.vcount()))
  new_graph = ig.Graph()
  new_graph.add_vertices(len(new_indices))
  
  # Assign names to the vertices in the new graph
  for i, vertex in enumerate(new_graph.vs):
    vertex['name'] = str(i)  # Assigning sequential names to vertices
  
  # predefined max number of VMs in the server
  v = 10 # max v type of VMs hosted on server
  alpha = 0
  beta = 10 # predefined to 10
  cumulative_server_cpu_value = 0
  cumulative_server_mem_value = 0
  
  # Set vertices attributes
  for vertex in new_graph.vs:
    # random alpha values to dictate different pwr consumption curve even with similar load on servers
    alpha = round(random.uniform(0.3, 0.8), 2)
    min_vm_cpu_and_ram_value = 1/v # min cpu and ram to host vm
    vm_cpu_value = min_vm_cpu_and_ram_value
    vm_mem_value = min_vm_cpu_and_ram_value
    cumulative_server_cpu_value = v
    cumulative_server_mem_value = v
    vertex['VM_type'] = [vm for vm in range(1, v+1)]
    vertex['VM_request_state'] = [0 for _ in range(v)]
    vertex['VM_CPU_and_MEM'] = {(vm_cpu_value, vm_mem_value)}
    vertex['Cumulative_Server_CPU_and_MEM'] = {
      (cumulative_server_cpu_value, cumulative_server_mem_value)}
    vertex['Power_Consumption_Coefficients'] = {(alpha, beta)}
  
  if new_graph.vcount() > 1:
    # Collect edge tuples from the original graph with adjusted IDs
    edges_to_add = [
      (new_indices[edge.source], new_indices[edge.target]) for edge in g.es]
    # Add edges to the new graph in a batch
    new_graph.add_edges(edges_to_add)
   
  # Set randomized weight to each edge in new graph
  #for edge in new_graph.es:
  #  edge['weight'] = round(random.uniform(0.0001, 1), 4)
  
  return new_graph

def get_number_of_active_VM_state_at_time_t(VM_request_state):
  active_VMs = VM_request_state.count(1)
  return active_VMs

def create_a_server_farm(num_servers, seed=None):
  if seed is not None:
    random.seed(seed)
  else:
    random.seed(42)
  
  if num_servers <= 6:
    g = ig.Graph.Full(
      n=num_servers, directed=False, loops=False)
  else:
    g = ig.Graph.Barabasi(
      n=num_servers, m=2, directed=False)
  g = add_graph_vertices_and_edges_attributes(g)
  return g

def create_server_farms(total_servers, num_farms, seed=None):
  if seed is not None:
    random.seed(seed)
  else:
    random.seed(42)
  
  farm_graphs = []
  
  servers_per_farm = total_servers // num_farms
  remaining_servers = total_servers % num_farms
  
  for i in range(num_farms):
    num_servers = servers_per_farm + (1 if i < remaining_servers else 0)
    farm_graphs.append(create_a_server_farm(num_servers, seed))
  
  return farm_graphs

def print_single_vertex_attributes(vertex):
  vm_request_state = vertex['VM_request_state']
  active_VMs = get_number_of_active_VM_state_at_time_t(vm_request_state)
  print(
    f"Vertex/Server {vertex['name']}\n",
    f"VM_type(s): {vertex['VM_type']}\n",
    f"VM_request_state(s): {vm_request_state}\n"
    f" Hosted VMs at current time: {active_VMs}"
    )
  vm_cpu_mem_attr = vertex.attributes().get('VM_CPU_and_MEM', set())
  print("VM(s)_CPU_and_MEM: ", end="")
  for vm_cpu_value, vm_mem_value in vm_cpu_mem_attr:
    print(f"CPU: {vm_cpu_value}, MEM: {vm_mem_value}")
  
  server_cpu_mem_attr = vertex.attributes().get(
    'Cumulative_Server_CPU_and_MEM', set())
  print("Cumulative_Server_CPU_and_MEM: ", end="")
  for server_cpu_value, server_mem_value in server_cpu_mem_attr:
    print(f"CPU: {server_cpu_value}, MEM: {server_mem_value}")
  
  pwr_consumption_coefficients = vertex.attributes().get(
    'Power_Consumption_Coefficients')
  print("Power_Consumption_Coefficients: ", end="")
  for alpha, beta in pwr_consumption_coefficients:
    print(f"Alpha: {alpha}, Beta: {beta}")
  print("\n", end="")

def print_single_edge_attributes(edge):
  attrs = edge.attributes()
  print(
    f"Edge {edge.tuple}: "
    f"weight/bandwidth={attrs['weight']}"
  )

def print_single_graph_attributes(graph):  
  print("Vertex attributes:")
  for vertex in graph.vs:
    print_single_vertex_attributes(vertex)

  #print("Edge attributes:")
  #for edge in graph.es:
  #  print_single_edge_attributes(edge)

  print("\n", end="")

def print_all_graph_attributes(farm_graphs):
  for idx, graph in enumerate(farm_graphs):
    print(f"Server farm {idx}:")
    print("Vertex attributes:")
    print_single_graph_attributes(graph)

def visualize_a_graph(idx, graph):
  # Display the created server farms
  fig, ax = plt.subplots()
  plt.title(f'server farm {idx}')
  ig.plot(
    graph,
    target=ax,
    layout=graph.layout_fruchterman_reingold(),
    #vertex_size=15,
    vertex_color="lightblue",
    vertex_label=[f"{vertex['name']}" for vertex in graph.vs],
    edge_color="#222",
    edge_width=1,
  )
  plt.show()

def visualize_graphs(farm_graphs):
  for idx, graph in enumerate(farm_graphs):
    print(f"Server farm {idx}:")
    visualize_a_graph(idx, graph)

def print_and_visualize_graphs(farm_graphs):
  for idx, graph in enumerate(farm_graphs):
    print(f"Server farm {idx}:")
    print_single_graph_attributes(graph)
    visualize_a_graph(idx, graph)

if __name__ == '__main__':
  total_servers = 9
  num_farms = 2
  
  farm_graphs = create_server_farms(total_servers, num_farms)
  print_and_visualize_graphs(farm_graphs)