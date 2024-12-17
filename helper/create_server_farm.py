from components.models import server, server_farm
from components.model_scripts.make_server_farms import create_server_farms, print_single_graph_attributes, print_all_graph_attributes
from components.models import vm

def initialize_server_farms(total_servers, num_farms, seed=None):
  if seed is not None:
    seed = seed
  farm_graphs = create_server_farms(total_servers, num_farms, seed)
  #print_all_graph_attributes(farm_graphs)
  server_farms = []

  for idx, graph in enumerate(farm_graphs):
    #print_single_graph_attributes(graph)
    server_list = []
    for vertex in graph.vs:
      server_name = vertex['name']
      vms_list_in_a_server = vertex['VM_type']
      vm_cpu_and_mem_attr = vertex.attributes().get('VM_CPU_and_MEM', set())
      server_cpu_mem_attr = vertex.attributes().get(
        'Cumulative_Server_CPU_and_MEM', set())
      pwr_consumption_coefficients = vertex.attributes().get(
        'Power_Consumption_Coefficients', set())
      vm_list = []
      for vm_idx in vms_list_in_a_server:
        vm_cpu = next(iter(vm_cpu_and_mem_attr))[0]
        vm_ram = next(iter(vm_cpu_and_mem_attr))[1]
        a_vm = vm.Vm(vm_idx, vm_cpu, vm_ram)
        vm_list.append(a_vm)

      server_id = server_name
      server_cpu = next(iter(server_cpu_mem_attr))[0]
      server_ram = next(iter(server_cpu_mem_attr))[1]
      server_alpha = next(iter(pwr_consumption_coefficients))[0]
      server_beta = next(iter(pwr_consumption_coefficients))[1]

      a_server = server.Server(
        server_id, idx, vm_list, server_cpu, server_ram, server_alpha, server_beta)
      server_list.append(a_server)

    num_servers = graph.vcount()
    #print(num_servers)
    a_server_farm = server_farm.Server_Farm(idx, graph, server_list, num_servers)
    server_farms.append(a_server_farm)

  return server_farms

#if __name__ == "__main__":
#  total_servers = 3
#  num_farms = 2
#  server_farms = initialize_server_farms(total_servers, num_farms)