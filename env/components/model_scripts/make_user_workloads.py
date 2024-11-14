import igraph as ig
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import random
import time
import tqdm
import os

# Ensure all vertices are connected
def ensure_connected(g):
  num_vertices = g.vcount()

  # Identify vertices with no incoming edges
  vertices_with_no_incoming = [
    vertex for vertex in range(num_vertices) if g.indegree(vertex) == 0
  ]

  # Determine the number of starting nodes to reserve
  num_starting_nodes = min(len(vertices_with_no_incoming), 5)
  starting_nodes = set(random.sample(vertices_with_no_incoming, num_starting_nodes))

  # Connect remaining vertices with no incoming edges
  for vertex in vertices_with_no_incoming:
    if vertex not in starting_nodes:
      # Connect to a random vertex
      target_vertex = random.choice([
        v for v in range(num_vertices) if v != vertex and g.indegree(v) > 0
      ])
      g.add_edge(vertex, target_vertex)

  # Ensure no completely isolated vertices remain
  for vertex in range(num_vertices):
    if g.indegree(vertex) == 0 and g.outdegree(vertex) == 0:
      target_vertex = vertex - 1 if vertex > 0 else 1
      g.add_edge(target_vertex, vertex)

  return g

# Set vertices and edges attributes
def add_graph_vertices_and_edges_attributes(graph_data):
  g, task_resources = graph_data
  
  # Create a new graph with manually set vertex IDs
  new_indices = list(range(g.vcount()))
  new_graph = ig.Graph(directed=True)
  new_graph.add_vertices(len(new_indices))
  
  # Assign names to the vertices in the new graph
  for i, vertex in enumerate(new_graph.vs):
    vertex['name'] = str(i)  # Assigning sequential names to vertices
  
  for vertex, resources in zip(new_graph.vs, task_resources):
    cpu_value, ram_value = resources
    vertex['Required_CPU_and_MEM'] = {(cpu_value, ram_value)}
 
  if new_graph.vcount() > 1:
    # Collect edge tuples from the original graph with adjusted IDs
    edges_to_add = [
      (new_indices[edge.source], new_indices[edge.target]) for edge in g.es]
    # Add edges to the new graph in a batch
    new_graph.add_edges(edges_to_add)
  
  # Free up memory by deleting variables
  del g, task_resources
  
  return new_graph

def generate_random_dag_chunk(chunk):
  jobs = chunk.groupby('Job ID')
  graphs = []
  
  for job_id, group in jobs:
    num_tasks = len(group)
    task_resources = list(
      zip(group['Resource Request CPU'], group['Resource Request RAM']))
    graphs.append((num_tasks, task_resources))
  
  return graphs

def generate_graph(graph_data, seed=None):
  if seed is not None:
    random.seed(seed)
  
  num_tasks, task_resources = graph_data
  if (num_tasks > 5):
    g = ig.Graph.Erdos_Renyi(
      n=num_tasks,
      m=num_tasks*2,
      directed=False,
      loops=False)
    g.to_directed(mode="acyclic")
    
    g = ensure_connected(g)
  
  elif (num_tasks <= 5 and num_tasks > 2):
    g = ig.Graph.Erdos_Renyi(
      n=num_tasks,
      m=num_tasks,
      directed=False,
      loops=False)
    g.to_directed(mode="acyclic")
  
    g = ensure_connected(g)
  
  elif (num_tasks == 2):
    g = ig.Graph(directed=True)
    g.add_vertices(2)
    g.add_edges([(0, 1)])
  
  else:
    g = ig.Graph(directed=True)
    g.add_vertices(1)

  graph_data = g, task_resources
  g = add_graph_vertices_and_edges_attributes(graph_data)
  del task_resources
  return g

def create_dags(file_path, total_jobs, seed=None):
  if seed is not None:
    random.seed(seed)
  
  chunk_size = 500
  if total_jobs < chunk_size:
    chunk_size = total_jobs
  tasks_processed = 0
  jobs_processed = 0
  chunks_to_process = []
  # Read the CSV file in chunks and process only up to total_jobs
  for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    tasks_processed += chunk['Task Index'].count()
    chunks_to_process.append(chunk)
    jobs_processed += chunk['Job ID'].nunique() # only increment unique job ids
    if jobs_processed >= total_jobs:
      break
  
  start_time = time.time()  # Start time measurement
  dags = []
  for chunk in tqdm.tqdm(chunks_to_process, desc='Processing Chunks', total=len(chunks_to_process)):
    graphs = generate_random_dag_chunk(chunk)
    
    for graph_data in graphs:
      dag = generate_graph(graph_data, seed)
      dags.append(dag)
  end_time = time.time()  # End time measurement
  elapsed_time = end_time - start_time

  #print(f"Total number of jobs (DAGs): {len(dags)}")
  print(f"Total number of processed tasks: {tasks_processed}")
  #print(f"Total processing time: {elapsed_time} seconds\n")

  return dags

def print_single_vertex_attributes(vertex):
  print(f"Vertex/Task {vertex['name']}:")
  # print cpu and mem values in each vertex
  cpu_mem_attr = vertex.attributes().get('Required_CPU_and_MEM', set())
  print(f"Required_CPU_and_MEM: ", end="")
  for cpu_value, mem_value in cpu_mem_attr:
    print(f"CPU: {cpu_value}, MEM: {mem_value}")

def print_single_dag_attributes(dag):
  print("Vertex attributes:")
  for vertex in dag.vs:
    print_single_vertex_attributes(vertex)
  print("\n", end="")

def visualize_a_dag(idx, dag):
  #print(dag.is_dag())
  fig, ax = plt.subplots()
  plt.title(f'DAG {idx}')
  ig.plot(
    dag,
    target=ax,
    layout="sugiyama", # alternative: kk
    #vertex_size=15,
    vertex_color="lightblue",
    vertex_label=[f"{vertex['name']}" for vertex in dag.vs],
    edge_color="#222",
    edge_width=1,
  )
  plt.show()

def visualize_dags(dags):
  for idx, dag in enumerate(dags):
    # Print vertices and edges attributes while visualizing DAG
    print(f"DAG {idx}: ")
    print_single_dag_attributes(dag)
    visualize_a_dag(idx, dag)

if __name__ == '__main__': # this script is runnable to visualize the dags
  current_dir = os.path.dirname(__file__)
  file_path = os.path.join(current_dir, "..", "..", "helper", "jobs_dataset", "google_cluster_trace.csv")
  
  seed = 42
  total_jobs = 2 # value set by user / representing user requests
  dags = create_dags(file_path, total_jobs)
  visualize_dags(dags)