import numpy as np
import matplotlib.pyplot as plt

def generate_arrival_times(total_jobs, lambda_):
  # Generate arrival times for each job
  arrival_times = np.random.exponential(scale=1/lambda_, size=total_jobs)
  arrival_times = np.cumsum(arrival_times)  # Convert inter-arrival times to arrival times
  return arrival_times

# Parameters
total_jobs = 100  # Total number of jobs
lambda_ = 0.5 # Average rate of job arrival (lambda) or arrival rate
# Ideally, the user sets the job arrival rate (lambda)

# Generate arrival times for the total number of jobs
arrival_times = generate_arrival_times(total_jobs, lambda_)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(arrival_times, range(1, total_jobs + 1), marker='o', linestyle='-', color='b')
plt.xlabel('Time')
plt.ylabel('Job Number')
plt.title('Arrival Times of Jobs')
plt.grid(True)
plt.show()