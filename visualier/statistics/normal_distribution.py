import numpy as np
import matplotlib.pyplot as plt

mu = 6
sigma = np.sqrt(2)

for i in range(5):
  random_int = round(np.random.default_rng().normal(mu, sigma), 2)
  print(f"Example of a random int from a random normal distribution: {random_int}")

random_numbers = np.random.default_rng().normal(mu, sigma, 5000)
plt.hist(random_numbers, bins=30, density=True, alpha=0.6, color='g')
plt.title('Histogram of Random Numbers from Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()