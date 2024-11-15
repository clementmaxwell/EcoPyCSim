# EcoPyCSim

EcoPyCSim (Economical Python Cloud Scheduling Simulator) is a novel multi-agent deep reinforcement learning (MADRL)-based cloud scheduling simulator designed for energy-aware job scheduling and resource allocation, implemented as a [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) environment.

![EcoPyCSim Framework](https://github.com/user-attachments/assets/dd577dde-f15e-4212-a06e-4ca3765886ef)

## Overview

EcoPyCSim leverages a Partially Observable Stochastic Game (POSG) model to simulate a realistic cloud scheduling environment. The simulator evaluates the MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm for efficient scheduling decisions within this dynamic environment.

Key features include:

- **Energy-Aware Scheduling**: Optimizes job scheduling to reduce energy consumption in cloud environments.
- **Resource Allocation**: Adapts resource distribution dynamically based on workload and environment changes.
- **Multi-Agent Framework**: Supports the evaluation of multi-agent reinforcement learning algorithms, like MADDPG, in a complex scheduling scenario.

## Installation

### Prerequisites

Ensure that you have Python 3.9+ installed. It's recommended to run this project in a virtual environment to avoid conflicts with other packages. Recommended version: Python 3.10.11.
Note: Avoid using Python 3.12.0 or newer, as there is a known bug in the dependencies with this project.

### Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/clementmaxwell/EcoPyCSim.git
   cd EcoPyCSim

2. **Create a Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. **Install Dependencies**:

   ```bash
   After activating your virtual environment, install the required packages:
   pip install -r requirements.txt

## Usage

Once the setup is complete, you can begin running simulations within the EcoPyCSim environment. 
Simply execute the script with this command:
```bash
python run_env.py
```

To train the maddpg agent within the environment, run:
```bash
python run_env_train_maddpg.py
```

To evaluate the performance of the trained model, run:
```bash
python run_env_trained_maddpg.py
```

### Adjustable Parameters
The following parameters can be adjusted for each script:
- num_jobs: The number of jobs to simulate
- num_server_farms: The number of server farms in the simulation
- num_servers: The number of servers per server farm

Additionally, you can adjust the episode length and the MADDPG-specific parameters for training and evaluation.

### Job Arrival Rate Configuration
To set the arrival rate of jobs, modify the following parameters in helper/create_jobs.py:
- lambda_: The rate at which jobs arrive
- mu: The rate at which jobs are processed
- sigma: The rate variability in job processing time

## Contributing

Contributions are welcome! Please fork the repository and create a pull request to submit your changes.

## Citation

If you use EcoPyCSim or refer to our work in your research, please cite the following paper:

### BibTeX

```bibtex
@article{hou2024multiagent,
   title={Ecopycsim: A Multi-Agent Reinforcement Learning-based Cloud Scheduling Simulator for Energy-Aware Job Scheduling and Resource Allocation},
   author={Huanhuan Hou and Clement Maxwell and Azlan Ismail},
   year={2024},
   journal={SSRN Electronic Journal},
   note={Preprint},
   url={http://ssrn.com/abstract=5020937}
}
```

## License

This project is licensed under the MIT License.
