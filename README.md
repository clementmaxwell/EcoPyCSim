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
Ensure that you have Python 3.8+ installed. It's recommended to run this project in a virtual environment to avoid conflicts with other packages.

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/clementmaxwell/EcoPyCSim.git
   cd EcoPyCSim

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

4. **Install Dependencies**:
   ```bash
   After activating your virtual environment, install the required packages:
   pip install -r requirements.txt

## Usage
Once the setup is complete, you can begin running simulations within the EcoPyCSim environment.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request to submit your changes.

## License
This project is licensed under the MIT License.

## Citation
If you use EcoPyCSim or refer to our work in your research, please cite the following paper:
(Paper is in pending submission)

### BibTeX
```bibtex
@misc{hou2024multiagent,
   title={A Multi-Agent Reinforcement Learning-based Cloud Scheduling Simulator for Energy-Aware Job Scheduling and Resource Allocation},
   author={Huanhuan Hou and Clement Maxwell and Azlan Ismail},
   year={2024},
   note={Preprint available at: \url{}},
}
