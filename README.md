# Adaptive Preference Scaling for RLHF

This repository contains the code for our paper [**Adaptive Preference Scaling for Reinforcement Learning with Human Feedback**](https://arxiv.org/abs/2406.02764). We introduce an **adaptive preference loss** for reward modeling within the context of Reinforcement Learning with Human Feedback (RLHF). This approach enhances the flexibility of the reward learning process by incorporating an adaptive scaling parameter for each sample in the preference dataset.

## Table of Contents

1. [Robotic Control](#1-robotic-control)
2. [Natural Language Generation](#2-natural-language-generation)

## 1. Robotic Control

Our implementation of robotic control tasks leverages the following frameworks:

- **Stable-Baselines3**: A set of reliable implementations of reinforcement learning algorithms.
  - *Reference*: Raffin et al., 2021
  - [GitHub Repository](https://github.com/DLR-RM/stable-baselines3)

- **RL Zoo Training Framework**: A comprehensive training framework for reinforcement learning.
  - *Reference*: Raffin, 2020
  - [GitHub Repository](https://github.com/DLR-RM/rl-baselines3-zoo)
  
## Setup

Follow the steps below to set up the environment and install the necessary dependencies for running the robotic control code.

Create a Python virtual environment using Conda:

```
conda create -n robotic_control python=3.9
conda activate robotic_control
```

Install the package dependencies:

```
git clone https://github.com/huggingface/alignment-handbook.git](https://github.com/IlgeeHong/Adaptive-Preference-Scaling.git
cd ./Adaptive-Preference-Scaling/robotic_control/
pip install -r requirements.txt
```
