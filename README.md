# Adaptive Preference Scaling for RLHF

This repository contains the code for our paper [Adaptive Preference Scaling for Reinforcement Learning with Human Feedback](https://arxiv.org/abs/2406.02764). We introduce an **adaptive preference loss** for reward modeling within the context of Reinforcement Learning with Human Feedback (RLHF). This approach improves the flexibility of the reward learning process by incorporating an adaptive scaling parameter for each sample in the preference dataset.

## Table of Contents

1. [Robotic Control](#1-robotic-control)
   - [1.1 Setup](#11-setup)
   - [1.2 Usage](#12-usage)
2. [Natural Language Generation](#2-natural-language-generation)
   - [2.1 Setup](#21-setup)
   - [2.2 Usage](#22-usage)
3. [Citation](#3-citation)

## 1. Robotic Control

Our implementations of robotic control tasks are based on the following frameworks:

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)

- [RL Zoo Training Framework](https://github.com/DLR-RM/rl-baselines3-zoo)
  
### 1-1. Setup

Follow the steps below to set up the environment and install the necessary dependencies for running the robotic control code.

- Create a Python virtual environment using Conda:

```
conda create -n robotic_control python=3.9
conda activate robotic_control
```

- Install the package dependencies:

```
git clone https://github.com/IlgeeHong/Adaptive-Preference-Scaling.git
cd ./Adaptive-Preference-Scaling/robotic_control/
pip install -r requirements.txt
```

### 1-2. Usage

Provide instructions on how to run the robotic control tasks. For example:

- Training an agent

```
bash run_rlhf.sh \
  --env_id AntBulletEnv-v0 \
  --enable_individualized_tau True \
  --n_rm_epochs 5 \
  --rm_lr 1e-4 \
  --rm_batch_size 64 \
  --n_tau_iters 3 \
  --tau_min 0.1 \
  --tau_max 1.0 \
  --rho 0.1
```

- Evaluating an agent

## 2. Natural Language Generation

Our implementations of natural language generation tasks are based on the following frameworks:

- [Transformers](https://github.com/huggingface/transformers)

- [TRL Training Framework](https://github.com/huggingface/trl)


### 2-1. Setup

Follow the steps below to set up the environment and install the necessary dependencies for running the natural language generation code.

- Create a Python virtual environment using Conda:

```

```

- Install the package dependencies:

```

```

### 2-2. Usage

```

```


## 3. Citation

```
@InProceedings{hong2024adaptive,
title={Adaptive Preference Scaling for Reinforcement Learning with Human Feedback},
author={Ilgee Hong and Zichong Li and Alexander Bukharin and Yixiao Li and Haoming Jiang and Tianbao Yang and Tuo Zhao},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
}
```
