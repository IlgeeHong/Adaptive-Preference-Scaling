# Adaptive Preference Scaling for RLHF

This repository contains the code for our paper [Adaptive Preference Scaling for Reinforcement Learning with Human Feedback](https://arxiv.org/abs/2406.02764). We propose an adaptive preference loss for reward modeling in the context of RLHF, which improves the flexibility of the reward learning process by adding an adaptive scaling parameter to the loss for each sample in the preference dataset.

## 1. Robotic Control

Our implementations of robotic control tasks are based on Stable-Baselines3 (Raffin et al., 2021) and RL Zoo training framework (Raffin, 2020).

## Setup

To run the code for robotic control, create a Python virtual environment using Conda:

```
conda create -n robotic_control python=3.9
conda activate robotic_control
```

Then install the package dependencies:

```
git clone https://github.com/huggingface/alignment-handbook.git
cd ./APS/robotic_control/
pip install -r requirements.txt
```
