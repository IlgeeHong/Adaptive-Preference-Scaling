import json
import numpy as np
import torch
import argparse
from matplotlib import pyplot as plt
from eval_utils import load_percentile_data, load_data, percentile_plot, learning_curve

def eval(args):
    args.seed = list(range(args.seed))
    percentile_data = load_percentile_data(args)
    learning_curve_data = load_data(args)
    percentile_plot(args, percentile_data)
    learning_curve(args, learning_curve_data)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--input_path", default="/path/to/train/result/directory", type=str)
    parser.add_argument("--output_path", default="/path/to/eval/result/directory", type=str)
    parser.add_argument("--env_id", default="AntBulletEnv-v0")
    parser.add_argument("--lr", default=3e-4, type=float)                  
    parser.add_argument("--ppo_clip", default=0.2, type=float)             
    parser.add_argument("--rl_steps", default=3000000, type=int)           
    parser.add_argument("--hidden_layers", default=2, type=int)            
    parser.add_argument("--hidden_size", default=64, type=int)                       
    parser.add_argument("--n_rm_epochs", default=1, type=int)              
    parser.add_argument("--rm_lr", default=5e-4, type=float)               
    parser.add_argument("--rm_batch_size", default=64, type=int)           
    parser.add_argument("--enable_individualized_tau", action='store_true')
    parser.add_argument("--quadratic_penalty", action='store_true')
    parser.add_argument("--n_tau_iters", default=0, type=int)
    parser.add_argument("--tau_init", default=1.0, type=float)
    parser.add_argument("--tau_min", default=0.0, type=float)
    parser.add_argument("--tau_max", default=0.0, type=float)
    parser.add_argument("--rho", default=0.0, type=float)
    parser.add_argument("--smoothing_factor", default=0.1, type=float)
    args = parser.parse_args()
    eval(args)