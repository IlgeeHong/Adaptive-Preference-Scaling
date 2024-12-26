import os
import numpy as np
from matplotlib import pyplot as plt

def load_percentile_data(args):
    results = []
    num_frac = int(args.rl_steps / 1e4)
    for seed in args.seed:
        raw_data = np.load(f'{args.input_path}/rlhf_{args.env_id}_{seed}_{args.lr}_{args.ppo_clip}_{args.rl_steps}_{args.hidden_layers}_{args.hidden_size}_{args.n_rm_epochs}_{args.rm_lr}_{args.rm_batch_size}_{args.enable_individualized_tau}_{args.quadratic_penalty}_{args.n_tau_iters}_{args.tau_init}_{args.tau_min}_{args.tau_max}_{args.rho}/evaluations.npz')
        results.append(raw_data['results'][:num_frac,:].mean(axis=1).max())
    return results

def load_data(args):
    num_frac = int(args.rl_steps / 1e4)
    true_reward = np.empty((num_frac,0))
    for seed in args.seed:
        raw_data = np.load(f'{args.input_path}/rlhf_{args.env_id}_{seed}_{args.lr}_{args.ppo_clip}_{args.rl_steps}_{args.hidden_layers}_{args.hidden_size}_{args.n_rm_epochs}_{args.rm_lr}_{args.rm_batch_size}_{args.enable_individualized_tau}_{args.quadratic_penalty}_{args.n_tau_iters}_{args.tau_init}_{args.tau_min}_{args.tau_max}_{args.rho}/evaluations.npz')
        true_reward = np.hstack((true_reward, raw_data['results'][:num_frac,:].mean(axis=1).reshape(-1,1)))
    best_ind = [i for i in range(len(true_reward)) if true_reward[i].mean() == true_reward.mean(axis=1).max()]
    episode_reward = true_reward.mean(axis=1).max()
    std = true_reward[best_ind].std()
    print(f"Return: {round(episode_reward,2)}, std: {round(std,2)}")
    return true_reward

def learning_curve(args, data):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    fig, ax = plt.subplots(figsize=(8,8))
    num_frac = args.rl_steps / 1e6
    color = "black"
    if args.enable_individualized_tau:
        label = "Ada-Pref"
    else:
        label = "Pref"
    x = np.arange(0.0, num_frac, 0.01)
    y = EMA(data, args.smoothing_factor)
    ax.plot(x, y.mean(axis=1), linewidth=5, color=color, label=label)
    ax.fill_between(x, y.mean(axis=1) + y.std(axis=1), y.mean(axis=1) - y.std(axis=1), alpha=0.3, color=color)
    ax.set_ylabel("Return", fontsize = 30)
    ax.legend(loc='lower right', prop={'size': 30})
    ax.set_xlabel("Timesteps (1e6)", fontsize = 30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(f'{args.output_path}/Learning_Curve-{args.env_id}-{label}.png', bbox_inches="tight") 

def percentile_plot(args, data):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    data.sort()
    x = np.arange(1/len(data), 1+1/len(data), 1/len(data))*100
    color = "black"
    if args.enable_individualized_tau:
        label = "Ada-Pref"
    else:
        label = "Pref"
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(x, data, linewidth=5, color=color, label=label)
    ax.set_ylabel("Return", fontsize = 30)
    ax.legend(loc='lower right', prop={'size': 30})
    ax.set_xlabel("Percentile", fontsize = 30)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(f'{args.output_path}/Percentile_Plot-{args.env_id}-{label}.png', bbox_inches="tight")  
    
def EMA(input_list, alpha):
    ema_values = [input_list[0,:]]
    for value in input_list[1:,:]:
        ema = np.average(np.vstack((ema_values[-1],value)), axis=0, weights=[1-alpha,alpha])
        ema_values.append(ema)
    return np.asarray(ema_values)


