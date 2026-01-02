import sys
import time
import signal
import argparse
import os

import numpy as np
import torch
import visdom
import data
import wandb
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from trainer import Trainer
from multi_processing import MultiProcessTrainer

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')
# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed') # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
parser.add_argument('--enable_mve', action='store_true', default=False,
                    help='Enable MVE counterfactual value learning phase.')
parser.add_argument('--mve_buffer_size', type=int, default=50000,
                    help='Replay buffer size for MVE training.')
parser.add_argument('--mve_batch_size', type=int, default=64,
                    help='Batch size for MVE updates.')
parser.add_argument('--mve_train_steps', type=int, default=0,
                    help='Number of gradient steps for MVE training.')
parser.add_argument('--mve_lr', type=float, default=1e-3,
                    help='Learning rate for MVE network.')
parser.add_argument('--enable_unlearning', action='store_true', default=False,
                    help='Enable value-aware unlearning (phase 3).')
parser.add_argument('--unlearn_episodes', type=int, default=0,
                    help='Number of episodes to run during value-aware unlearning.')
parser.add_argument('--unlearn_lr', type=float, default=None,
                    help='Learning rate for value-aware unlearning (defaults to lrate).')
parser.add_argument('--unlearn_lambda', type=float, default=0.0,
                    help='Sparsity penalty applied to message magnitude during unlearning.')
parser.add_argument('--unlearn_anchor_beta', type=float, default=0.0,
                    help='Weight for action anchoring loss (KL divergence from base policy) during unlearning.')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--save', default='', type=str,
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')
parser.add_argument('--wandb_run', default=None, type=str,
                    help='WandB run name')


parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_prob', type=float, default=None,
                    help='Probability of communicating when using hard attention')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')


init_args_for_env(parser)
args = parser.parse_args()

wandb_run = wandb.init(project="ic3net", name=args.wandb_run, config=vars(args))
wandb.config.update(args)  # Add all arguments to config

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    if args.env_name == "traffic_junction":
        args.comm_action_one = True
# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'


parse_action_args(args)

if args.enable_mve and args.nprocesses > 1:
    print("MVE training currently supports only nprocesses=1. Disabling MVE.")
    args.enable_mve = False

if args.enable_unlearning and args.nprocesses > 1:
    print("Value-aware unlearning currently supports only nprocesses=1. Disabling unlearning.")
    args.enable_unlearning = False

if args.enable_unlearning and not args.enable_mve:
    print("Value-aware unlearning requires MVE. Disabling unlearning.")
    args.enable_unlearning = False

if args.enable_unlearning and not (args.commnet and args.hard_attn):
    print("Value-aware unlearning currently supports hard-attention communication. Disabling unlearning.")
    args.enable_unlearning = False

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print(args)


if args.commnet:
    policy_net = CommNetMLP(args, num_inputs)
elif args.random:
    policy_net = Random(args, num_inputs)
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)

if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.nprocesses > 1:
    trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    trainer = Trainer(args, policy_net, data.init(args.env_name, args))

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()

log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')

# Unlearning log
unlearn_log = dict()
unlearn_log['epoch'] = LogField(list(), False, None, None)
unlearn_log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
unlearn_log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
unlearn_log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
unlearn_log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
unlearn_log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
unlearn_log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
unlearn_log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
unlearn_log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
unlearn_log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
unlearn_log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
unlearn_log['unlearn_loss'] = LogField(list(), True, 'epoch', 'num_steps')
unlearn_log['unlearn_comm_loss'] = LogField(list(), True, 'epoch', 'num_steps')
unlearn_log['unlearn_anchor_loss'] = LogField(list(), True, 'epoch', 'num_steps')
unlearn_log['unlearn_samples'] = LogField(list(), True, 'epoch', None)

if args.plot:
    vis = visdom.Visdom(env=args.plot_env)


def _vector_mean(value):
    if torch.is_tensor(value):
        if value.numel() > 1:
            return value.detach().double().mean().item()
    elif isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.size > 1:
            return float(arr.mean())
    return None


def _add_metric(payload, key, value):
    payload[key] = value
    mean_value = _vector_mean(value)
    if mean_value is not None:
        payload[f'{key}_mean'] = mean_value


def _print_progress(step, total, prefix, metrics=None):
    bar_len = 30
    filled = int(bar_len * step / float(total))
    bar = '█' * filled + '-' * (bar_len - filled)
    percent = 100.0 * step / float(total)
    end_char = '\n' if step >= total else '\r'
    metrics_str = ''
    if metrics:
        parts = []
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, float):
                parts.append(f'{key}={value:.4f}')
            elif isinstance(value, (int, np.integer)):
                parts.append(f'{key}={value}')
            else:
                parts.append(f'{key}={value}')
        if parts:
            metrics_str = ' ' + ' '.join(parts)
    sys.stdout.write(f'{prefix} |{bar}| {percent:6.2f}% ({step}/{total}){metrics_str}{end_char}')
    sys.stdout.flush()

def run(num_epochs):
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            s = trainer.train_batch(ep)
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)

        print('Epoch {}\tReward {}\tTime {:.2f}s'.format(
                epoch, stat['reward'], epoch_time
        ))

        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            print('Success: {:.2f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            print('Steps-taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))

        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

        if wandb_run is not None:
            payload = {'epoch': epoch, 'epoch_time': epoch_time}
            # Only log keys that were computed in this epoch
            for key in ['reward', 'enemy_reward', 'add_rate', 'success', 'steps_taken',
                        'comm_action', 'enemy_comm', 'value_loss', 'action_loss', 'entropy']:
                if key in stat:
                    _add_metric(payload, key, stat[key])
            wandb.log(payload, step=epoch)
        _print_progress(ep + 1, num_epochs, 'Training')

def run_mve_phase():
    if not args.enable_mve or args.mve_train_steps <= 0:
        return

    target_trainer = trainer.trainer if isinstance(trainer, MultiProcessTrainer) else trainer
    if not hasattr(target_trainer, 'train_mve_step'):
        print("MVE components not initialized; skipping MVE phase.")
        return

    # Clear stale transitions from pre-training and refill with fresh on-policy data
    # This ensures MVE learns value differences for the final converged policy only
    print("Clearing MVE buffer and collecting fresh on-policy transitions...")
    buffer_size = target_trainer.fill_mve_buffer()
    print(f"Collected {buffer_size} fresh transitions from converged policy")

    print(f"Starting MVE training for {args.mve_train_steps} steps using replay buffer of size {buffer_size}")
    mve_log = dict()
    mve_log['mve_step'] = LogField(list(), False, None, None)
    mve_log['epoch'] = LogField(list(), False, None, None)
    mve_log['epoch_time'] = LogField(list(), False, None, None)
    mve_log['mve_loss'] = LogField(list(), True, 'mve_step', None)
    mve_log['mve_samples'] = LogField(list(), True, 'mve_step', None)
    mve_log['value_loss'] = LogField(list(), True, 'mve_step', 'num_steps')
    mve_log['action_loss'] = LogField(list(), True, 'mve_step', 'num_steps')
    mve_log['entropy'] = LogField(list(), True, 'mve_step', 'num_steps')
    mve_log['reward'] = LogField(list(), True, 'mve_step', 'num_episodes')
    mve_log['success'] = LogField(list(), True, 'mve_step', 'num_episodes')
    mve_log['steps_taken'] = LogField(list(), True, 'mve_step', 'num_episodes')
    mve_log['comm_action'] = LogField(list(), True, 'mve_step', 'num_steps')
    next_wandb_step = getattr(wandb.run, 'step', 0) if wandb_run is not None else 0
    for step in range(args.mve_train_steps):
        step_begin_time = time.time()
        stat = dict()
        res = target_trainer.train_mve_step(step)
        if res is None:
            print("MVE training halted early due to insufficient samples.")
            break
        merge_stat(res, stat)

        step_time = time.time() - step_begin_time
        mve_step = len(mve_log['mve_step'].data) + 1
        stat['epoch'] = mve_step
        stat['epoch_time'] = step_time
        if 'reward' in stat and 'reward_mean' not in stat:
            reward_mean = _vector_mean(stat['reward'])
            if reward_mean is not None:
                stat['reward_mean'] = reward_mean
        if 'comm_action' in stat and 'comm_action_mean' not in stat:
            comm_action_mean = _vector_mean(stat['comm_action'])
            if comm_action_mean is not None:
                stat['comm_action_mean'] = comm_action_mean
        for k, v in mve_log.items():
            if k == 'mve_step':
                v.data.append(mve_step)
            elif k == 'epoch':
                v.data.append(stat['epoch'])
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)

        if 'mve_loss' in stat:
            print('MVE Step {}\tMVE Loss {:.4f}\tTime {:.2f}s'.format(
                    mve_step, stat['mve_loss'], step_time
            ))
        else:
            print('MVE Step {}\tTime {:.2f}s'.format(
                    mve_step, step_time
            ))

        if 'mve_samples' in stat:
            print('MVE Samples: {}'.format(stat['mve_samples']))

        # Print policy stats (policy performance monitoring during MVE training)
        if 'reward' in stat:
            print('Reward: {}'.format(stat['reward']))
        if 'success' in stat:
            print('Success: {:.2f}'.format(stat['success']))
        if 'steps_taken' in stat:
            print('Steps-taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat:
            print('Comm-Action: {}'.format(stat['comm_action']))

        if args.plot:
            for k, v in mve_log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(mve_log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

        if wandb_run is not None:
            next_wandb_step += 1
            payload = {
                'mve_step': mve_step,
                'mve_step_time': step_time,
                'epoch': stat['epoch'],
                'epoch_time': stat['epoch_time']
            }
            for key in ['mve_loss', 'mve_samples', 'value_loss', 'action_loss', 'entropy',
                        'reward', 'success', 'steps_taken', 'comm_action']:
                if key in stat:
                    _add_metric(payload, key, stat[key])
            wandb.log(payload, step=next_wandb_step)
        _print_progress(step + 1, args.mve_train_steps, 'MVE    ')
    if args.mve_train_steps > 0:
        sys.stdout.write('\n')
        sys.stdout.flush()
    if mve_log['mve_loss'].data and wandb_run is not None:
        avg_loss = np.mean(mve_log['mve_loss'].data)
        total_samples = sum(mve_log['mve_samples'].data)
        next_wandb_step += 1
        wandb.log({
            'mve_avg_loss': avg_loss,
            'mve_total_samples': total_samples,
            'mve_steps_completed': len(mve_log['mve_step'].data)
        }, step=next_wandb_step)

def run_unlearning_phase(base_policy=None):
    if not args.enable_unlearning or args.unlearn_episodes <= 0:
        return

    target_trainer = trainer.trainer if isinstance(trainer, MultiProcessTrainer) else trainer
    if not hasattr(target_trainer, 'train_value_unlearning_episode') or target_trainer.unlearn_optimizer is None:
        print("Unlearning components not initialized; skipping value-aware unlearning.")
        return

    anchor_msg = f" with action anchoring (beta={args.unlearn_anchor_beta})" if base_policy is not None and args.unlearn_anchor_beta > 0 else ""
    print(f"Starting value-aware unlearning for {args.unlearn_episodes} episodes{anchor_msg}.")
    next_wandb_step = getattr(wandb.run, 'step', 0) if wandb_run is not None else 0

    for ep in range(args.unlearn_episodes):
        epoch_begin_time = time.time()
        stat = dict()

        res = target_trainer.train_value_unlearning_episode(ep, base_policy=base_policy)
        if res is None:
            print("Unlearning halted early due to missing samples or comm actions.")
            break
        merge_stat(res, stat)

        epoch_time = time.time() - epoch_begin_time
        epoch = len(unlearn_log['epoch'].data) + 1
        for k, v in unlearn_log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)

        print('Epoch {}\tReward {}\tTime {:.2f}s'.format(
                epoch, stat['reward'], epoch_time
        ))

        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            print('Success: {:.2f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            print('Steps-taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))
        if 'unlearn_comm_loss' in stat.keys():
            print('Unlearn-Comm-Loss: {:.4f}'.format(stat['unlearn_comm_loss']))
        if 'unlearn_anchor_loss' in stat.keys() and stat['unlearn_anchor_loss'] > 0:
            print('Unlearn-Anchor-Loss: {:.4f}'.format(stat['unlearn_anchor_loss']))

        if args.plot:
            for k, v in unlearn_log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(unlearn_log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

        if wandb_run is not None:
            next_wandb_step += 1
            payload = {'epoch': epoch, 'epoch_time': epoch_time}
            # Only log keys that were computed in this epoch
            for key in ['reward', 'enemy_reward', 'add_rate', 'success', 'steps_taken',
                        'comm_action', 'enemy_comm', 'value_loss', 'action_loss', 'entropy',
                        'unlearn_loss', 'unlearn_comm_loss', 'unlearn_anchor_loss', 'unlearn_samples']:
                if key in stat:
                    _add_metric(payload, key, stat[key])
            wandb.log(payload, step=next_wandb_step)
        _print_progress(ep + 1, args.unlearn_episodes, 'Unlearning')

def save(path):
    # Always treat path as directory and write model.pt inside it
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, 'model.pt')

    target_trainer = trainer.trainer if isinstance(trainer, MultiProcessTrainer) else trainer
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if hasattr(target_trainer, 'mve_net') and target_trainer.mve_net is not None:
        d['mve_net'] = target_trainer.mve_net.state_dict()
    torch.save(d, save_path)

def load(path):
    d = torch.load(path, weights_only=False)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])
    target_trainer = trainer.trainer if isinstance(trainer, MultiProcessTrainer) else trainer
    if 'mve_net' in d and hasattr(target_trainer, 'mve_net') and target_trainer.mve_net is not None:
        target_trainer.mve_net.load_state_dict(d['mve_net'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.load != '':
    load(args.load)

run(args.num_epochs)

# Save a frozen copy of the base policy for action anchoring during unlearning
base_policy_net = None
if args.enable_unlearning and args.unlearn_anchor_beta > 0:
    import copy
    base_policy_net = copy.deepcopy(policy_net)
    base_policy_net.eval()
    for param in base_policy_net.parameters():
        param.requires_grad = False
    print(f"Saved frozen base policy for action anchoring (beta={args.unlearn_anchor_beta})")

run_mve_phase()
run_unlearning_phase(base_policy=base_policy_net)
if args.display:
    env.end_display()

if args.save != '':
    save(args.save)

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)
