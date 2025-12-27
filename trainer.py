from collections import namedtuple
from inspect import getargspec
import random
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from models import MVENetwork
from utils import *
from action_utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, item):
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]
        self.mve_enabled = getattr(args, 'enable_mve', False)
        if self.mve_enabled:
            self.mve_net = MVENetwork(args)
            self.mve_optimizer = optim.Adam(self.mve_net.parameters(), lr=args.mve_lr)
            self.mve_buffer = ReplayBuffer(args.mve_buffer_size)
        else:
            self.mve_net = None
            self.mve_optimizer = None
            self.mve_buffer = None

        self.unlearning_enabled = getattr(args, 'enable_unlearning', False) and self.mve_enabled
        if self.unlearning_enabled:
            lr = args.unlearn_lr if getattr(args, 'unlearn_lr', None) not in (None, 0) else args.lrate
            self.unlearn_optimizer = optim.RMSprop(policy_net.parameters(),
                lr = lr, alpha=0.97, eps=1e-6)
        else:
            self.unlearn_optimizer = None


    def get_episode(self, epoch):
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.args.max_steps):
            misc = dict()
            hidden_repr = None
            comm_action_snapshot = None
            prev_hid_for_buffer = None
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                if self.mve_enabled:
                    prev_hid_for_buffer = self._detach_hidden_state(prev_hid)
                x = [state, prev_hid]
                if self.mve_enabled:
                    action_out, value, prev_hid, hidden_repr = self.policy_net(x, info, return_hidden=True)
                else:
                    action_out, value, prev_hid = self.policy_net(x, info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                if self.mve_enabled:
                    action_out, value, hidden_repr = self.policy_net(x, info, return_hidden=True)
                else:
                    action_out, value = self.policy_net(x, info)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)
                comm_action_snapshot = np.array(info['comm_action'], copy=True)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            if hidden_repr is not None:
                misc['hidden_repr'] = hidden_repr.detach().clone()
            if comm_action_snapshot is not None:
                misc['comm_action'] = np.array(comm_action_snapshot, copy=True)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            if self.mve_enabled:
                self._store_mve_transition(state, value, hidden_repr, comm_action_snapshot, prev_hid_for_buffer, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)

    def _detach_hidden_state(self, hidden):
        if hidden is None:
            return None
        if isinstance(hidden, tuple):
            return tuple(h.detach().clone() for h in hidden)
        return hidden.detach().clone()

    def _store_mve_transition(self, state, value, hidden_repr, comm_action, prev_hidden, misc):
        if self.mve_buffer is None or hidden_repr is None or comm_action is None:
            return

        sample = {
            'state': state.detach().clone(),
            'value': value.detach().clone(),
            'hidden_state': hidden_repr.detach().clone(),
            'comm_action': np.array(comm_action, copy=True),
            'prev_hidden': self._detach_hidden_state(prev_hidden),
            'alive_mask': misc.get('alive_mask', None)
        }
        self.mve_buffer.push(sample)

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)

        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        deltas = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy


        loss.backward()

        return stat

    def compute_policy_stats(self, batch):
        """Compute policy loss statistics WITHOUT calling backward.

        This is used for logging purposes when we don't want to affect gradients.
        Returns the same stats as compute_grad but without the backward pass.

        Returns:
            stat: Dict with 'action_loss', 'value_loss', 'entropy' as scalars
        """
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)

        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        deltas = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()

        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()

        # Do NOT call backward here - that's the key difference from compute_grad
        return stat

    def _compute_delta_q(self, sample, agent_idx):
        comm_action = sample.get('comm_action', None)
        if comm_action is None:
            return None

        state = sample['state']
        value = sample['value']
        prev_hidden = sample.get('prev_hidden', None)
        alive_mask = sample.get('alive_mask', None)

        if alive_mask is not None:
            mask = np.asarray(alive_mask).reshape(-1)
            if agent_idx >= mask.shape[0] or mask[agent_idx] == 0:
                return None

        q_real = value.view(state.shape[0], self.args.nagents, -1)[0, agent_idx, 0].detach()

        null_comm = np.array(comm_action, copy=True)
        null_comm[agent_idx] = 0
        info = {'comm_action': null_comm}
        if alive_mask is not None:
            info['alive_mask'] = alive_mask

        with torch.no_grad():
            if self.args.recurrent:
                hidden_in = prev_hidden
                if hidden_in is None:
                    if self.args.rnn_type == 'LSTM':
                        hidden_in = self.policy_net.init_hidden(batch_size=state.shape[0])
                    else:
                        hidden_in = torch.zeros(state.shape[0], self.args.nagents, self.args.hid_size, dtype=state.dtype)
                _, null_value, _ = self.policy_net([state, hidden_in], info)
            else:
                _, null_value = self.policy_net(state, info)

        q_null = null_value.view(state.shape[0], self.args.nagents, -1)[0, agent_idx, 0]
        return q_real - q_null

    def train_mve_step(self, epoch=0):
        if self.mve_buffer is None or self.mve_net is None:
            return None

        if len(self.mve_buffer) < self.args.mve_batch_size:
            return None

        # Train MVE network from replay buffer
        transitions = self.mve_buffer.sample(self.args.mve_batch_size)
        preds = []
        targets = []
        for sample in transitions:
            comm_action = sample.get('comm_action', None)
            hidden_state = sample.get('hidden_state', None)

            if comm_action is None or hidden_state is None:
                continue

            hidden_state = hidden_state.view(-1, self.args.nagents, self.args.hid_size)
            for agent_idx in range(self.args.nagents):
                delta_q = self._compute_delta_q(sample, agent_idx)
                if delta_q is None:
                    continue
                h_i = hidden_state[0, agent_idx].unsqueeze(0)
                msg = torch.tensor([[comm_action[agent_idx]]], dtype=h_i.dtype)
                preds.append(self.mve_net(h_i, msg))
                targets.append(delta_q.view(1))

        if len(preds) == 0:
            return None

        preds = torch.cat(preds, dim=0).squeeze()
        targets = torch.cat(targets, dim=0).squeeze()
        loss = (preds - targets.detach()).pow(2).mean()

        self.mve_optimizer.zero_grad()
        loss.backward()
        self.mve_optimizer.step()

        res = {'mve_loss': loss.item(), 'mve_samples': preds.numel()}

        # Collect policy stats for logging (policy is frozen during MVE training)
        # Run an episode to get current policy performance
        batch, stat = self.run_batch(epoch)
        if len(batch.state) > 0:
            policy_stats = self.compute_policy_stats(batch)
            merge_stat(policy_stats, res)
            res.update(stat)

        return res

    def train_value_unlearning_episode(self, epoch):
        if not self.unlearning_enabled or self.unlearn_optimizer is None or self.mve_net is None:
            return None

        # Only applicable when communication actions are present (hard attention path).
        comm_head_idx = self.args.dim_actions - 1 if self.args.hard_attn and self.args.commnet else None
        if comm_head_idx is None or comm_head_idx >= len(self.args.num_actions):
            return None

        self.policy_net.train()
        self.mve_net.eval()
        batch, stat = self.run_batch(epoch)
        if len(batch.state) == 0:
            return None

        loss_terms = []
        samples = 0
        for idx in range(len(batch.state)):
            misc = batch.misc[idx] if isinstance(batch.misc[idx], dict) else dict()
            hidden_repr = misc.get('hidden_repr', None)
            comm_action = misc.get('comm_action', None)
            alive_mask = misc.get('alive_mask', None)

            if hidden_repr is None or comm_action is None:
                continue

            action_out = batch.action_out[idx]
            if action_out is None or len(action_out) <= comm_head_idx:
                continue

            comm_logits = action_out[comm_head_idx]
            log_probs = comm_logits.view(-1, self.args.nagents, self.args.num_actions[comm_head_idx])
            hidden_repr = hidden_repr.view(-1, self.args.nagents, self.args.hid_size)
            comm_action_arr = np.asarray(comm_action).reshape(-1)
            alive_arr = np.asarray(alive_mask).reshape(-1) if alive_mask is not None else None

            for agent_idx in range(self.args.nagents):
                if agent_idx >= comm_action_arr.shape[0]:
                    continue
                if alive_arr is not None and (agent_idx >= alive_arr.shape[0] or alive_arr[agent_idx] == 0):
                    continue

                msg_val = comm_action_arr[agent_idx]
                msg_tensor = torch.tensor([[msg_val]], dtype=hidden_repr.dtype, device=hidden_repr.device)
                with torch.no_grad():
                    value_est = self.mve_net(hidden_repr[:, agent_idx, :].detach(), msg_tensor).squeeze()

                log_prob = log_probs[:, agent_idx, int(msg_val)].squeeze()
                advantage = value_est - self.args.unlearn_lambda * torch.abs(msg_tensor.squeeze())
                loss_terms.append(-advantage * log_prob)
                samples += 1

        if samples == 0 or len(loss_terms) == 0:
            return None

        loss = torch.stack(loss_terms).mean()
        self.unlearn_optimizer.zero_grad()
        loss.backward()
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= samples
        self.unlearn_optimizer.step()

        res = {'unlearn_loss': loss.item(), 'unlearn_samples': samples}
        res.update(stat)

        # Collect policy stats (value/action/entropy) without affecting gradients, to mirror main training logs.
        policy_stats = self.compute_policy_stats(batch)
        merge_stat(policy_stats, res)

        return res

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat

    def state_dict(self):
        state = {'optimizer': self.optimizer.state_dict()}
        if self.mve_optimizer is not None:
            state['mve_optimizer'] = self.mve_optimizer.state_dict()
        if self.unlearn_optimizer is not None:
            state['unlearn_optimizer'] = self.unlearn_optimizer.state_dict()
        return state

    def load_state_dict(self, state):
        if isinstance(state, dict) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            if self.mve_optimizer is not None and 'mve_optimizer' in state:
                self.mve_optimizer.load_state_dict(state['mve_optimizer'])
            if self.unlearn_optimizer is not None and 'unlearn_optimizer' in state:
                self.unlearn_optimizer.load_state_dict(state['unlearn_optimizer'])
        else:
            self.optimizer.load_state_dict(state)
