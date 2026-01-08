import numpy as np
import torch
from torch.autograd import Variable

def parse_action_args(args):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        args.continuous = False
        # assert args.dim_actions == 1
        # support multi action
        args.naction_heads = [int(args.num_actions[i]) for i in range(args.dim_actions)]
    else:
        # environment takes continuous action
        actions_heads = args.nactions.split(':')
        if len(actions_heads) == 1 and int(actions_heads[0]) == 1:
            args.continuous = True
        elif len(actions_heads) == 1 and int(actions_heads[0]) > 1:
            args.continuous = False
            args.naction_heads = [int(actions_heads[0]) for _ in range(args.dim_actions)]
        elif len(actions_heads) > 1:
            args.continuous = False
            args.naction_heads = [int(i) for i in actions_heads]
        else:
            raise RuntimeError("--nactions wrong format!")


def select_action(args, action_out, avail_actions=None):
    if args.continuous:
        action_mean, _, action_std = action_out
        action = torch.normal(action_mean, action_std)
        return action.detach()
    else:
        log_p_a = action_out

        # Apply action masking if available actions are provided
        if avail_actions is not None:
            # Convert to torch tensor if needed
            if not isinstance(avail_actions, torch.Tensor):
                avail_actions = torch.from_numpy(avail_actions).float()

            # Apply mask to each action head
            # avail_actions shape: [n_agents, n_actions]
            # log_p_a is list of tensors, each with shape [batch_size, n_agents, n_actions]
            masked_log_p_a = []
            for action_head_idx, action_head_logits in enumerate(log_p_a):
                # For the main action head (index 0), apply action masking
                # For additional heads (e.g., communication), no masking needed
                if action_head_idx == 0:
                    # Clone to avoid modifying original
                    masked_logits = action_head_logits.clone()
                    # Expand avail_actions to match batch dimension: [batch_size, n_agents, n_actions]
                    # avail_actions is [n_agents, n_actions], expand to [1, n_agents, n_actions]
                    avail_mask = avail_actions.unsqueeze(0)
                    # Set log prob to very negative for unavailable actions
                    masked_logits[avail_mask == 0] = -1e10
                    masked_log_p_a.append(masked_logits)
                else:
                    # No masking for other action heads (e.g., communication)
                    masked_log_p_a.append(action_head_logits)
            log_p_a = masked_log_p_a

        p_a = [[z.exp() for z in x] for x in log_p_a]
        ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
        return ret

def translate_action(args, env, action):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        action = [x.squeeze().data.numpy() for x in action]
        actual = action
        return action, actual
    else:
        if args.continuous:
            action = action.data[0].numpy()
            cp_action = action.copy()
            # clip and scale action to correct range
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                cp_action[i] = cp_action[i] * args.action_scale
                cp_action[i] = max(-1.0, min(cp_action[i], 1.0))
                cp_action[i] = 0.5 * (cp_action[i] + 1.0) * (high - low) + low
            return action, cp_action
        else:
            actual = np.zeros(len(action))
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                actual[i] = action[i].data.squeeze()[0] * (high - low) / (args.naction_heads[i] - 1) + low
            action = [x.squeeze().data[0] for x in action]
            return action, actual
