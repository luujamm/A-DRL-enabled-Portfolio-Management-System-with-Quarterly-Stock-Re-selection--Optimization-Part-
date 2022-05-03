from .PPO import PPO
from .SAC import SAC
from .DDPG import DDPG


def Agent(args, action_dim):
    if args.algo == 'PPO':
        return PPO(args, action_dim)
    elif args.algo == 'SAC':
        return SAC(args, action_dim)
    elif args.algo == 'DDPG':
        return DDPG(args, action_dim)
    else:
        raise NotImplementedError

