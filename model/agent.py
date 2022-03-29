from .PPO import PPO


def Agent(args, action_dim):
    if args.algo == 'PPO':
        return PPO(args, action_dim)
    else:
        raise NotImplementedError

