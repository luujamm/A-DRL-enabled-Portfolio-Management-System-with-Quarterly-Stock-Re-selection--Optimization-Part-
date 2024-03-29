import argparse


def define_args():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--algo', default='PPO')
    parser.add_argument('--model', default='tcn')
    parser.add_argument('--case', default=3, type=int)
    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('-t', '--trial', action='store_true')
    parser.add_argument('-bt', '--backtest', action='store_true')
    
    # env
    parser.add_argument('--trading_cost', default=0.002, type=float)
    parser.add_argument('--state_length', default=50, type=int)
    parser.add_argument('--lam', default=0.5, type=float)
    parser.add_argument('-l', '--train_period_length', default=652, type=int)
    parser.add_argument('-lv', '--val_period_length', default=102, type=int)
    parser.add_argument('-lt', '--test_period_length', default=64, type=int)
    
    # model
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('--lrv', default=1e-4, type=float)
    parser.add_argument('--lra', default=0.0015, type=float)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('-p', '--dropout_rate', default=0, type=float)
    
    # algo
    parser.add_argument('--gamma', default=0.99, type=float) 

    # algo (PPO)
    parser.add_argument('--action_std_train', default=0.01, type=float)
    parser.add_argument('--action_std_decay_rate', default=0.99, type=float)
    parser.add_argument('--action_std_test', default=1e-10, type=float)
    parser.add_argument('-s', '--trajectory_sample_times', default=8, type=int)
    parser.add_argument('-k', '--K_epochs', default=10, type=int)
    parser.add_argument('--eps_clip', default=0.2, type=float)
    parser.add_argument('--dist_entropy_coef', default=1e-2, type=float)
    
    # algo (DDPG)
    parser.add_argument('--Gau_var', default=0.2, type=float)
    parser.add_argument('--Gau_decay', default=0.99995, type=float)
    parser.add_argument('--capacity', default=10000, type=int)

    # algo (SAC)
    parser.add_argument('-a', '--alpha', default=0.0002, type=float)

    # train
    parser.add_argument('--train_iter', default=30, type=int)
        
    # test
    parser.add_argument('-i', '--iter', default='all')
    parser.add_argument('--test_diff', default=[], type=list)
    
    return parser.parse_args()