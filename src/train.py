import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from src.environment.portfolio_env import PortfolioEnv
from src.test import test
from src.utils.abstract import write_abstract
from src.utils.create_repository import create_q_path
from src.utils.data import *
from src.utils.draw import show_train_results
from src.utils.evaluation import risk_free_return
from src.utils.recorder import Recorder


SEED_STEP = 42
EPS = 1e-8


def train(args, agent, recorder, target_stocks, train_history, train_dating, train_start_date, iteration, path):
    agent.train()
    
    action_dim = len(target_stocks) + 1
    sample_times = args.trajectory_sample_times if args.algo == 'PPO' else 1
    rfr = risk_free_return()
    iteration_start_time = time.time()
    train_history, train_dating, train_data = transform_data(args, train_history, train_dating)
       
    model_fn = path + '/agent_test{}_iter{}.pth'.format(args.case, iteration)
    
    if args.algo == 'PPO':
        agent.std = agent.std_train
                            
    for st in range(sample_times):
        if args.algo == 'DDPG': 
            agent.action_noise.reset()

        env = PortfolioEnv(args, train_history, train_data, action_dim, 
                            train_dating, train_history, steps=args.train_period_length,
                        sample_start_date=train_start_date)

        trajectory_reward = 0
        daily_return = []
        observation, _ = env.reset()
        state = generate_state(observation)
        current_weights = get_init_action(action_dim, random=True)
        old_action = current_weights.copy()
        
        for t in itertools.count(start=1):
            # choose action
            if args.algo == 'PPO':       
                use_action, action, action_log_prob, _ = agent.choose_action(state, current_weights)

            elif args.algo == 'DDPG':
                use_action = agent.choose_action(state, old_action)

            elif args.algo == 'SAC':
                use_action, action, action_log_prob = agent.choose_action(state, current_weights)
              
            # execute action
            new_weights, next_observation, reward, excess_ew_return, done, trade_info, _ = env.step(current_weights, use_action)
            
            state_ = generate_state(next_observation)

            if args.algo == 'PPO':
                next_value = agent.choose_action(state_, new_weights)[-2].item()

            else: next_value = 0
                            
            daily_return.append(trade_info["rate_of_return"])

            if args.algo == 'DPG' and agent.algo.buffer.__len__() == args.batch_size:
                done = 1

            # store transition
            if args.algo == 'PPO':
                agent.append(state, next_value, current_weights, action, action_log_prob, reward, done)

            elif args.algo == 'SAC':
                agent.append(state, state_, current_weights, new_weights, use_action, reward, done)

            elif args.algo == 'DDPG':
                agent.append(old_action, state, use_action, reward, state_, done)

            else:
                raise NotImplementedError
            
            if args.algo == 'DDPG' and agent.memory.__len__() > args.batch_size:
                agent.update()

            state = state_
            current_weights = new_weights
            old_action = use_action
            trajectory_reward += reward
            
            if done:
                recorder.train.values.append(trade_info["portfolio_value"])
                recorder.train.rewards.append(trajectory_reward)
                break

    if args.algo == 'PPO' or args.algo == 'SAC':
        agent.update()
                
    mean_reward = np.mean(recorder.train.rewards) / args.train_period_length
    
    # recorder
    agent.train_reward.append(mean_reward)
    agent.train_value.append(np.mean(recorder.train.values))
    # recorder
    
    print('=' * 120, '\nIter {}  Mean reward: {:.6f} Portfolio value: {:.4f}'
          .format(iteration, mean_reward, np.mean(recorder.train.values)))
    return model_fn


def policy_learn(args, agent, target_stocks, path, year, Q):
    print('Start Training')
    last_use_time = 0
    start_time = time.time()
    seed = args.seed
    recorder = Recorder()
    
    pretrain_start_date, tu_start, train_start_date, train_end_date, val_end_date = define_dates(args, year, Q)
    
    train_history, train_dating = get_data(target_stocks, year, Q, 'train')
    val_history, val_dating = get_data(target_stocks, year, Q, 'val') 
    tu_his = get_data(target_stocks, year, Q, 'tu')[0]
    start_idx = np.argwhere(val_dating == train_end_date)[0][0] + 1
    benchmarks = get_data(['^GSPC', '^OEX'], year, Q, 'val', bench=True)[0][:, start_idx:, :] # S&P 500, S&P 100
    recorder.benchmark.values.append(benchmarks)
    
    quarter = str(year) + 'Q' + str(Q)
    print(quarter)

    if args.case == 3:
        path = create_q_path(path, quarter)

    write_abstract(args, path, target_stocks, train_start_date, train_end_date)

    for it in range(args.train_iter):
        agent.setup_seed_(seed)
        recorder.clear()

        model_fn = train(args, agent, recorder, target_stocks, train_history, train_dating, train_start_date, it+1, path)  

        args.val = True  
        test(args, agent, recorder, target_stocks, val_history,  val_dating, train_end_date,
             it+1, tu_his, model_fn=model_fn, path=path)

        use_time = time.time() - start_time
        remain_time = (use_time - last_use_time) * (args.train_iter - it - 1)
        print('Time usage: {:.0f} min {:.0f} s, remain: {:.0f} min {:.0f} s'
              .format(use_time//60, use_time%60, remain_time//60, remain_time%60))
        last_use_time = use_time
        seed+=SEED_STEP

    show_train_results(args, agent, path)
    plt.plot(agent.train_loss)
    plt.savefig(path + '/loss.png')