import itertools
import numpy as np
import time
import random
import matplotlib.pyplot as plt

from test import test
from environment.portfolio_new import PortfolioEnv
from utils.data import *
from utils.yahoodownloader import get_data
from utils.create_repository import create_q_path
from utils.abstract import write_abstract
from utils.draw import draw_train_summary
from utils.recorder import Recorder
from utils.evaluation import risk_free_return


SEED_STEP = 42
EPS = 1e-8


def train(args, agent, recorder, target_stocks, train_history, train_dating, train_start_date, iteration, path):
    agent.train()
    
    action_dim = len(target_stocks) + 1
    sample_times = args.trajectory_sample_times# if args.algo == 'PPO' else 1
    rfr = risk_free_return()
    
    iteration_start_time = time.time()
    train_history, train_dating, train_data = transform_data(args, train_history, train_dating)
    
    # recorder
    train_correct = 0
    # recorder
       
    model_fn = path + '/agent_test{}_iter{}.pth'.format(args.case, iteration)
    
    if args.algo == 'PPO':
        agent.std = agent.std_train
    '''env = PortfolioEnv(args, train_history, train_data, action_dim, 
                            train_dating, train_history, steps=args.train_period_length,
                        sample_start_date=train_start_date)'''
                            
    for st in range(sample_times):
        if args.algo == 'DDPG':
            start_date = index_to_date(date_to_index(train_start_date, train_dating) + st, train_dating) 
        else: start_date = train_start_date
        env = PortfolioEnv(args, train_history, train_data, action_dim, 
                            train_dating, train_history, steps=args.train_period_length,
                        sample_start_date=train_start_date)
        trajectory_reward = 0
        daily_return = []
        observation, _ = env.reset()
        state = generate_state(observation)
        current_weights = get_init_action(action_dim, random=True)
        
        for t in itertools.count(start=1):
            # choose action
            if args.algo == 'PPO':       
                use_action, action, action_log_prob, _ = agent.choose_action(state, current_weights)
            elif args.algo == 'DDPG':
                use_action = agent.choose_action(state, current_weights)
              
            # execute action
            new_weights, next_observation, reward, excess_ew_return, done, trade_info, _ = env.step(current_weights, use_action)
            
            # recorder
            if excess_ew_return > 0:
                train_correct += 1 / args.train_period_length / sample_times

            
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
            elif args.algo == 'DDPG':
                agent.append(current_weights, state, use_action, reward, state_, done)
            else: #DPG
                agent.append(state, state_, next_value, current_weights, action, action_log_prob, reward, done, trade_info['return'])
            
            if args.algo == 'DDPG' and len(agent.memory.epi_buffer) == args.capacity:
                agent.update()

            state = state_
            current_weights = new_weights
            trajectory_reward += reward
            
            if done:
                recorder.train.values.append(trade_info["portfolio_value"])
                recorder.train.rewards.append(trajectory_reward)
                if args.algo == 'DDPG':
                    agent.epi_append()
                break
    if args.algo == 'PPO':
        agent.update()
                
    mean_reward = np.mean(recorder.train.rewards) / args.train_period_length
    
    # recorder
    agent.train_reward.append(mean_reward)
    agent.train_value.append(np.mean(recorder.train.values))
    agent.train_acc.append(train_correct)
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
    if args.algo == 'DPG':
        start_idx = date_to_index(train_start_date, train_dating)
        train_start_date = index_to_date(start_idx + random.randint(0, len(train_dating) - start_idx - args.batch_size), train_dating)
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

    draw_train_summary(args, agent, path)
    plt.plot(agent.train_loss)
    plt.savefig(path + '/loss.png')