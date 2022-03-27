import itertools
import numpy as np
import time
import random

from test import test
from environment.portfolio_new import PortfolioEnv
from utils.data import *
from utils.yahoodownloader import get_data
from utils.create_repository import create_path, create_q_path
from utils.abstract import write_abstract
from utils.draw import draw_train_summary
from utils.recorder import Recorder
from utils.evaluation import risk_free_return


SEED_STEP = 42
EPS = 1e-8


def train(args, agent, recorder, target_stocks, train_history, train_dating, train_start_date, iteration, path):
    agent.train()
    
    action_dim = len(target_stocks) + 1
    sample_times = args.trajectory_sample_times if args.algo == 'PPO' else 1
    rfr = risk_free_return()
    
    iteration_start_time = time.time()
    train_history, train_dating, train_data = transform_data(args, train_history, train_dating)
    
    # recorder
    train_correct = 0
    # recorder
    
    if args.closeae: 
        index_bias = 51
    else: 
        index_bias = 21
    max_period = len(train_dating) - args.state_length - index_bias - 1
   
    args.train_period_length = max_period 
    
    for epi_itr in reversed(range(max_period - args.train_period_length + 1)): 
        model_fn = path + '/agent_test{}_iter{}.pth'.format(args.case, iteration)
        epi_end_idx = len(train_dating) - epi_itr - index_bias
        if args.algo == 'PPO':
            agent.std = agent.std_train
        env = PortfolioEnv(args, train_history, train_data, action_dim, 
                               train_dating, train_history, steps=args.train_period_length,
                           sample_start_date=train_start_date)
                               
        for st in range(sample_times):
            trajectory_reward = 0
            state, observation, _ = env.reset()
            
            state = generate_state(observation)
            
            current_weights = get_init_action(action_dim, random=True)
            daily_return = []
            
            for t in itertools.count(start=1):
                
                use_action, action, action_log_prob, _ = agent.choose_action(state, current_weights)
                
                
                # execute action
                new_weights, state_, next_observation, reward, done, trade_info, _ = env.step(current_weights, use_action)
                # recorder
                if reward > 0:
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
                else: #DPG
                    agent.append(state, state_, next_value, current_weights, action, action_log_prob, reward, done, trade_info['return'])
                    
                state = state_
                current_weights = new_weights
                trajectory_reward += reward
                
                if done:
                    recorder.ptfl_values.append(trade_info["portfolio_value"])
                    recorder.rewards.append(trajectory_reward)
                    
                    break
        agent.update()
                
    mean_reward = np.mean(recorder.rewards) / args.train_period_length
    
    # recorder
    agent.train_reward.append(mean_reward)
    agent.train_value.append(np.mean(recorder.ptfl_values))
    agent.train_acc.append(train_correct)
    # recorder
    
    print('=' * 120, '\nIter {}  Mean reward: {:.6f} Portfolio value: {:.4f}'
          .format(iteration, mean_reward, np.mean(recorder.ptfl_values)))
    
    return model_fn


def policy_learn(args, agent, target_stocks, path, year, Q):
    print('Start Training')
    last_use_time = 0
    start_time = time.time()
    seed = args.seed
    train_recorder = Recorder()
    val_recorder = Recorder()
    pretrain_start_date, tu_start, train_start_date, train_end_date, val_end_date = define_dates(args, year, Q)
    
    if args.pretrain:
        pretrain_history, pretrain_dating = get_history(target_stocks, args.state_length, 
                                              pretrain_start_date, train_start_date)
        ae.pretrain(pretrain_history, pretrain_dating)
        ae.save(args.pretrain_ae)
    elif not args.closeae:
        ae.load(args.pretrain_ae)
    
    
    #train_history, train_dating = get_history(target_stocks, args.state_length, 
    #                                          train_start_date, train_end_date) 
    train_history, train_dating = get_data(target_stocks, year, Q, 'train')
    if args.algo == 'DPG':
        start_idx = date_to_index(train_start_date, train_dating)
        train_start_date = index_to_date(start_idx + random.randint(0, len(train_dating) - start_idx - args.batch_size), train_dating)
    #val_history, val_dating = get_history(target_stocks, args.state_length, 
    #                                          train_end_date, val_end_date)
    val_history, val_dating = get_data(target_stocks, year, Q, 'val')
    #tu_his, _ = get_history(target_stocks, args.state_length, tu_start, train_end_date) 
    tu_his = get_data(target_stocks, year, Q, 'tu')[0]
    #'2014-01-05'
    start_idx = np.argwhere(val_dating == train_end_date)[0][0] + 1
    #benchmarks = get_history(['^GSPC', '^OEX'], args.state_length, 
    #                        train_end_date, val_end_date)[0][:, start_idx:, :] # S&P 500, S&P 100
    benchmarks = get_data(['^GSPC', '^OEX'], year, Q, 'val', bench=True)[0] # S&P 500, S&P 100
    val_recorder.benchmarks.append(benchmarks)
    quarter = str(year) + 'Q' + str(Q)
    print(quarter)
    if args.case == 3:
        path = create_q_path(path, quarter)
    write_abstract(args, path, target_stocks, train_start_date, train_end_date)
    for it in range(args.train_iter):
        agent.setup_seed_(seed)
        train_recorder.clear()
        val_recorder.clear()
        model_fn = train(args, agent, train_recorder, target_stocks, train_history, train_dating, train_start_date, it+1, path)   
        test(args, agent, val_recorder, target_stocks, val_history,  val_dating, train_end_date,
             it+1, tu_his, model_fn=model_fn, path=path)
        use_time = time.time() - start_time
        remain_time = (use_time - last_use_time) * (args.train_iter - it - 1)
        print('Time usage: {:.0f} min {:.0f} s, remain: {:.0f} min {:.0f} s'
              .format(use_time//60, use_time%60, remain_time//60, remain_time%60))
        last_use_time = use_time
        seed+=SEED_STEP

    draw_train_summary(args, agent, path)