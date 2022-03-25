import itertools
import matplotlib.pyplot as plt
import numpy as np
import csv

from environment.portfolio_new import PortfolioEnv
from utils.data import *
from utils.yahoodownloader import get_data
from utils.draw import draw_test_figs, show_test_results, show_val_results
from utils.evaluation import evaluation_metrics
from utils.recorder import Recorder

TEST_NUM = 5
TURBULENCE_THRESHOLD = 140


def test(args, agent, ae, recorder, target_stocks, test_history, 
         test_dating, sample_start_date, iteration, tu_his, test_dir=None, model_fn=None, path=None):
    agent.eval()
    if args.algo == 'PPO':
        agent.std = args.action_std_test
    action_dim = len(target_stocks) + 1
    seeds = (args.seed + i for i in range(TEST_NUM)) 
    tu_list = []
    
    test_history, test_dating, test_data = transform_data(args, test_history, test_dating)
    init_action = get_init_action(action_dim)
    
    # recorder
    test_correct = 0
    # recorder
    
    if test_dir == None:
        period_length = args.val_period_length
    else:
        period_length = args.test_period_length
    
    
    for n_episode, seed in enumerate(seeds): 
        trajectory_reward = 0
        
        agent.setup_seed_(seed)
        env = PortfolioEnv(args, test_history, test_data, action_dim, 
                           test_dating, tu_his, steps=period_length,
                           sample_start_date=sample_start_date)
        eqwt_env = PortfolioEnv(args, test_history, test_data, action_dim, 
                           test_dating, tu_his, steps=period_length, 
                           sample_start_date=sample_start_date)
        state, observation, _ = env.reset()
        eqwt_env.reset()
        
        state = transform_state(args, ae, state, observation)
        current_weights = init_action.copy()
        eqwt_action = get_init_action(action_dim, ew=True)
        trade = True
        for t in itertools.count(start=1): 
            if trade:
                use_action, action, _, _ = agent.choose_action(state, current_weights)
            else:
                use_action = init_action
                
            current_weights, state_, next_observation, reward, done, trade_info, tu = env.step(current_weights, use_action)
            eqwt_action, _, _, _, _, eqwt_trade_info, _ = eqwt_env.step(eqwt_action, eqwt_action)
            
            trajectory_reward += reward
            state_ = transform_state(args, ae, state_, next_observation)
            state = state_
            recorder.test_record(use_action, trade_info, eqwt_trade_info)
            
            # recorder
            if reward > 0:
                test_correct += 1 / period_length / TEST_NUM 
            # recorder    
            
            
            if n_episode == 0:   
                recorder.test_record_once(eqwt_trade_info)
                tu_list.append(tu[0][0])
            
            if tu[0][0] > TURBULENCE_THRESHOLD:
                trade = False
            else:
                trade = True
                
            if done:    
                break
        
        recorder.rewards.append(trajectory_reward)
        
        
    # recorder    
    agent.val_acc.append(test_correct)
    # recorder
    
    #val
    if test_dir == None: 
        show_val_results(args, agent, recorder, target_stocks, TEST_NUM, iteration, model_fn, path)
    #test   
    else:
        output = show_test_results(args, recorder, target_stocks, TEST_NUM, iteration, test_dir)
        plt.plot(tu_list)
        plt.savefig(test_dir + '/tu.png')
        plt.close()
        return output


def policy_test(args, agent, ae, target_stocks, test_dir, year=None, Q=None):
    print('Start Testing')
    test_recorder = Recorder()
    test_start_date, test_end_date = define_dates(args, year, Q)
    #test_history, test_dating = get_history(target_stocks, args.state_length, test_start_date, test_end_date) 
    test_history, test_dating = get_data(target_stocks, year, Q, 'test') 
    

    start_idx = np.argwhere(test_dating == test_start_date)[0][0] + 1
    #benchmarks = get_history(['^GSPC', '^OEX'], args.state_length, 
    #                            test_start_date, test_end_date)[0][:, start_idx:, :] # S&P 500, S&P 100
    benchmarks = get_data(['^GSPC', '^OEX'], year, Q, 'test', bench=True)[0][:, start_idx:, :]
    #tu_his, _ = get_history(target_stocks, args.state_length, '2014-01-05', test_start_date) 
    tu_his, _ = get_data(target_stocks, year, Q, 'tu')
    test_recorder.benchmarks.append(benchmarks)
    print('=' * 120, '\nStart date: ' + test_start_date)

    if args.iter == 'all':
        testcase = [i+1 for i in range(args.train_iter)]
    else:
        testcase = [int(args.iter)]

    for it in testcase:
        test_model_path = test_dir + '/agent_test{}_iter{}.pth'.format(args.case, it)
        print( 'Test model:', test_model_path)
        agent.load(test_model_path) 
        test_recorder.clear()
        output = test(args, agent, ae, test_recorder, target_stocks, test_history, 
             test_dating, test_start_date, it, tu_his, test_dir=test_dir)
        return output
    '''if args.iter == 'all':
        draw_test_summary(args, agent, test_dir) '''    