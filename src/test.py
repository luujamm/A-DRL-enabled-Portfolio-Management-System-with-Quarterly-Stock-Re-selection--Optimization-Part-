import itertools
import matplotlib.pyplot as plt
import numpy as np

from src.environment.portfolio_env import PortfolioEnv
from src.utils.data import *
from src.utils.draw import draw_test_figs, show_test_results, show_val_results
from src.utils.evaluation import evaluation_metrics
from src.utils.recorder import Recorder


TEST_NUM = 5
TURBULENCE_THRESHOLD = 140


def test(args, agent, recorder, target_stocks, test_history, test_dating, 
         sample_start_date, iteration, tu_his, test_dir=None, model_fn=None, path=None):
    agent.eval()

    if args.algo == 'PPO' or args.algo == 'SAC':
        agent.std = args.action_std_test

    action_dim = len(target_stocks) + 1
    seeds = (args.seed + i for i in range(TEST_NUM)) 
    tu_list = []
    test_history, test_dating, test_data = transform_data(args, test_history, test_dating)
    init_action = get_init_action(action_dim)
    
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
        observation, _ = env.reset()
        eqwt_env.reset()
        
        state = generate_state(observation)
        current_weights = init_action.copy()
        old_action = init_action.copy()
        eqwt_action = get_init_action(action_dim, ew=True)
        trade = True
        
        for t in itertools.count(start=1): 
            if trade:
                if args.algo == 'PPO':
                    use_action, action, _, _ = agent.choose_action(state, current_weights)

                elif args.algo == 'DDPG':
                    use_action = agent.choose_action(state, old_action, noise_inp=False)

                elif args.algo == 'SAC':
                    _, use_action, _ = agent.choose_action(state, current_weights)

            else:
                use_action = init_action
                
            current_weights, next_observation, reward, excess_ew_return, done, trade_info, tu = env.step(current_weights, use_action)
            eqwt_action, _, _, _, _, eqwt_trade_info, _ = eqwt_env.step(eqwt_action, eqwt_action)
            
            trajectory_reward += reward
            state_ = generate_state(next_observation)
            state = state_
            old_action = use_action
            recorder.test.record_trades(use_action, trade_info)
            
            if n_episode == 0:   
                recorder.ew.record_trades(eqwt_action, eqwt_trade_info)
                recorder.test.record_date(trade_info)
                tu_list.append(tu[0][0]) 
            
            if args.test and tu[0][0] > TURBULENCE_THRESHOLD:
                trade = False

            else:
                trade = True
                
            if done:    
                break
        
        recorder.test.rewards.append(trajectory_reward)

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


def policy_test(args, agent, target_stocks, test_dir, year=None, Q=None):
    print('Start Testing')
    recorder = Recorder()

    test_start_date, test_end_date = define_dates(args, year, Q)

    test_history, test_dating = get_data(target_stocks, year, Q, 'test') 
    start_idx = np.argwhere(test_dating == test_start_date)[0][0] + 1
    benchmarks = get_data(['^GSPC', '^OEX'], year, Q, 'test', bench=True)[0][:, start_idx:, :] 
    tu_his, _ = get_data(target_stocks, year, Q, 'tu')
    recorder.benchmark.values.append(benchmarks)

    print('=' * 120, '\nStart date: ' + test_start_date)

    if args.iter == 'all':
        testcase = [i+1 for i in range(args.train_iter)]

    else:
        testcase = [int(args.iter)]

    for it in testcase:
        test_model_path = test_dir + '/agent_test{}_iter{}.pth'.format(args.case, it)
        print( 'Test model:', test_model_path)
        agent.load(test_model_path) 

        output = test(args, agent, recorder, target_stocks, test_history, 
             test_dating, test_start_date, it, tu_his, test_dir=test_dir)

        return output 