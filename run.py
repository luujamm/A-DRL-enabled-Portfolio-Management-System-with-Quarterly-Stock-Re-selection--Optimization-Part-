# -*- coding: utf-8 -*-

import os
from train import policy_learn
from test import policy_test
from model.PPOagent import Agent
from model.autoencoder import Autoencoder
from utils.define_args import define_args
from utils.data import get_targets
from utils.create_repository import create_path


def targets(year=None, Q=None, num=None):
    if year == None:
        target_stocks = ['AAPL', 'ADBE', 'AMZN', 'BA', 'BAC', 'CSCO', 'DIS', 'HD', 'HSBC', 'IBM', 'INTC', 
                         'JNJ', 'KO', 'MRK', 'MSFT', 'NVDA', 'NVO', 'NVS', 'ORCL',  'RY', 'TD', 'VZ', 'WMT']   # DeepBreath target stocks
        #target_stocks = ['AAPL', 'AXP', 'BA', 'CAT', 'CVX', 'DD','DIS', 'HD', 'IBM', 'INTC', 'JNJ', 'KO',
        #                 'MCD', 'MMM', 'MRK', 'NKE', 'PFE', 'PG', 'RTX', 'UNH',  'VZ', 'WBA', 'WMT', 'XOM'] # SSRN target

        #target_stocks = ['AXP', 'CAT', 'CVX', 'DD',
        #                 'MCD', 'MRK',  'PFE', 'PG', 'RTX', 'UNH','WBA', 'XOM'] # SSRN target

        #target_stocks = ['AAPL', 'ADBE', 'AMZN', 'BA', 'BAC', 'CSCO', 'DIS', 'HD', 'HSBC'] # Model test target stocks
    else:
        target_stocks = get_targets(year, Q, num)
    
    return target_stocks , len(target_stocks) + 1

    
def main():
    args = define_args()
    years = [2018, 2019, 2020, 2021] 
    quaters = [1, 2, 3, 4]
    agent_name = args.algo + '_' + args.model
    
    ae = Autoencoder(args)  
    
    # train
    if not args.test and not args.backtest:  
        path = create_path(args)
        for year in years[:]:
            for Q in quaters[:]:
                target_stocks, action_dim = targets(year=year, Q=Q, num=20)
                agent = Agent(args, action_dim, agent_name)
                policy_learn(args, agent, ae, target_stocks, path, year, Q)  
    # test   
    else:
        test_dir = './save_/2022-02-25/013418/'
        testcases = []
        with open(test_dir+'test.txt', 'r') as f:
            for line in f:
                testcases.append(line.replace('\n', ''))
        
        testfile = test_dir + 'output.txt'
        if os.path.exists(testfile):
            os.remove(testfile)
        
        for year in years:
            for Q in quaters:
                num = (year - 2018) * 4 + (Q - 1)
                args.iter = testcases[num]
                target_stocks, action_dim = targets(year=year, Q=Q, num=20)
                agent = Agent(args, action_dim, agent_name)

                if args.case == 3:
                    test_dir_ = test_dir + str(year) + 'Q' + str(Q)

                output = policy_test(args, agent, ae, target_stocks, test_dir_, year, Q)

                with open(testfile, 'a') as o:
                    o.write(output)

if __name__ == '__main__':
    main()