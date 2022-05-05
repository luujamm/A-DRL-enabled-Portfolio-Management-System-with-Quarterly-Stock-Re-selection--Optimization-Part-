# -*- coding: utf-8 -*-

from train import policy_learn
from test import policy_test
from model.agent import Agent

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
    agent_name = args.algo + '_' + args.model
    
    
    # train
    if not args.test and not args.backtest: 
        path = create_path(args)
        
        year = 2020
        Q = 1
        target_stocks, action_dim = targets(year=year, Q=Q, num=20)
        
        agent = Agent(args, action_dim)
        policy_learn(args, agent, target_stocks, path, year, Q)  
    # test   
    else: 
        year = 2020
        Q = 1
        target_stocks, action_dim = targets(year=year, Q=Q, num=20)
        
        agent = Agent(args, action_dim)
        test_dir = './save_/2022-05-04/210213'
        if args.case == 3:
            test_dir += ('/' + str(year) + 'Q' + str(Q))
        policy_test(args, agent, target_stocks, test_dir, year, Q)

    
if __name__ == '__main__':
    main()