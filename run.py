# -*- coding: utf-8 -*-

import os
from src.train import policy_learn
from src.test import policy_test
from src.path import test_path
from src.model.agent import Agent
from src.utils.define_args import define_args
from src.utils.data import *
from src.utils.create_repository import create_path


def targets(year=None, Q=None, num=None):
    target_stocks = get_targets(year, Q, num)
    return target_stocks , len(target_stocks) + 1

    
def main():
    args = define_args()
    years, quaters = get_years_and_quarters()
    target_num = 20

    # train
    if not args.test and not args.backtest:  
        path = create_path(args)
        for year in years[2:]:
            for Q in quaters[:]:
                target_stocks, action_dim = targets(year=year, Q=Q, num=target_num)
                agent = Agent(args, action_dim)
                policy_learn(args, agent, target_stocks, path, year, Q)  
    # test   
    else:
        test_dir = test_path()
        testcases = []
        with open(test_dir + 'test.txt', 'r') as f:
            for line in f:
                testcases.append(line.replace('\n', ''))
        
        testfile = test_dir + 'output.txt'

        if os.path.exists(testfile):
            os.remove(testfile)
        
        for year in years:
            for Q in quaters:
                num = (year - 2018) * 4 + (Q - 1)
                args.iter = testcases[num]
                target_stocks, action_dim = targets(year=year, Q=Q, num=target_num)
                agent = Agent(args, action_dim)

                if args.case == 3:
                    test_dir_ = test_dir + str(year) + 'Q' + str(Q)

                output = policy_test(args, agent, target_stocks, test_dir_, year, Q)

                with open(testfile, 'a') as o:
                    o.write(output)


if __name__ == '__main__':
    main()