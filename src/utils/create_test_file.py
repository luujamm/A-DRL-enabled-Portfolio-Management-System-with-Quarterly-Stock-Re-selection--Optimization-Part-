import os
from src.path import test_path
from .data import get_years_and_quarters


def create_testfile():
    test_dir = test_path()
    years, quarters = get_years_and_quarters()
    case = 3
    testfile = test_dir + 'test.txt'

    if os.path.exists(testfile):
        os.remove(testfile)

    for year in years:
        for Q in quarters:
            path = test_dir + str(year) + 'Q' + str(Q)

            for it in range(1000, 9, -1):
                model =  path + '/agent_test' + str(case) + '_iter' + str(it) + '.pth'

                if os.path.exists(model):
                    print('Testfile add ' + model)

                    with open(testfile, 'a') as f:
                        f.write(str(it) + '\n')
                        
                    break