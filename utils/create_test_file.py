import os
from path import test_path
from data import get_years_and_quarters

#dir = './save_/2022-02-26/003436/'
def create_testfile():
    test_dir = test_path()

    years, quaters = get_years_and_quarters()
    case = 3
    testfile = test_dir + 'test.txt'
    if os.path.exists(testfile):
        os.remove(testfile)

    for year in years:
        for Q in quarters:
            path = test_dir + str(year) + 'Q' + str(Q)
            
            #find = False
            for it in range(100, 9, -1):
                model =  path + '/agent_test' + str(case) + '_iter' + str(it) + '.pth'
                if os.path.exists(model):#and not find:
                    print('Testfile add ' + model)
                    with open(test_dir + 'test.txt', 'a') as f:
                        f.write(str(it) + '\n')
                    break
                