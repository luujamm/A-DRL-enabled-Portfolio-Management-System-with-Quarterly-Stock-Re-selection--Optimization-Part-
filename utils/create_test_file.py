import os
from path import test_path

#dir = './save_/2022-02-26/003436/'
def create_testfile():
    dir = test_path()

    years = [2018, 2019, 2020, 2021]
    quarters = [1, 2, 3, 4]
    case = 3
    testfile = dir + 'test.txt'
    if os.path.exists(testfile):
        os.remove(testfile)

    for year in years:
        for Q in quarters:
            path = dir + str(year) + 'Q' + str(Q)
            
            #find = False
            for it in range(100, 9, -1):
                model =  path + '/agent_test' + str(case) + '_iter' + str(it) + '.pth'
                if os.path.exists(model):#and not find:
                    print('Testfile add ' + model)
                    with open(dir + 'test.txt', 'a') as f:
                        f.write(str(it) + '\n')
                    break
                