import os
import time


def create_path(args):
    if not args.trial:
        date = time.strftime('%Y-%m-%d', time.localtime())
        date_path = './save_/' + date
        if not os.path.exists(date_path):
            os.mkdir(date_path)

        clock = time.strftime('%H%M%S', time.localtime())
        clock_path = date_path + '/' + clock    
        if not os.path.exists(clock_path):
            os.mkdir(clock_path)
    else:
        clock_path = ''  # process will terminate after first iteration
    
    return clock_path

def create_q_path(path, quarter):
    q_path = path + '/' + quarter
    if not os.path.exists(q_path):
        os.mkdir(q_path)
    return q_path