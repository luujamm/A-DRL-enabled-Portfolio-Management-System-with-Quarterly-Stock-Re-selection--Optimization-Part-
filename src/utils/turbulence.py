import numpy as np

def tu_index(observation, tu_his):
    returns = observation[1:,1:,3] / observation[1:,:-1,3] - 1
    y = returns[:, -1:]
    tu_returns = tu_his[:,1:,3] / tu_his[:,:-1,3] - 1
    u = np.mean(tu_returns, axis=1, keepdims=True)
    diff = y - u
    cov = np.cov(tu_returns)
    cov_ = np.linalg.pinv(cov)
    tu = np.matmul(diff.T, np.matmul(np.linalg.pinv(cov), diff))
    
    return tu