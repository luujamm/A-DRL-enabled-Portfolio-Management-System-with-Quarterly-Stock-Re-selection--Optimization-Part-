import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pickle
from .evaluation import evaluation_metrics


def draw_test_figs(args, recorder, target_stocks, test_num):
    
    stocks_label = target_stocks.copy()
    stocks_label.insert(0, 'Cash')
    action_dim = len(stocks_label)
    
    test_date = recorder.test.date
    weights = recorder.test.cal_weights(action_dim, test_num)
    final_value, ptfl_return = recorder.test.cal_returns(test_num)
    eqwt_return = recorder.ew.cal_returns(1)[1]
    benchmark_returns = recorder.benchmark.cal_benchmark_returns(ptfl_return)

    plt.figure(figsize=(6, 10))
    ax = plt.subplot(211)
    ax.plot(test_date, ptfl_return, test_date, eqwt_return)#, test_date, benchmark_returns[0], test_date, benchmark_returns[1])
    fmt_month = mdates.DayLocator(bymonthday=1)#, interval=20)
    fmt_day = mdates.DayLocator(interval=1)
    ax.xaxis.set_major_locator(fmt_month)
    ax.xaxis.set_minor_locator(fmt_day)   
    ax.legend(['RL', 'EW', 'S&P500', 'S&P100'])
    ax.set_title('Cumulative return = {:.5f}'.format(final_value))
    plt.grid()
    
    bx = plt.subplot(212)
    for w in weights:
        bx.plot(w)
    bx.legend(stocks_label)
    plt.tight_layout()
    
    return plt, test_date, final_value, ptfl_return, eqwt_return, benchmark_returns


def show_val_results(args, agent, recorder, target_stocks, test_num, iteration, model_fn, path):
    mean_reward = np.mean(recorder.test.rewards)/args.val_period_length
    result = recorder.test.cal_returns(test_num)[0]
    
    # recorder
    agent.val_reward.append(mean_reward)
    agent.val_value.append(result)
    # recorder
    if args.algo == 'SAC':
        t = 99
    else:
        t = 9

    if len(agent.val_reward) > t and mean_reward >= np.max(agent.val_reward[t:]):
        agent.save(model_fn)
        val_dir = path + '/iter{:d}_val.png'.format(iteration)
        plt = draw_test_figs(args, recorder, target_stocks, test_num)[0]
        plt.savefig(val_dir)
        plt.close()
    
    sharpe, sortino, _ = evaluation_metrics(np.array(recorder.test.daily_return), np.array(recorder.test.daily_return))
    print('Val rewards = {:.6f}, Value = {:.3f}, Diff = {:2f}, SR = {:.3f}, StR = {:.3f}, Cost = {:.3f}'
          .format(mean_reward, result, result - recorder.ew.values[-1], 
           sharpe, sortino, recorder.test.cost/test_num))


def show_test_results(args, recorder, target_stocks, test_num, iteration, test_dir):
    
    plt, test_date, final_value, ptfl_return, eqwt_return, benchmark_returns = draw_test_figs(args, recorder, target_stocks, test_num)
    if args.test:
        plt.savefig('./' + test_dir + '/test{}_iter{}.png'.format(args.case, iteration))
    else:
        plt.savefig('./' + test_dir + '/test{}_iter{}_bt.png'.format(args.case, iteration))
        
    plt.close()

    with open('./' + test_dir + '/result.pickle', 'wb') as f:
        pickle.dump(test_date, f)
        pickle.dump(ptfl_return, f)
        pickle.dump(eqwt_return, f)
        pickle.dump(benchmark_returns[0], f)
        pickle.dump(benchmark_returns[1], f)
        pickle.dump(recorder.test.daily_return, f)
        pickle.dump(recorder.ew.daily_return, f)
    
    sp500_return = benchmark_returns[0, -1]
    
    sharpe, sortino, mdd = evaluation_metrics(np.array(recorder.test.daily_return), np.array(ptfl_return))
    ew_sharpe, ew_sortino, ew_mdd = evaluation_metrics(np.array(recorder.ew.daily_return), np.array(eqwt_return))
    print('\nAverage Reward {:.5f}'.format(np.mean(recorder.test.rewards)/args.test_period_length))
    print('Average Portfolio Value {:.5f}, SR = {:.3f}, StR = {:.3f}, MDD = {:.3f}, Cost = {:.3f}'
          .format(final_value, sharpe, sortino, mdd, recorder.test.cost/test_num))
    print('EW Value: {:.5f}, Diff= {:.2f}%, SR = {:.3f}, StR = {:.3f}, MDD = {:.3f}'.format(eqwt_return[-1], (final_value - eqwt_return[-1]) * 100, ew_sharpe, ew_sortino, ew_mdd))
    print('S&P 500 : {:.5f}, Diff= {:.2f}%'.format(sp500_return, (final_value - sp500_return) * 100))
    args.test_diff.append((final_value - eqwt_return[-1]) * 100)
    output = '{:.3f} {:.3f} {:.3f} {:.3f}\n'.format(final_value, sharpe, sortino, mdd) 
    return output

    
'''def draw_test_summary(args, agent, test_dir):
    print('=' * 120 + '\nDiff mean = {:.3f}%'.format(np.mean(args.test_diff)))
    print('Diff var  = {:.3f}'.format(np.var(args.test_diff)))
    print('Best val result: iter {}, {:3f}'.format(np.argmax(agent.val_reward)+1, np.max(agent.val_reward)))
    plt.plot([i + 1 for i in range(args.train_iter)], agent.val_reward)
    plt.title('Best: iter{}, value={:3f}'.format(np.argmax(agent.agent.val_reward)+1, np.max(agent.val_reward)))
    plt.savefig('./' + test_dir +  '/val_reward.png')
    plt.close()'''


def draw_train_summary(args, agent, path):
    train_idx = np.argmax(agent.train_reward)
    val_idx = np.argmax(agent.val_reward)
    
    print('='*120,'\nTrain Result:')
    print('Best Train Iter = {:d}: Reward = {:.5f}, Value = {:.3f}'
          .format(train_idx+1, agent.train_reward[train_idx], agent.train_value[train_idx]))
    print('Best Val Iter   = {:d}: Reward = {:.5f}, Value = {:.3f}'
          .format(val_idx+1, agent.val_reward[val_idx], agent.val_value[val_idx]))
    
    x_axis = [i + 1 for i in range(args.train_iter)]
    plt.plot(x_axis, agent.train_acc, x_axis, agent.val_acc)
    plt.legend(['Train', 'Val']) 
    plt.savefig(path + '/train_val_acc.png')
    plt.close()
    plt.plot(x_axis, agent.train_reward, x_axis, agent.val_reward)
    plt.legend(['Train', 'Val']) 
    plt.savefig(path + '/train_val_reward.png')
    plt.close()