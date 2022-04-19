import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


PPO_TCN_PATH = 'save_/2022-04-12/124043'
PPO_EIIE_PATH = 'save_/2022-04-15/165621'

def load_values(path):

    with open('./' + path + '/result.pickle', 'rb') as f:
        ptfl = pickle.load(f)
        mv = pickle.load(f)
        ew = pickle.load(f)
        sp500 = pickle.load(f)
        sp100 = pickle.load(f)
        dates = pickle.load(f)
    
    return ptfl, mv, ew, sp500, sp100, dates


def main():
    ppo_tcn, mv, ew, sp500, sp100, dates = load_values(PPO_TCN_PATH)
    ppo_eiie = load_values(PPO_EIIE_PATH)[0]

    plt.figure(figsize=(8, 6))
    ax = plt.subplot()

    ax.plot(dates, ppo_tcn, dates, ppo_eiie)
    
    ax.plot(dates, mv, dates, ew, dates, sp500, dates, sp100)
    fmt_year = mdates.AutoDateLocator()
    fmt = mdates.ConciseDateFormatter(fmt_year)
    fmt_month = mdates.DayLocator(interval=21)
    ax.xaxis.set_major_locator(fmt_year)
    ax.legend(['PPO + TCN', 'PPO + EIIE', 'MV', 'EW', 'S&P 500', 'S&P 100'])
    ax.set_ylabel('Cumulative Return')
    ax.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./final.png')


if __name__ == '__main__':
    main()