import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# final
PPO_TCN_PATH = 'save_/2022-04-24/120927'
PPO_EIIE_PATH = 'save_/2022-04-24/231537'
DDPG_TCN_PATH = 'save_/2022-04-24/123209'
DDPG_EIIE_PATH = 'save_/2022-04-24/123311'
# turbulance
TU_140 = 'save_/2022-04-24/120927'
TU_NONE = 'save_/2022-04-24/120927_none'
# reward
R = 'save_/2022-04-24/120927'
R0 = 'save_/2022-04-24/121248'
# ew
top10 = 'data/ew/ew10.pickle'
top30 = 'data/ew/ew30.pickle'
ALL = 'data/cb5_2_0410/ew_all.pickle'

def load_values(path):

    with open('./' + path + '/result.pickle', 'rb') as f:
        ptfl = pickle.load(f)
        mv = pickle.load(f)
        ew = pickle.load(f)
        sp500 = pickle.load(f)
        sp100 = pickle.load(f)
        dates = pickle.load(f)
    
    return ptfl, mv, ew, sp500, sp100, dates

def fig_setting(ax, legend, file):
    fmt_year = mdates.AutoDateLocator()
    fmt = mdates.ConciseDateFormatter(fmt_year)
    fmt_month = mdates.DayLocator(interval=21)
    ax.xaxis.set_major_locator(fmt_year)
    ax.legend(legend)
    ax.set_ylabel('Accumulative Portfolio Value')
    ax.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./' + file + '.png')


def final():
    ppo_tcn, mv, ew, sp500, sp100, dates = load_values(PPO_TCN_PATH)
    ppo_eiie = load_values(PPO_EIIE_PATH)[0]
    ddpg_tcn = load_values(DDPG_TCN_PATH)[0]
    ddpg_eiie = load_values(DDPG_EIIE_PATH)[0]
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    ax.plot(dates, ppo_tcn, dates, ppo_eiie)
    ax.plot(dates, ddpg_tcn, dates, ddpg_eiie)
    ax.plot(dates, mv, dates, ew, dates, sp500, dates, sp100)
    legend = ['PPO+TCN', 'PPO+EIIE', 'DDPG+TCN', 'DDPG+EIIE', 'MV', 'EW', 'S&P 500', 'S&P 100']
    f = 'final'
    fig_setting(ax, legend, f)


def reward():
    r, _, _, _, _, dates = load_values(R)
    r0 = load_values(R0)[0]
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    ax.plot(dates, r, dates, r0)
    legend = ['λ=0.1', 'λ=0']
    f = 'reward'
    fig_setting(ax, legend, f)

def turbulence():
    tu_140, _, _, _, _, dates = load_values(TU_140)
    tu_none = load_values(TU_NONE)[0]
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    ax.plot(dates, tu_140, dates, tu_none)
    legend = ['with turbulence', 'without turbulence']
    f = 'tu'
    fig_setting(ax, legend, f)


def load_ew(path):
    with open('./' + path, 'rb') as f:
        ew = pickle.load(f)
    return ew


def ew():
    _, _, ew, _, sp100, dates = load_values(TU_140)
    #ew_10 = load_ew(top10)
    #ew_30 = load_ew(top30)
    ew_all = load_ew(ALL)
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    ax.plot(dates, ew, dates, ew_all, dates, sp100)
    #ax.plot(dates, ew_10, dates, ew_30)
    legend = ['Top 20', 'All stocks', 'S&P 100']#, '10', '30']
    f = 'ew'
    fig_setting(ax, legend, f)
    

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        final()
    elif len(args) == 1 and args[0] == '--tu':
        turbulence()
    elif len(args) == 1 and args[0] == '--r':
        reward()
    elif len(args) == 1 and args[0] == '--ew':
        ew()

        

    
    


if __name__ == '__main__':
    main()