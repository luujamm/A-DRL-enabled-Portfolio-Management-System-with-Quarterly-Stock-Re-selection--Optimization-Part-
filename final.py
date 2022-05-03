import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# final
PPO_TCN_PATH = 'save_/2022-04-26/075400'
PPO_EIIE_PATH = 'save_/2022-04-24/231537'
DDPG_TCN_PATH = 'save_/2022-04-24/123209'
DDPG_EIIE_PATH = 'save_/2022-04-24/123311'
# turbulance
TU_140 = PPO_TCN_PATH
TU_NONE = 'save_/2022-04-26/075400_none'
# reward
R = PPO_TCN_PATH
R1 = 'save_/2022-04-24/120927'
R0 = 'save_/2022-04-24/121248'
# ew
GROUP1 = PPO_TCN_PATH
GROUP2 = 'data/ew/ew_g2.pickle'
GROUP3 = 'data/ew/ew_g3.pickle'
GROUP4 = 'data/ew/ew_g4.pickle'
GROUP5 = 'data/ew/ew_g5.pickle'
ALL = 'data/cb5_2_0410/ew_all.pickle'
# ew 2
RE = PPO_TCN_PATH
NO_RE = 'data/ew/ew_no_re.pickle'

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
    r1 = load_values(R1)[0]
    r0 = load_values(R0)[0]
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    ax.plot(dates, r, dates, r1, dates, r0)
    legend = ['λ=0.5', 'λ=0.1', 'λ=0']
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
    _, _, group1, _, sp100, dates = load_values(GROUP1)
    group2 = load_ew(GROUP2)
    group3 = load_ew(GROUP3)
    group4 = load_ew(GROUP4)
    group5 = load_ew(GROUP5)
    all_ = load_ew(ALL)
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    ax.plot(dates, group1, dates, group2, dates, group3, dates, group4, dates, group5, dates, all_, dates, sp100)
    #ax.plot(dates, ew_10, dates, ew_30)
    legend = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5', 'Full Sample', 'S&P 100']#, '10', '30']
    f = 'ew'
    fig_setting(ax, legend, f)
    print(group1[-1], group2[-1], group3[-1], group4[-1], group5[-1], all_[-1], sp100[-1])


def re():
    _, _, re, _, sp100, dates = load_values(RE)
    no_re = load_ew(NO_RE)
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    ax.plot(dates, re, dates, no_re, dates, sp100)
    #ax.plot(dates, ew_10, dates, ew_30)
    legend = ['With Re-selection', 'Without Re-selection', 'S&P 100']
    f = 're'
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
    elif len(args) == 1 and args[0] == '--re':
        re()

        

    
    


if __name__ == '__main__':
    main()