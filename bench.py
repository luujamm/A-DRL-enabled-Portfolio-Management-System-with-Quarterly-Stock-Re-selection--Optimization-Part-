import numpy as np
from final import load_values
from utils.evaluation import evaluation_metrics







def main():
    path = 'save_/2022-04-24/120927'
    _, _, _, sp500, sp100, _ = load_values(path)
    sp500_returns = sp500[1:] / sp500[:-1] - 1
    sp100_returns = sp100[1:] / sp100[:-1] - 1
    sr5, str5, mdd5 = evaluation_metrics(sp500_returns, sp500)
    print(sp500[-1], sr5, str5, mdd5)
    sr1, str1, mdd1 = evaluation_metrics(sp100_returns, sp100)
    print(sp100[-1], sr1, str1, mdd1)


if __name__ == '__main__':
    main()