# A DRL-enabled Portfolio Management System with Quarterly Stock Re-selection (Optimization Part)

## Table of Contents
1. [About](#About)
2. [Create Environment](#Create-Environment)
3. [Create Dataset](#Create-Dataset)
4. [Training](#Training)
5. [Trading Simulation](#Trading-Simulation)

## About
This is the second phase of the portfolio management system. 
With components of the portfolio selected in the previous phase, portfolio optimization is performed in this phase.

## Create Environment
Download **environment.yml** and modify the name of the environment and the prefix to the directory of conda.
Create a conda environment with **environment.yml** and activate it with the name of the environment **{env_name}**.
```shell
conda env create -f /path/to/environment.yml
conda activate {env_name}
```

## Create Dataset
```shell
python3 create_dataset.py
```
This command will create a dataset repository **/data/yyyy-mm-dd**. It will take some time to download the whole dataset and convert it to csv files.
After the download is complete, modify the repository name and modify the path in the function **get_data_repo()** in **/src/utils/data.py**.
```
repo = 'data/{repo_name}'
```

## Training
```
python3 run.py                            # default --algo=PPO --model=TCN
python3 run.py --algo=SAC
python3 run.py --algo=DDPG --model=EIIE
python3 run.py {train_args}
```
Check **/src/utils/define_args.py** to see what arguments can be defined in the command line.

Trained models will be saved in **/save_**.

## Trading Simulation
1. Modify the repository name of the trained model in **/src/path.py**.
```
path = '/path/to/repository'
```
2. Simulate for every quarter.
```
python3 out.py --test {train_args}         # same args as training
```
