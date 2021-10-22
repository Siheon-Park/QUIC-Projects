#!/home/quic/anaconda3/envs/QUIC/bin/python

import sys
quic_path = '/home/quic/QUIC-Projects'
if not quic_path in sys.path:
    sys.path.append(quic_path)

from pathlib import Path
import argparse
import dill
import numpy as np
from matplotlib import pyplot as plt
from seaborn import PairGrid, scatterplot, kdeplot, histplot
from pandas import DataFrame, read_csv, read_pickle

parser = argparse.ArgumentParser(description='sample qasvm load results.')
parser.add_argument("-D", "--dir", type=str, default='models', required=False,
                    help='location to store results')
parser.add_argument("-R", "--reps", type=int, default=16,
                    help="maximum repeatition")
parser.add_argument("--plot", default=False, help="only ploting", choices=[True, False, "kde", "scatter"])

args = parser.parse_args()

def mean_square_error(regression:np.ndarray, true_val:np.ndarray):
    est = regression.mean(axis=0)
    var = regression.std(axis=0)**2
    bias2 = (est-true_val)**2
    return (bias2+var).mean()

def post_process(directory:Path, reps:int):
    qasvm_path = directory/f'qasvm(reps={reps}).pkl'
    storage_path = directory/f'storage(reps={reps}).pkl'
    reg_path = directory/f'regression(reps={reps}).npy'
    true_val_path = directory/f'true_regression.npy'

    with qasvm_path.open('rb') as f:
        qasvm = dill.load(f)
    with storage_path.open('rb') as g:
        storage = dill.load(g)
    regression = np.load(reg_path)
    true_val = np.load(true_val_path)
    mse = mean_square_error(regression, true_val)
    num_iter = storage.num_accepted()
    num_params = qasvm.num_parameters
    return mse, num_iter, num_params

def main():
    reps = len(list(Path(args.dir).glob('qasvm*.pkl')))
    data = np.array([post_process(Path(args.dir), i) for i in range(1, reps+1)])
    df = DataFrame(data, columns=['mse', 'num_iter', 'num_params'])
    g = PairGrid(df, diag_sharey=False)
    g.map_offdiag(kdeplot)
    g.map_diag(kdeplot)
    g.fig.savefig(Path(args.dir)/'result.png')

def main2():
    df = DataFrame(columns=['num_params', 'num_iter', 'mse'])
    main_dir = Path(args.dir)
    for sub_dir in main_dir.glob('*'):
        if sub_dir.is_dir():
            reps = len(list(sub_dir.glob('qasvm*.pkl')))
            for i in range(1, reps+1):
                df = df.append(dict(zip(['mse', 'num_iter', 'num_params'], post_process(sub_dir, i))), ignore_index=True)
    df.to_csv(main_dir/'result.csv', mode='w')
    df.to_pickle(main_dir/'result.pkl')
    g = PairGrid(df, diag_sharey=False)
    g.map_offdiag(kdeplot)
    g.map_diag(kdeplot)
    g.fig.savefig(main_dir/'result.png')

def main3(option):
    main_dir = Path(args.dir)
    df = read_pickle(main_dir/'result.pkl')
    g = PairGrid(df, diag_sharey=False)
    if option=="scatter":
        g.map_offdiag(scatterplot)
        g.map_diag(scatterplot)
        g.fig.savefig(main_dir/'result.png')
    else:
        g.map_offdiag(kdeplot)
        g.map_diag(kdeplot)
        g.fig.savefig(main_dir/'result.png')
if __name__=="__main__":
    if args.plot:
        main3(args.plot)
    else:
        main2()


