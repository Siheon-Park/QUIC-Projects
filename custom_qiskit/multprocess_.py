# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import logging
from qiskit.aqua.components.optimizers.spsa import logger
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)


# %%
from .swap_classifier import pseudo_SWAP_classifier, SVM
from .datageneration import DataGeneration, iris_data
import numpy as np
import dill
import os
import time
import multiprocessing as mp

x, y = iris_data()


# %%
def f(i):
    print('process id: ',os.getpid(), 'circuit id: ', i)
    swp = pseudo_SWAP_classifier(x, y, i, 5)
    svm = SVM(x, y, 1000)
    initial1 = np.random.rand(len(swp.theta))
    initial2 = np.random.rand(svm.num_data)
    swp.optimize(initial1, maxiter=2**10, disp=True)
    svm.optimize(initial2, maxiter=2**10, disp=True)
    with open(f'/home/quic/QUIC-Projects/multiprocess/peseudoSVM_id{i}.dill', 'wb') as f:
        dill.dump(swp, f)
    with open(f'/home/quic/QUIC-Projects/multiprocess/SVM_id{i}.dill', 'wb') as f:
        dill.dump(swp, f)


# %%
if __name__ == "__main__":
    '''
    start = time.time()
    for i in range(1, 2):
        f(i)
    end = time.time()
    print('for loop:', end-start)
    
'''
    start = time.time()
    with mp.Pool(processes=16) as pool:
        pool.map(f, range(1, 20))
    end = time.time()

    print('multiprocess:', end-start)

    
        


# %%


    
# %%
