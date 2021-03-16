import dill
from .datageneration import DataGeneration
import multiprocessing as mp
import numpy as np
import time
import logging
import sys

x, y = DataGeneration(2, 100000, 1, 0.1, True)
def f(i, q:mp.Queue):
    with open(f'mp_swp_id{i+1}_lr5_iter1000.dill', 'rb') as f:
        swp = dill.load(f)
    val0=i
    val1 = swp.check_performance(x, y)
    val2 = swp.check_performance_if(x, y, np.random.rand(len(swp.theta)))
    q.put((val0, val1, val2))
    

if __name__ == "__main__":
    start = time.time()
    procs = []
    q = mp.Queue()
    for i in range(1, 20):
        p = mp.Process(target=f, args=(i,q))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()
    end = time.time()
    print('multiprocess time: ', end-start)
    for i in range(1, 20):
        t = q.get()
        print(f'circuit id: {t[0]}, optimal: {t[1]}, random: {t[2]}')
'''
    start = time.time()
    for i in range(1, 20):
        f(i)
    end = time.time()
    print('single time: ', end-start)
    '''