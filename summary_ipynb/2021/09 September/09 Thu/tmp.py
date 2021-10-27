from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import dill
import json
import os
from multiprocessing import Pool


import sys

sys.path.extend(['/home/quic/QUIC-Projects'])
from classifiers.datasets import IrisDataset

from tqdm import tqdm
from pathlib import Path
import logging

with open('./result/trial7_ds64/setting.json', 'r') as setting_file:
    SETTING = json.load(setting_file)


def get_logger():
    _logger = logging.getLogger(__name__)
    _logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(SETTING['BASE_DIR'] + '/logging.log')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s : %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    return _logger


def main():
    logger = get_logger()
    ds = IrisDataset(feature_range=(-np.pi, np.pi), true_hot=0)
    Xt = ds.data
    yt = ds.target
    logger.info(f"Dataset size: {Xt.shape}")

    with Pool(os.cpu_count()) as pool:

        exp_dicts = []
        for si in range(SETTING['NUM_SETS']):
            for cid in SETTING['CIRCUIT_ID']:
                for l in SETTING['LAYERS']:
                    for r in range(SETTING['REPEATS']):
                        _path = Path(SETTING['BASE_DIR']) / f"Dataset #{si}/Circuit #{cid}/layer={l}/{r}/"
                        with open(_path / 'nqsvm', 'rb') as _nqsvm_file:
                            _nqsvm = dill.load(_nqsvm_file)
                        exp_dicts.append({"path": _path, "nqsvm": _nqsvm})

        def calculate_accuracy(_dict):
            path = _dict["path"]
            nqsvm = _dict['nqsvm']

            params = [(xt,) for xt in Xt]

            asyn_result = pool.map_async(nqsvm.f, params)
            logger.debug("asyn ++ 1")
            return asyn_result, path

        returns = list(map(calculate_accuracy, exp_dicts))
        for asyn_result, path in tqdm(returns, total=len(returns)):
            result = np.array(asyn_result.get())
            acc = sum(np.where(result > 0, 1, 0) == yt) / len(yt)

            with open(path / 'full_result.json', 'w') as fp:
                json.dump(dict(f=list(result), accuracy=acc), fp=fp)
            logger.info(f"got result of {path}")

    logger.info('JOBS FINISHED')


if __name__ == "__main__":
    main()
