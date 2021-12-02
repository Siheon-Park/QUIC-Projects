import json
import os
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pandas import DataFrame
from tqdm import tqdm

from classifiers.quantum.ansatz import sample_circuit, PQC_Properties

with open('./run_experiment_setting.json', 'r') as f:
    SETTING = json.load(fp=f)

# CIRCUIT_ID = SETTING["CIRCUIT_ID"]
# LAYERS = SETTING["LAYERS"]
# TRAINING_SIZE = SETTING["TRAINING_SIZE"]
CIRCUIT_ID = list(range(1, 20))
LAYERS = list(range(1, 17))
TRAINING_SIZE = 64
PQC_REPEAT = 8


def pqc_props(_args):
    _circuit_id, _layer = _args
    sCircuit = sample_circuit(_circuit_id)
    var_form = sCircuit(int(np.log2(TRAINING_SIZE)), reps=_layer, )
    pqcp = PQC_Properties(var_form)
    expr = pqcp.expressibility()
    entcap = pqcp.entangling_capability()
    return _circuit_id, _layer, expr, entcap


def main():
    args = [(circuit_id, layer) for _, circuit_id, layer in product(list(range(PQC_REPEAT)), CIRCUIT_ID, LAYERS)]
    with Pool(processes=os.cpu_count()) as pool:
        result = np.array(list(tqdm(pool.imap_unordered(pqc_props, args), total=len(args), desc="PQC Sampling")))
    data = DataFrame(data=result, columns=['circuit_id', 'layer', 'expr', 'entcap'])
    data.to_csv(f'./pqc_prop_raw(trs={TRAINING_SIZE}).csv')
    temp = data.pivot_table(values=['expr', 'entcap'], index=['circuit_id', 'layer'], aggfunc='mean')
    _df = DataFrame(columns=list(temp.index.names) + list(temp.columns),
                    data=np.hstack([np.asarray(list(temp.index)), temp.to_numpy()]))
    _df.to_csv(f'./pqc_prop(trs={TRAINING_SIZE}).csv', index=False)
    temp = data.pivot_table(values=['expr', 'entcap'], index=['circuit_id', 'layer'], aggfunc='std')
    _df = DataFrame(columns=list(temp.index.names) + list(temp.columns),
                    data=np.hstack([np.asarray(list(temp.index)), temp.to_numpy()]))
    _df.to_csv(f'./pqc_prop_std(trs={TRAINING_SIZE}).csv', index=False)


if __name__ == "__main__":
    main()
