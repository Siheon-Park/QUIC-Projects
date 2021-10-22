import sys
sys.path.extend(['/home/quic/QUIC-Projects'])

import json
import dill

import logging
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from tqdm import tqdm

with open('./result/sampling/setting.json', 'r') as f:
    ALPHA_SETTING = json.load(fp=f)
with open('./result/trial5/setting.json', 'r') as f:
    PQC_SETTING = json.load(fp=f)

def total_variance_distance(P, Q):
    return 0.5*sum(np.abs(P-Q))

def main():
    alpha_dir = Path(ALPHA_SETTING["BASE_DIR"])
    pqc_dir = Path(PQC_SETTING["BASE_DIR"])

    alpha_list = []
    for si in tqdm(range(ALPHA_SETTING["NUM_ALPHA_SAMPLE"]), desc='Alpha_Sampling', total=ALPHA_SETTING["NUM_ALPHA_SAMPLE"]):
        with open(alpha_dir / f"Dataset #{si}" / "result.json", 'r') as _f:
            _result = json.load(_f)
        _alpha = np.abs(np.array(_result['statevector_real']) + 1j * np.array(_result['statevector_imag']))**2
        alpha_list.append(_alpha)

    pqc_list = {}
    proc_bar = tqdm(desc='PQC_Sampling',
                    total=PQC_SETTING["NUM_SETS"] * len(PQC_SETTING["CIRCUIT_ID"]) * len(PQC_SETTING["LAYERS"]) *PQC_SETTING["REPEATS"])
    for si in range(PQC_SETTING["NUM_SETS"]):
        for ci in PQC_SETTING["CIRCUIT_ID"]:
            for l in PQC_SETTING["LAYERS"]:
                pqc_list[(ci, l)] = []
                for r in range(PQC_SETTING["REPEATS"]):
                    with open(pqc_dir / f"Dataset #{si}/Circuit #{ci}/layer={l}/{r}/nqsvm", 'rb') as _nqsvm_file:
                        _nqsvm = dill.load(_nqsvm_file)
                    _var_form = _nqsvm.var_form
                    _params = _nqsvm.parameters
                    _sv = Statevector(_var_form.assign_parameters(dict(zip(_var_form.parameters, _params))))
                    pqc_list[(ci, l)].append(_sv.probabilities())
                    proc_bar.update()







if __name__=="__main__":
    main()