# QUIC-Projects

## Version Information

#### Qiskit Software	Version
- Qiskit	0.25.4
- Terra	0.17.2
- Aer	0.8.2
- Ignis	0.6.0
- Aqua	0.9.1
- IBM Q Provider	0.12.3
#### System information
- Python	3.9.2 (default, Mar 3 2021, 20:02:32) [GCC 7.3.0]
- OS	Linux
- CPUs	8
- Memory (Gb)	15.561397552490234
- Mon Nov 01 09:49:13 2021 KST

## Environment setting
Install proper packages by
```bash
$ pip install -r run_requirements.txt
```

## Notice

After setting environments, run `ibmq_device_run.ipynb`.
Please check lines with `# TODO:` for they are configuration controllers.
The experiement result will be stored at default directory `ibmq_device_run_results`.

## Config. in `ibmq_device_run.ipynb`:
- `I_HAVE_ACCESS`: Boolean. Set to `True` if have vaild IBMQ access.
- `DATA_TYPE` : Str. Either 'balanced' or 'unbalanced'
- `DEVICE` : Str. Either 'montreal' or 'toronto'
- `TEST_SIZE` : Int. Size of test dataset. Choose suitable value in terms of experiment time.
- `MAXITER` : Int. Maximum number of iteration. default =  2**10
- `LAST_AVG` : Int. Number of last samples to average. default = 2**4
- `DIR_NAME` : Str. Directory to save results. default = 'ibmq_device_run_results'
- `provider` : Your IBMQ provider.
- `layout` : Mapping between virtual and physical qubits.

## How to choose `layout`
The hardware-noise-robustness of QASVM heavily depends on `layout`.
### Decision Rule
#### Priority 1. Should be one of four

- (yi - i1 - i0 - xi - a - xj - j0 - j1 - yj) connection
- (i1 - i0 - xi - a - xj - j0 - j1) + (i0 - yi) + (j0 - yj) connection
- (yi - i0 - i1 - xi - a - xj - j1 - j0 - yj) connection
- (i0 - i1 - xi - a - xj - j1 - j0) + (i1 - yi) + (j1 - yj) connection

#### Priority 2. Should minimize TwoQubit (gate) error of possible connections in layout

#### Priority 3. Should minimize SingleQubit (gate) error of qubits in layoyt

#### Priority 4. Overall error on `*i` register should be lower than error on `*j` register

### Examples
```python
from classifiers.quantum import Qasvm_Mapping_4x2

layout = Qasvm_Mapping_4x2(backend, a=3, i0=8, i1=11, xi=5, yi=14, j0=1, j1=4, xj=2, yj=7)
layout = Qasvm_Mapping_4x2(backend, a=16, i0=11, i1=8, xi=14, yi=5, j0=22, j1=25, xj=19, yj=24)
layout = Qasvm_Mapping_4x2(backend, a=3, i0=9, i1=11, xi=5, yi=8, j0=0, j1=4, xj=2, yj=1)
layout = Qasvm_Mapping_4x2(backend, a=13, i0=10, i1=7, xi=12, yi=6, j0=11, j1=8, xj=14, yj=9)
```

## Results
If dataset is unbalanced, QASVM performs better than uniform weight STC.