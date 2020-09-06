import math
from qiskit import QuantumCircuit


# m = logM copy of datasets


def weighting(qc, qr, alpha=math.pi / 2, barrier=True):
    qc.ry(alpha, qr).inverse()
    if barrier:
        qc.barrier()


def training(qc, q_m, q_x, q_y, barrier=True):
    qc.h(q_x)
    qc.rz(math.pi, q_x).inverse()
    qc.s(q_x)
    qc.cz(q_m, q_x)
    qc.cx(q_m, q_y)
    if barrier:
        qc.barrier()


def test_encoding(qc, q_test, theta=math.pi / 2, barrier=True):
    qc.rx(theta, q_test)
    if barrier:
        qc.barrier()


def swap_classifier(qc, q_a, q_x, q_test, q_y=None, **kwargs):
    qc.h(q_a)
    qc.cswap(q_a, q_x, q_test)
    qc.h(q_a)
    cbits = kwargs.get('measure')
    if cbits is not None:
        qc.measure([q_a, q_y], cbits)


def weighting_gate(theta=math.pi / 2):
    qc = QuantumCircuit(1, name='weight')
    weighting(qc, 0, theta, barrier=False)
    return qc.to_instruction()


def training_gate():
    qc = QuantumCircuit(3, name='train_data')
    training(qc, 0, 1, 2, barrier=False)
    return qc.to_instruction()


def test_encoding_gate(theta=math.pi / 2):
    qc = QuantumCircuit(1, name='test_data')
    test_encoding(qc, 0, theta, barrier=False)
    return qc.to_instruction()


def swap_classifier_gate():
    qc = QuantumCircuit(3, name='SWAP_classifier')
    swap_classifier(qc, 0, 1, 2, barrier=False)
    return qc.to_instruction()


def swap_test_postprocess(counts):
    return [correlation(count_dict) for count_dict in counts]


def correlation(count_dict, op='ZZ'):
    if op == 'ZZ':
        c00 = count_dict.get('00', 0)
        c01 = count_dict.get('01', 0)
        c10 = count_dict.get('10', 0)
        c11 = count_dict.get('11', 0)
        return (c00 + c11 - c01 - c10) / sum(count_dict.values())
    else:
        pass
