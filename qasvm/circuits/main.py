


if __name__=='__main__':
    import sys
    sys.path.append('/home/quic/QUIC-Projects')
    from qasvm.circuits.classifier import BinaryQASVM, OneLocal
    from qasvm.datasets import ToyBlochSphereLoader
    from qiskit.aqua.components.optimizers import SLSQP
    from qiskit.circuit.library import TwoLocal, EfficientSU2
    from qiskit.providers.aer import QasmSimulator
    from qiskit.aqua import QuantumInstance

    dl = ToyBlochSphereLoader()
    X, y = dl(4, 0.1)

    qi = QuantumInstance(QasmSimulator(shots=1024))
    opt = SLSQP()
    feature_map = OneLocal(1)
    var_form = EfficientSU2(2)
    dataset = {'data':X, 'label':y}
    print(feature_map)
    qasvm = BinaryQASVM(opt, feature_map, var_form, dataset, None, qi)
