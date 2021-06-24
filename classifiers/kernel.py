import numpy as np
from qiskit.circuit.library import PauliFeatureMap
from qiskit.providers.aer import StatevectorSimulator
from qiskit import execute


class _Kernel(object):
    def __init__(self) -> None:
        self.kernel = None
        self.feature_map = None

    def __call__(self, X, Y) -> np.ndarray:
        return self.kernel(X, Y)


class LinearKernel(_Kernel):
    def __init__(self) -> None:
        super().__init__()
        self.kernel = lambda X, Y: X @ Y.T
        self.feature_map = None


class Pow2Kernel(_Kernel):
    def __init__(self) -> None:
        super().__init__()
        self.kernel = lambda X, Y: np.abs(X @ Y.T) ** 2
        self.feature_map = lambda X: X


# noinspection SpellCheckingInspection
class PhaseKernel(_Kernel):
    def __init__(self) -> None:
        super().__init__()

        def Phasekernel(X, Y):
            assert len(X) == len(Y)
            N = len(X)
            cos = sum([np.cos(X[i] - Y[i]) for i in range(N)])
            # sin = sum([np.sin(X[i]-Y[i]) for i in range(N)])**2
            return cos / N  # return (cos+sin)/N/N

        def Phasemapping(X):
            N = len(X)
            return np.array([np.exp(1j * x) for x in X]) / np.sqrt(N)

        self.kernel = Phasekernel
        self.feature_map = Phasemapping


# noinspection SpellCheckingInspection
class CosineKernel(_Kernel):
    def __init__(self) -> None:
        super().__init__()

        def Cosinekernel(X, Y):
            assert len(X) == len(Y)
            N = len(X)
            cos = np.prod([np.cos(X[i] - Y[i]) for i in range(N)])
            return cos

        def Anglemapping(X):
            ret = 1
            for x in X:
                ret = np.kron(ret, [np.cos(x), np.sin(x)])
            return ret

        self.kernel = Cosinekernel
        self.feature_map = Anglemapping


# noinspection SpellCheckingInspection
class RBFKernel(_Kernel):
    def __init__(self) -> None:
        super().__init__()

        def Coherentmapping(X):
            ret = 1
            for x in X:
                nw = np.exp(-abs(x) ** 2 / 2) * np.array(
                    [(x ** n) / np.math.sqrt(np.math.factorial(n)) for n in range(1000)])
                ret = np.kron(ret, nw)
            return ret

        self.kernel = lambda X, Y: np.exp(-np.linalg.norm(X - Y) ** 2)
        self.feature_map = Coherentmapping


# noinspection SpellCheckingInspection
class PauliKernel(_Kernel):
    def __init__(self, reps: int = 2) -> None:
        super().__init__()

        def Paulimapping(X):
            N = len(X)
            qc = PauliFeatureMap(feature_dimension=N, reps=reps)
            qc = qc.assign_parameters({list(qc.parameters)[i]: X[i] for i in range(N)})
            return execute(qc, backend=StatevectorSimulator()).result().get_statevector()

        self.feature_map = Paulimapping
        self.kernel = lambda X, Y: np.abs(np.vdot(Paulimapping(X), Paulimapping(Y))) ** 2


# noinspection SpellCheckingInspection
class SingleQuibtKernel(_Kernel):
    def __init__(self) -> None:
        super().__init__()

        def Singlequbitencoding(X):
            assert len(X) == 2
            theta = X[0]
            phi = X[1]
            retX = np.array((np.cos(theta / 2) * np.exp(-1j * phi / 2), np.sin(theta / 2) * np.exp(1j * phi / 2)))
            assert retX.shape == X.shape
            return retX

        self.feature_map = Singlequbitencoding
        self.kernel = lambda X, Y: np.abs(np.vdot(self.feature_map(X), self.feature_map(Y))) ** 2


class Kernel(_Kernel):
    def __init__(self, kind: str, reps: int = 2) -> None:
        super().__init__()
        possible = ['Pow2', 'RBF', 'linear', 'Phase', 'Cosine', 'Pauli', 'SingleQubit']
        if kind not in possible:
            raise ValueError('Expect one of {:}, received {:}'.format(possible, kind))
        self.kind = kind
        if self.kind == 'RBF':
            self._kernel = RBFKernel()
        elif self.kind == 'Pow2':
            self._kernel = Pow2Kernel()
        elif self.kind == 'linear':
            self._kernel = LinearKernel()
        elif self.kind == 'Phase':
            self._kernel = PhaseKernel()
        elif self.kind == 'Cosine':
            self._kernel = CosineKernel()
        elif self.kind == 'Pauli':
            self._kernel = PauliKernel(reps)
        else:  # self.kind == 'SingleQubit':
            self._kernel = SingleQuibtKernel()

    def __call__(self, X, Y) -> np.ndarray:
        return self._kernel(X, Y)

    def feature_map(self, X) -> np.ndarray:
        return self._kernel.feature_map(X)

    def __repr__(self) -> str:
        return self.kind
