from typing import Union
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt


class DataLoader:
    pass


# noinspection SpellCheckingInspection
class Sklearn_DataLoader(DataLoader):
    """
        Dataset loader for iris, wine data provided in Scikit-learn package

        Attributes:
            lb - LabelBinarizer()
            yhot - one-hot encoded label of original data
            data - transformed data
            label - transformed binary label (0, 1)
    """

    def __init__(self, dataset: str = 'iris', labels: tuple = None) -> None:
        """
            Args:
                dataset:str='iris' - type of dataset. i.e. 'iris', 'wine'
                labels:tuple=None - labels to be considered. if None(default), select all labels in data
        """
        super().__init__()
        if dataset == 'iris':
            ds = datasets.load_iris
        elif dataset == 'wine':
            ds = datasets.load_wine
        elif dataset == 'cancer':
            ds = datasets.load_breast_cancer
        else:
            raise Exception('Not valid Data keyword')
        data, label = ds(return_X_y=True)

        if labels is not None:
            _temp = [t in labels for t in label]
            data = data[_temp]
            label = label[_temp]

        self.limit_num_data = len(label)

        self.lb = LabelBinarizer()
        self.yhot = self.lb.fit_transform(label)
        self.data = data

    def __call__(self, num_data: int, true_hot: int, shuffle: bool = True):
        """
            Args:
                num_data:int - number of training data
                true_hot:int - label to be encoded as 1
                shuffle:bool=True - to shuffle training, test dataset
            
            return:
                X, y, Xt, yt - training data, training label, test data, test label
        """
        assert num_data > 0
        self.label = self.yhot[:, true_hot].flatten()
        if num_data < self.limit_num_data:
            X, Xt, y, yt = train_test_split(self.data, self.label, train_size=num_data, shuffle=shuffle,
                                            stratify=self.label)
        else:
            X = self.data
            y = self.label
            Xt = None
            yt = None
        return X, y, Xt, yt


class Toy2DLinearLoader(DataLoader):
    def __init__(self, w: tuple, b: float) -> None:
        assert len(w) == 2
        self.w = np.array(w).reshape(-1, 1)
        self.b = b

        self.A = np.array(
            [
                [self.w[1, 0], self.w[0, 0]],
                [-self.w[0, 0], self.w[1, 0]]
            ]
        ) / (np.linalg.norm(self.w) ** 2)
        self.a = -self.b * self.w / (np.linalg.norm(self.w) ** 2)

    def __call__(self, num_data: int, noise: float = 0.0) -> np.ndarray:
        base = np.random.randn(num_data, 2)
        y = base[:, 1]
        base[:, 1] = np.where(y > 0, y + 1, y - 1)
        X = np.array([(self.A @ x.reshape(-1, 1) + self.a).reshape(-1) for x in base]) + noise * np.random.randn(
            num_data, 2)
        label = np.where(y > 0, 1, 0)
        return X, label

    def plot(self, X, y, ax=plt, c=None):
        ax.scatter(X[:, 0], X[:, 1], c=c if c is not None else y)
        xrg = np.linspace(min(X[:, 0]), max(X[:, 0]))
        yrg = -self.w[0, 0] / self.w[1, 0] * xrg - self.b / self.w[1, 0]
        ind1 = yrg <= max(X[:, 1])
        ind2 = yrg >= min(X[:, 1])
        ind = ind1 * ind2
        ax.plot(xrg[ind], yrg[ind], 'k--')


class Toy2DXORLoader(DataLoader):
    def __init__(self, *theta):
        self.points = np.pi * np.array(theta)

    def __call__(self, num_data: int, noise: float = 0.0) -> np.ndarray:
        _temp = []
        _temp_y = []
        flag = 1
        for point in self.points:
            assert isinstance(point, float)
            theta = point + noise * np.random.randn(num_data)
            _temp.append(np.vstack((np.cos(theta), np.sin(theta))).T)
            _temp_y.append(flag * np.ones(num_data))
            if flag == 1:
                flag = 0
            else:
                flag = 1
        X = np.vstack(_temp)
        y = np.concatenate(_temp_y)
        return X, y


class ToyBlochSphereLoader(DataLoader):
    def __init__(self, gap: float = np.pi / 4):
        self.gap = gap

    def __call__(self, num_data: int, noise: float = 0.0) -> np.ndarray:
        X1 = np.vstack((np.ones(num_data // 4) * np.pi / 2, np.random.uniform(self.gap, np.pi / 2, num_data // 4))).T
        X2 = np.vstack((np.ones(num_data // 4) * np.pi / 2, -np.random.uniform(self.gap, np.pi / 2, num_data // 4))).T
        X3 = np.vstack(
            (np.random.uniform(0, np.pi / 2 - self.gap, 2 * (num_data // 4)), np.zeros(2 * (num_data // 4)))).T
        y1 = np.zeros(num_data // 4)
        y2 = np.zeros(num_data // 4)
        y3 = np.ones(2 * (num_data // 4))
        X, y = (np.vstack((X1, X2, X3)), np.concatenate((y1, y2, y3)))
        return X + noise * np.random.randn(*X.shape), y


class Only2DataOnBlochSphereLoader(DataLoader):
    def __init__(self, x1, x2) -> None:
        self.x1 = x1
        self.x2 = x2

    def __call__(self):
        return np.array([self.x1, self.x2]), np.array([0, 1])


class ExampleDataLoader(DataLoader):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __call__(self):
        return self.X, self.y


class Example_4x2(ExampleDataLoader):
    def __init__(self, balanced: Union[bool, str]) -> None:
        if isinstance(balanced, str):
            balanced = True if balanced == 'balanced' else False
        if balanced:
            X = np.array([[0.72294659, -1.00386432],
                          [-0.60553577, 2.29966755],
                          [-2.50699176, -1.03101898],
                          [2.63961761, 2.21632328]])
            y = np.array([0, 0, 1, 1])
        else:
            X = np.array([[1.52122222, -2.00528202],
                          [1.7253286, 1.30938536],
                          [1.61226076, 1.12382948],
                          [1.80995737, 1.21355977]])
            y = np.array([0, 1, 1, 1])
        super().__init__(X, y)


class Example_4xn(object):
    def __init__(self, balanced: Union[bool, str], logn: int, seed=None) -> None:
        np.random.seed(seed)
        XX, yy = Example_4x2(balanced)()

        def s2c(x):
            bloch = [1, x[0], x[1]]
            r, theta, phi = bloch[0], bloch[1], bloch[2]
            bloch[0] = r * np.sin(theta) * np.cos(phi)
            bloch[1] = r * np.sin(theta) * np.sin(phi)
            bloch[2] = r * np.cos(theta)
            return np.asarray(bloch)

        def c2s(bloch, negate=False):
            r = np.linalg.norm(bloch)
            if negate:
                x = [-np.arccos(bloch[2] / r), np.arctan(bloch[1] / bloch[0])]
            else:
                x = [np.arccos(bloch[2]/r), np.arctan(bloch[1] / bloch[0])]
            return np.asarray(x)

        vecs = np.array(list(map(s2c, XX)))
        vecA = vecs[yy == 0].mean(axis=0)
        vecA = vecA / np.linalg.norm(vecA)
        vecB = vecs[yy == 1].mean(axis=0)
        vecB = vecB / np.linalg.norm(vecB)
        vecP = (vecA - vecB) / np.linalg.norm(vecA - vecB)
        vecQ = -(vecA - vecB) / np.linalg.norm(vecA - vecB)

        # B.add_annotation(vecP, text='P', color='r')
        # B.add_annotation(vecQ, text='Q', color='b')

        noise = 0.1 * np.random.randn(2 ** logn, 3)
        if balanced:
            XA = np.vstack([c2s(vecP + noise[i]) for i in range(int(0.5 * (2 ** logn)))])
            XB = np.vstack([c2s(vecQ + noise[i], True) for i in range(int(0.5 * (2 ** logn)), int(2 ** logn))])
            y = np.concatenate([np.zeros(int(0.5 * (2 ** logn))), np.ones(int(0.5 * (2 ** logn)))])
        else:
            XA = np.vstack([c2s(vecP + noise[i], True) for i in range(int(0.25 * (2 ** logn)))])
            XB = np.vstack([c2s(vecQ + noise[i]) for i in range(int(0.25 * (2 ** logn)), int(2 ** logn))])
            y = np.concatenate([np.zeros(int(0.25 * (2 ** logn))), np.ones(int(0.75 * (2 ** logn)))])
        X = np.vstack([XA, XB])
        self.X = X
        self.y = y

    def __call__(self):
        return self.X, self.y


if __name__ == '__main__':
    print(Example_4xn(True, 3)())
