from .iris import IrisDataset
from .sklearn import SklearnDataset


class DatasetError(BaseException):
    def __init__(self, *message: object) -> None:
        self.message = ' '.join(message)
        super().__init__(self.message)

    def __str__(self) -> str:
        return repr(self.message)
