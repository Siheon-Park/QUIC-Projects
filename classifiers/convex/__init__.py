from abc import ABCMeta
from .. import Classifier


class ConvexClassifier(Classifier, metaclass=ABCMeta):
    pass


class ConvexError(BaseException):
    def __init__(self, *message: object) -> None:
        self.message = ' '.join(message)
        super().__init__(self.message)

    def __str__(self) -> str:
        return repr(self.message)
