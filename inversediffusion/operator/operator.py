from abc import ABC, abstractmethod


class OperatorInterface(ABC):
    @abstractmethod
    def init(self, x):
        pass

    @abstractmethod
    def forward(self, x):
        pass
