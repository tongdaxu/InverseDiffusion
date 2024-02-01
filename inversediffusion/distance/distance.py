from abc import ABC, abstractmethod


class DistanceInterface(ABC):
    @abstractmethod
    def forward(self, y, y_hat):
        pass
