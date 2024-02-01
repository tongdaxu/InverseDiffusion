from abc import ABC, abstractmethod


class PipelineInterface(ABC):
    @abstractmethod
    def __call__(self, model, scheduler, operator, y, distance, **kwargs):
        pass
