from abc import ABC, abstractmethod

class ScoringMethod(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, logprobs: list[float]) -> float:
        raise NotImplementedError('Subclasses must implement this method')

    @abstractmethod
    def __str__(self):
        raise NotImplementedError('Subclasses must implement this method')
