from abc import abstractmethod, ABC

class Job(ABC):

    @abstractmethod
    def start(self, *args, **kwargs):
        pass