from abc import ABC

class Singleton(ABC):
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise Exception("This class is a singleton and does not have an instance!")
        return cls._instance

    def __new__(self, *args, **kwargs):
        if not self._instance:
            self._instance = super(Singleton, self).__new__(self)
        return self._instance