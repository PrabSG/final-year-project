from abc import ABC, abstractclassmethod


class Shield(ABC):
    def __init__(self):
        super().__init__()

    @abstractclassmethod
    def step(self, action):
        pass
