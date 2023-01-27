from abc import ABC


class AbsModule(ABC):

  def __init__(self):
    self.name = None

  def create(self):
    raise NotImplementedError
