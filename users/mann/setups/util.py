from collections import UserDict

class Selector(UserDict):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  
  @classmethod
  def from_dict(cls, d):
    return cls(**d)
  
  @classmethod
  def _get_value(cls, obj, key):
    if isinstance(obj, (dict, Selector)):
      return obj.get(key, obj)
    return obj
  
  def select(self, *keys):
    res = self.data
    for key in keys:
      res = Selector(**{
        k: self._get_value(v, key) for k, v in res.items()
      })
    return res

    # return Selector(**{
    #   k: self._get_value(v, key) for k, v in self.data.items()
    # })
  
  def __repr__(self):
    return f"Selector({self.data})"

