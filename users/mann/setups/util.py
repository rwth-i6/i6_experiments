from sisyphus import delayed_ops

from collections import UserDict
from itertools import product

from i6_core.lib import lexicon

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
  
  def __repr__(self):
    return f"Selector({self.data})"

class DelayedPhonemeIndex(delayed_ops.DelayedBase):

    def __init__(self, lexicon_path, phoneme):
        super().__init__(lexicon_path)
        self.lexicon_path = lexicon_path
        self.phoneme = phoneme
    
    def get(self):
        lex = lexicon.Lexicon()
        lex.load(self.lexicon_path.get())
        return list(lex.phonemes).index(self.phoneme) + 1 # counting also "#" symbol
    
    def __sis_state__(self):
        return self.lexicon_path, self.phoneme

class DelayedPhonemeInventorySize(delayed_ops.DelayedBase):

    def __init__(self, lexicon_path):
        super().__init__(lexicon_path)
        self.lexicon_path = lexicon_path
    
    def get(self):
        lex = lexicon.Lexicon()
        lex.load(self.lexicon_path.get())
        return len(list(lex.phonemes)) + 1
    
    def __sis_state__(self):
        return self.lexicon_path


class P:
    def __init__(self, *params):
        self.params = params
    
    @staticmethod
    def maybe_tuple(p):
        if isinstance(p, tuple):
            return p
        return (p,)
    
    def __iter__(self):
        return iter(sorted(self.params))
    
    def __mul__(self, other):
        return P(*(self.maybe_tuple(p) + self.maybe_tuple(o) for p, o in product(self.params, other.params)))
    
    def __add__(self, other):
        """ Concatenate parameter lists. """
        return P(*self.params, *other.params)
