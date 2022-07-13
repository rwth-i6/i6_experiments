""" Wrappers for some pretraining stuff. """

class WrapEpochValueWrapper:

    def __init__(self, func=None):
        self._func = func
        self._name = "Pretrain.WrapEpochValue"

    def __str__(self):
        return self._name + "({})".format(self._func.__name__)

    def __repr__(self):
        return str(self)

    def get_function(self):
        return self._func

    def set_function(self, func):
        self._func = func


def collect_function_wrappers(net_json):
  """
  See also :func:`Pretrain._resolve_wrapped_values`.
  Recursively goes through dicts, tuples and lists.
  This is a simple check to see if this is needed,
  i.e. if there are any :class:`WrapEpochValue` used.

  :param dict[str] net_json: network dict
  :return: whether there is some :class:`WrapEpochValue` in it
  :rtype: bool
  """
  assert isinstance(net_json, dict)
  # print(net_json)

  def _check(d):
    if isinstance(d, WrapEpochValueWrapper):
      print("Function found")
      return {d.get_function()}
    elif isinstance(d, dict):
      #print("True")
      res = set()
      for k, v in sorted(d.items()):
        # print(k)
        if k == 'am_scale':
          print(_check(v))
        res = res.union(_check(v))
      return res
    elif isinstance(d, (tuple, list)):
      res = set()
      for v in d:
        res = res.union(_check(v))
      return res
    else:
      return set()

  return _check(net_json)

