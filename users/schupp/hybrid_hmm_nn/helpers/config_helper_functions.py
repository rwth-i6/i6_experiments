def all_sets_recog(inputs, only_eps, system):
  orig_name = inputs["name"]
  for _set in ["dev-other", "dev-clean", "test-other", "test-clean"]:
    if only_eps is not None:
      inputs["epochs"] = only_eps
    recog_only_s(orig_name, _set, inputs, system)

def recog_only_s(name, _set, inputs, system=None):
  assert system is not None, "No training/recognition system provided"
  inputs["name"] = name + "_" + _set
  inputs["recog_corpus_key"] = _set
  inputs["use_gpu"] = True
  system.nn_recog(**inputs)

def recog(inputs ,only_dev_other=True, only_eps=None, system=None):
  orig_name = inputs["name"]
  if only_dev_other:
    recog_only_s(orig_name, "dev-other", inputs, system)
  else:
    all_sets_recog(inputs, only_eps, system)


class experiment(object):
  def __init__(self, method):
    self._method = method
  def __call__(self, *args, **kwargs):
    return self._method(*args, **kwargs)
  @classmethod
  def get_all(cls, subject):
    def g():
      for name in dir(subject):
        method = getattr(subject, name)
        if isinstance(method, experiment):
          yield name, method
    return {name: method for name,method in g()}


