
from returnn.util import better_exchook
from returnn.config import Config
from returnn.tf.network import *
from returnn.tf.layers.basic import *
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util
from returnn.tf.util.data import Dim, SpatialDim, FeatureDim, BatchInfo
from returnn.util.basic import hms, NumbersDict, BackendEngine, BehaviorVersion
import contextlib

@contextlib.contextmanager
def make_scope():
  """
  :rtype: tf.compat.v1.Session
  """
  with tf.Graph().as_default() as graph:
    with tf_compat.v1.Session(graph=graph) as session:
      yield session

def make_feed_dict(data_list, same_time=False, n_batch=3, n_time=7):
  """
  :param list[returnn.tf.util.data.Data]|ExternData data_list:
  :param bool same_time:
  :param int n_batch:
  :param int n_time:
  :rtype: dict[tf.Tensor,numpy.ndarray]
  """
  if isinstance(data_list, ExternData):
    data_list = [value for (key, value) in sorted(data_list.data.items())]
  assert n_time > 0 and n_batch > 0
  rnd = numpy.random.RandomState(42)
  existing_sizes = {}  # type: typing.Dict[tf.Tensor,int]
  d = {}
  for data in data_list:
    shape = list(data.batch_shape)
    if data.batch_dim_axis is not None:
      shape[data.batch_dim_axis] = n_batch
    for axis, dim in enumerate(shape):
      if dim is None:
        axis_wo_b = data.get_batch_axis_excluding_batch(axis)
        assert axis_wo_b in data.size_placeholder
        dyn_size = data.size_placeholder[axis_wo_b]
        if dyn_size in existing_sizes:
          shape[axis] = existing_sizes[dyn_size]
          continue
        existing_sizes[dyn_size] = n_time
        shape[axis] = n_time
        dyn_size_v = numpy.array([n_time, max(n_time - 2, 1), max(n_time - 3, 1)])
        if dyn_size_v.shape[0] > n_batch:
          dyn_size_v = dyn_size_v[:n_batch]
        elif dyn_size_v.shape[0] < n_batch:
          dyn_size_v = numpy.concatenate(
            [dyn_size_v, rnd.randint(1, n_time + 1, size=(n_batch - dyn_size_v.shape[0],))], axis=0)
        d[dyn_size] = dyn_size_v
        if not same_time:
          n_time += 1
    print("%r %r: shape %r" % (data, data.placeholder, shape))
    if data.sparse:
      d[data.placeholder] = rnd.randint(0, data.dim or 13, size=shape, dtype=data.dtype)
    else:
      d[data.placeholder] = rnd.normal(size=shape).astype(data.dtype)
  return d