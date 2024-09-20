Serialization is commonly used/needed for RETURNN configs.

We have various options now:

- :mod:`i6_experiments.common.setups.serialization` /
  :mod:`i6_core.serialization` (copy of :mod:`i6_experiments.common.setups.serialization`):

  There is a :class:`SerializerObject` base class, deriving from Sisyphus :class:`DelayedBase`.
  There is also a :class:`Collection` to collect multiple such objects.
  The most commonly used serializer object is maybe the :class:`Import`,
  which imports some function from some module,
  also making sure that the path of the module is added to ``sys.path``,
  and properly setting a hash based on the module name,
  potentially also using the ``unhashed_package_root`` logic.

  For every serializer object, you can choose whether it is part of the hash or not.
  And if it is part of the hash, how the hash is defined exactly
  (``unhashed_package_root``, ``ignore_import_as_for_hash``, etc.).

  Those objects are intended to be put as the ``python_epilog`` in a :class:`ReturnnConfig`.

  As an example, see :func:`i6_experiments.users.zeyer.train_v3.train`,
  :func:`i6_experiments.users.zeyer.recog.search_dataset`,
  :func:`i6_experiments.users.zeyer.forward_to_hdf.forward_to_hdf`.

- :class:`returnn_common.nn.ReturnnConfigSerializer`

  Used to serialize dim tags (:class:`Dim`)
  (and more, also the RETURNN-common ``nn.Module`` instances,
  transforming those into a RETURNN TF net dict, and handling dim tag refs properly;
  but we only use it for the dim tag serialization now).
  Specifically, we mostly just use :func:`ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct`.
  This function uses :class:`returnn_common.nn.ReturnnDimTagsProxy` internally.

  As an example, see :func:`i6_experiments.users.zeyer.train_v3.train`,
  :func:`i6_experiments.users.zeyer.recog.search_dataset`,
  :func:`i6_experiments.users.zeyer.forward_to_hdf.forward_to_hdf`.

- :mod:`i6_experiments.common.setups.returnn.serialization.get_serializable_config`

  Operates on an existing :class:`ReturnnConfig` instance,
  going through all the config entries, checking whether they can be serialized directly,
  and if not, moving them to the ``python_epilog``.
  This handles dim tags (:class:`Dim`) directly using :class:`returnn_common.nn.ReturnnDimTagsProxy`
  and functions (:class:`FunctionType`) by copying the function source code
  (just as :class:`ReturnnConfig` also does).
  Functions are wrapped via :class:`CodeFromFunction`,
  and hashing can be controlled via ``hash_full_python_code``.

  As an example, see :func:`i6_experiments.users.zeyer.train_v3.train`,
  :func:`i6_experiments.users.zeyer.recog.search_dataset`,
  :func:`i6_experiments.users.zeyer.forward_to_hdf.forward_to_hdf`.

- :class:`ReturnnConfig` itself, e.g. ``python_epilog``:

  There is no special logic for the ``config`` or ``post_config``.
  It basically uses ``repr``.
  So that will not directly work for any special objects (dim tags, functions, etc).

  However, for ``python_epilog`` (also ``python_prolog``),
  it accepts :class:`DelayedBase`, and thus any custom logic can be done this way
  (see :class:`SerializerObject` or :func:`get_serializable_config` above).
  Additionally, when it finds a function (:class:`FunctionType`) or class,
  it will copy the function/class source code.

  Regarding hashing, ``config`` is used as-is, by default (the way we normally use it),
  also ``python_epilog`` is used as-is.
  Most of the :class:`SerializerObject` define a custom Sisyphus hash.
  When a function/class is directly used in ``python_epilog`` (not via :class:`SerializerObject`),
  it uses the hash of the function/class directly.
  The hash of a function/class is defined via ``(obj.__module__, obj.__qualname__)``.
