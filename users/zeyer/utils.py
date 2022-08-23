
def generic_open(filename, mode="r"):
  """
  Wrapper around :func:`open`.
  Automatically wraps :func:`gzip.open` if filename ends with ``".gz"``.

  :param str filename:
  :param str mode: text mode by default
  :rtype: typing.TextIO|typing.BinaryIO
  """
  if filename.endswith(".gz"):
    import gzip
    if "b" not in mode:
      mode += "t"
    return gzip.open(filename, mode)
  return open(filename, mode)

