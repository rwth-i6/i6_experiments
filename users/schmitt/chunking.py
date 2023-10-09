def chunking(dataset, seq_idx_start, **_kwargs):
  from returnn.util.basic import NumbersDict
  seq_idx = seq_idx_start
  while dataset.is_less_than_num_seqs(seq_idx):
    length = dataset.get_seq_length(seq_idx)
    t = NumbersDict.constant_like(0, numbers_dict=length)
    labels = dataset.get_data(seq_idx, "alignment")
    while length["data"] > t["data"]:
      chunk_start = NumbersDict(t)
      chunk_end = NumbersDict.min([t + chunk_size, length])
      # only yield the chunk if it contains at least one non-blank label
      if (labels[chunk_start["alignment"]:chunk_end["alignment"]] != 1030).any():
        yield seq_idx, chunk_start, chunk_end
      t += chunk_step
    seq_idx += 1


custom_chunkin_func_str = """
def chunking(dataset, seq_idx_start, **_kwargs):
    from returnn.util.basic import NumbersDict
    seq_idx = seq_idx_start
    while dataset.is_less_than_num_seqs(seq_idx):
        length = dataset.get_seq_length(seq_idx)
        t = NumbersDict.constant_like(0, numbers_dict=length)
        labels = dataset.get_data(seq_idx, "targets")
        while length["data"] > t["data"]:
            chunk_start = NumbersDict(t)
            chunk_end = NumbersDict.min([t + chunk_size, length])
            # only yield the chunk if it contains at least one non-blank label
            if (labels[chunk_start["targets"]:chunk_end["targets"]] != {blank_idx}).any():
                yield seq_idx, chunk_start, chunk_end
            t += chunk_step
        seq_idx += 1
"""
