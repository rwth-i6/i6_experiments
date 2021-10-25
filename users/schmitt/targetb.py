

def targetb_linear(source, **kwargs):
  from TFUtil import get_rnnt_linear_aligned_output
  enc = source(1, as_data=True, auto_convert=False)
  dec = source(0, as_data=True, auto_convert=False)
  enc_lens = enc.get_sequence_lengths()
  dec_lens = dec.get_sequence_lengths()
  out, out_lens = get_rnnt_linear_aligned_output(
    input_lens=enc_lens,
    target_lens=dec_lens, targets=dec.get_placeholder_as_batch_major(),
    blank_label_idx=eval("targetb_blank_idx"),
    targets_consume_time=True)
  return out


def targetb_linear_out(sources, **kwargs):
  from TFUtil import Data
  enc = sources[1].output
  dec = sources[0].output
  size = enc.get_sequence_lengths()  # + dec.get_sequence_lengths()
  # output_len_tag.set_tag_on_size_tensor(size)
  return Data(name="targetb_linear", sparse=True, dim=eval("targetb_num_labels"), size_placeholder={0: size})


def targetb_search_or_fallback(source, **kwargs):
  import tensorflow as tf
  from TFUtil import where_bc
  ts_linear = source(0)  # (B,T)
  ts_search = source(1)  # (B,T)
  l = source(2, auto_convert=False)  # (B,)
  return where_bc(tf.less(l[:, None], 0.01), ts_search, ts_linear)