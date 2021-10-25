def cf(filename):
  """Cache manager"""
  import os
  from subprocess import check_output, CalledProcessError

  if filename in eval("_cf_cache"):
    return eval("_cf_cache")[filename]
  if check_output(["hostname"]).strip().decode("utf8") in ["cluster-cn-211", "sulfid"]:
    print("use local file: %s" % filename)
    return filename  # for debugging
  try:
    cached_fn = check_output(["cf", filename]).strip().decode("utf8")
  except CalledProcessError:
    print("Cache manager: Error occured, using local file")
    return filename
  assert os.path.exists(cached_fn)
  eval("_cf_cache")[filename] = cached_fn
  return cached_fn


def get_dataset_dict(data):
  import os
  assert data in {"train", "devtrain", "cv", "dev", "hub5e_01", "rt03s"}
  epoch_split = {
    "train": eval("epoch_split")}.get(data, 1)
  corpus_name = {
    "cv": "train", "devtrain": "train"}.get(data, data)  # train, dev, hub5e_01, rt03s

  hdf_files = None
  if data in {"train", "cv", "devtrain"} and eval("_alignment"):
    hdf_files = ["%s.data-%s.hdf" % (eval("_alignment"), {"cv": "dev", "devtrain": "train"}.get(data, data))]

  files = {
    "config": eval("rasr_config"), "corpus": "/work/asr3/irie/data/switchboard/corpora/%s.corpus.gz" % corpus_name}
  if data in {"train", "cv", "devtrain"}:
    files["segments"] = "/u/schmitt/experiments/transducer/config/dependencies/seg_%s" % {
      "train": "train", "cv": "cv_head3000", "devtrain": "train_head3000"}[data]
  files["features"] = "/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.%s.bundle" % corpus_name

  for k, v in sorted(files.items()):
    assert os.path.exists(v), "%s %r does not exist" % (k, v)
  estimated_num_seqs = {
    "train": 227047, "cv": 3000, "devtrain": 3000}  # wc -l segment-file

  args = ["--config=" + files["config"],
          lambda: "--*.corpus.file=" + cf(files['corpus']),
          lambda: "--*.corpus.segments.file=" + cf(files["segments"]) if "segments" in files else "",
          lambda: "--*.feature-cache-path=" + cf(files["features"]), "--*.log-channel.file=sprint-log",
          "--*.window-size=1"]

  if not hdf_files:
    args += ["--*.corpus.segment-order-shuffle=true", "--*.segment-order-sort-by-time-length=true",
             "--*.segment-order-sort-by-time-length-chunk-size=%i" % {"train": epoch_split * 1000}.get(data, -1)]

  d = {
    "class": "ExternSprintDataset",
    "sprintTrainerExecPath": "/u/schmitt/experiments/transducer/config/sprint-executables/nn-trainer",
    "sprintConfigStr": args, "suppress_load_seqs_print": True,  # less verbose
    "input_stddev": 3.,
    "orth_vocab": eval("vocab")}

  partition_epochs_opts = {
    "partition_epoch": epoch_split,
    "estimated_num_seqs": (estimated_num_seqs[data] // epoch_split) if data in estimated_num_seqs else None, }

  if hdf_files:
    align_opts = {
      "class": "HDFDataset", "files": hdf_files, "use_cache_manager": True,
      "seq_list_filter_file": files["segments"]}  # otherwise not right selection
    align_opts.update(partition_epochs_opts)  # this dataset will control the seq list
    if data == "train":
      align_opts["seq_ordering"] = "laplace:%i" % (estimated_num_seqs[data] // 1000)
      align_opts["seq_order_seq_lens_file"] = "/u/zeyer/setups/switchboard/dataset/data/seq-lens.train.txt.gz"
    d = {
      "class": "MetaDataset", "datasets": {"sprint": d, "align": align_opts}, "data_map": {
        "data": ("sprint", "data"), # target: ("sprint", target),
        "alignment": ("align", "data"), # "align_score": ("align", "scores")
      }, "seq_order_control_dataset": "align",  # it must support get_all_tags
    }
  else:
    d.update(partition_epochs_opts)

  return d
