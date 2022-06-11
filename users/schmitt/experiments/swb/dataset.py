from sisyphus.delayed_ops import DelayedFormat

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


def get_dataset_dict_wo_alignment(
        data, rasr_config_path, rasr_nn_trainer_exe, vocab, epoch_split=6, concat_seqs=False, concat_seq_tags=None,
        concat_seq_lens=None):
  assert data in {"train", "devtrain", "cv", "dev", "hub5e_01", "rt03s"}
  assert not concat_seqs or (concat_seq_tags is not None and concat_seq_lens is not None)
  epoch_split = {
    "train": epoch_split}.get(data, 1)

  estimated_num_seqs = {
    "train": 227047, "cv": 3000, "devtrain": 3000}  # wc -l segment-file

  args = [DelayedFormat("--config={}", rasr_config_path),
          "--*.corpus.segment-order-shuffle=true" if not concat_seqs else "--*.corpus.segment-order-shuffle=false",
          "--*.segment-order-sort-by-time-length=true" if not concat_seqs else "--*.segment-order-sort-by-time-length=false",
          "--*.segment-order-sort-by-time-length-chunk-size=%i" % {"train": epoch_split * 1000}.get(data, -1)]

  d = {
    "class": "ExternSprintDataset",
    "sprintTrainerExecPath": rasr_nn_trainer_exe,
    "sprintConfigStr": args, "suppress_load_seqs_print": True,  # less verbose
    "input_stddev": 3.}

  if vocab is not None:
    if "bpe_file" in vocab:
      d["bpe"] = vocab
    else:
      d["orth_vocab"] = vocab

  if data == "train":
    partition_epochs_opts = {
      "partition_epoch": epoch_split,
      "estimated_num_seqs": (estimated_num_seqs[data] // epoch_split) if data in estimated_num_seqs else None, }

    d.update(partition_epochs_opts)

  if concat_seqs:
    d = {
      "class": "ConcatSeqsDataset", "seq_list_file": concat_seq_tags, "seq_len_file": concat_seq_lens,
      "dataset": d, "seq_ordering": "sorted_reverse",
    }

  return d


def get_dataset_dict_w_alignment(data, rasr_config_path, rasr_nn_trainer_exe, segment_file, alignment, epoch_split=6):
  hdf_files = [alignment]

  d = {
    "class": "ExternSprintDataset",
    "sprintTrainerExecPath": rasr_nn_trainer_exe,
    "sprintConfigStr": DelayedFormat("--config={}", rasr_config_path),
    "suppress_load_seqs_print": True,  # less verbose
    "input_stddev": 3.}

  align_opts = {
    "class": "HDFDataset", "files": hdf_files, "use_cache_manager": True,
    "seq_list_filter_file": segment_file}  # otherwise not right selection
  if data == "train":
    estimated_num_seqs = 227047
    align_opts["partition_epoch"] = epoch_split
    align_opts["estimated_num_seqs"] = (estimated_num_seqs // epoch_split)
    align_opts["seq_ordering"] = "laplace:%i" % (estimated_num_seqs // 1000)
    align_opts["seq_order_seq_lens_file"] = "/u/zeyer/setups/switchboard/dataset/data/seq-lens.train.txt.gz"
  d = {
    "class": "MetaDataset", "datasets": {"sprint": d, "align": align_opts}, "data_map": {
      "data": ("sprint", "data"),
      "alignment": ("align", "data"),
    }, "seq_order_control_dataset": "align",
  }

  return d


def get_dataset_dict_w_labels(data, rasr_config_path, rasr_nn_trainer_exe, segment_file, label_hdf, label_name, epoch_split=6):
  hdf_files = [label_hdf]

  d = {
    "class": "ExternSprintDataset",
    "sprintTrainerExecPath": rasr_nn_trainer_exe,
    "sprintConfigStr": DelayedFormat("--config={}", rasr_config_path),
    "suppress_load_seqs_print": True,  # less verbose
    "input_stddev": 3.}

  label_opts = {
    "class": "HDFDataset", "files": hdf_files, "use_cache_manager": True,
    "seq_list_filter_file": segment_file}  # otherwise not right selection
  if data == "train":
    estimated_num_seqs = 227047
    label_opts["partition_epoch"] = epoch_split
    label_opts["estimated_num_seqs"] = (estimated_num_seqs // epoch_split)
    label_opts["seq_ordering"] = "laplace:%i" % (estimated_num_seqs // 1000)
    label_opts["seq_order_seq_lens_file"] = "/u/zeyer/setups/switchboard/dataset/data/seq-lens.train.txt.gz"
  d = {
    "class": "MetaDataset", "datasets": {"sprint": d, "hdf": label_opts}, "data_map": {
      "data": ("sprint", "data"),
      label_name: ("hdf", "data"),
    }, "seq_order_control_dataset": "hdf",
  }

  return d


def get_phoneme_dataset():
  dataset_dict = {
    'class': 'ExternSprintDataset',
    'sprintConfigStr': '--config=/u/schmitt/experiments/transducer/config/rasr-configs/zhou-phon-trans.config --*.LOGFILE=nn-trainer.train.log --*.TASK=1 --*.corpus.segment-order-shuffle=true',
    'sprintTrainerExecPath': '/u/zhou/rasr-dev/arch/linux-x86_64-standard-label_sync_decoding/nn-trainer.linux-x86_64-standard-label_sync_decoding'}

  return dataset_dict
