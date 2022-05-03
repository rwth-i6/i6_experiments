import os

def get_libri_mfcc_dataset_dict(
    base_dataset_location,
    bpe_file_location,
    dataset_prefix,
    for_train = False
    ):
    # return a librispeech dataset dict, to be used with a returnn config
    # train = {'class': 'LibriSpeechCorpus', 'path': '/u/jli/setups/librispeech/2021-03-14-conformer/dataset/ogg-zips', 'use_zip': True, 'use_ogg': True, 'use_cache_manager': True, 'prefix': 'train', 'audio': {'norm_mean': '/u/jli/setups/librispeech/2021-03-14-conformer/dataset/stats.mean.txt', 'norm_std_dev': '/u/jli/setups/librispeech/2021-03-14-conformer/dataset/stats.std_dev.txt', 'window_len': 0.025, 'step_len': 0.01, 'features': 'mfcc', 'pre_process': data_augment}, 'targets': {'class': 'BytePairEncoding', 'bpe_file': "{}.codes".format(bpe_file), 'vocab_file': "{}.vocab".format(bpe_file), 'unknown_label': None, 'seq_postfix': [0]}, 'partition_epoch': 20, 'epoch_wise_filter': None, 'seq_ordering': 'laplace:281'}  
    d = {dataset_prefix : {
        'class': 'LibriSpeechCorpus', 
        'path': '%s/ogg-zips' % base_dataset_location, 
        'use_zip': True, 'use_ogg': True, 'use_cache_manager': True, 'prefix': dataset_prefix, 
        'audio': {
            'norm_mean': '%s/stats.mean.txt' % base_dataset_location, 
            'norm_std_dev': '%s/stats.std_dev.txt' % base_dataset_location, 
            'window_len': 0.025, 'step_len': 0.01, 'features': 'mfcc'}, 
        'targets': {
            'class': 'BytePairEncoding','bpe_file': "%s.codes" % bpe_file_location,
            'vocab_file': "%s.vocab" % bpe_file_location, 
            'unknown_label': None, 'seq_postfix': [0]}, 
        'fixed_random_seed': 1, 'seq_ordering': 'sorted_reverse'
        }}

    if for_train:
        d[dataset_prefix].update({
            'partition_epoch': 20, 'epoch_wise_filter': None, 'seq_ordering': 'laplace:281'
        })
    return d


_cf_cache = {}

def cf(filename):
    """Cache manager"""
    if filename in _cf_cache:
        return _cf_cache[filename]
    try:
        cached_fn = check_output(["cf", filename]).strip().decode("utf8")
    except CalledProcessError:
        print("Cache manager: Error occured, using local file")
        return filename
    assert os.path.exists(cached_fn)
    _cf_cache[filename] = cached_fn
    return cached_fn


def get_sprint_dataset(data, split_eps):
    assert data in {"train", "cv", "dev", "hub5e_01", "rt03s"}
    epoch_split = {"train": split_eps}.get(data, 1)
    corpus_name = {"cv": "train"}.get(data, data)  # train, dev, hub5e_01, rt03s

    # see /u/tuske/work/ASR/switchboard/corpus/readme
    # and zoltans mail https://mail.google.com/mail/u/0/#inbox/152891802cbb2b40
    files = {}
    files["config"] = "config/training.config"
    files["corpus"] = "/work/asr3/irie/data/switchboard/corpora/%s.corpus.gz" % corpus_name
    if data in {"train", "cv"}:
        files["segments"] = "dependencies/seg_%s" % {"train":"train", "cv":"cv_head3000"}[data]
    files["features"] = "/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.%s.bundle" % corpus_name
    for k, v in sorted(files.items()):
        assert os.path.exists(v), "%s %r does not exist" % (k, v)
    estimated_num_seqs = {"train": 227047, "cv": 3000}  # wc -l segment-file

    args = [
        "--config=" + files["config"],
        lambda: "--*.corpus.file=" + cf(files["corpus"]),
        lambda: "--*.corpus.segments.file=" + (cf(files["segments"]) if "segments" in files else ""),
        "--*.corpus.segment-order-shuffle=true",
        "--*.segment-order-sort-by-time-length=true",
        "--*.segment-order-sort-by-time-length-chunk-size=%i" % {"train": (split_eps or 1) * 1000}.get(data, -1),
        lambda: "--*.feature-cache-path=" + cf(files["features"]),
        "--*.log-channel.file=/dev/null",
        "--*.window-size=1",
    ]
    d = {
        "class": "ExternSprintDataset", "sprintTrainerExecPath": "sprint-executables/nn-trainer",
        "sprintConfigStr": args,
        "partitionEpoch": epoch_split,
        "estimated_num_seqs": (estimated_num_seqs[data] // epoch_split) if data in estimated_num_seqs else None,
    }
    d.update(sprint_interface_dataset_opts)
    return d

sprint_interface_dataset_opts = {
    "input_stddev": 3.,
    "bpe": {
        'bpe_file': '/work/asr3/irie/data/switchboard/subword_clean/ready/swbd_clean.bpe_code_1k',
        'vocab_file': '/work/asr3/irie/data/switchboard/subword_clean/ready/vocab.swbd_clean.bpe_code_1k',
        'seq_postfix': [0]
    }}