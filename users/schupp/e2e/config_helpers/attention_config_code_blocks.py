import textwrap

LOCAL_FUSION_CODE = textwrap.dedent(
    """
def local_fusion(source, **kwargs):
    # see /u/zeyer/setups/switchboard/2020-06-09--e2e-multi-gpu/localfusion.conv2l.specaug4a.wdrop03.adrop01.l2a_0001.ctc.devtrain.sgd.lr1e_3.lrwa.lrt_0005.mgpu4.htd100
    import tensorflow as tf
    from TFUtil import safe_log
    import TFCompat
    am_score = source(0)
    lm_score = source(1)
    am_data = source(0, as_data=True)
    out = ext_am_scale * safe_log(am_score) + ext_lm_scale * safe_log(lm_score)
    return out - TFCompat.v1.reduce_logsumexp(out, axis=am_data.feature_dim_axis, keepdims=True)
    """
)

TRANSFORM_CODE = textwrap.dedent( # code from albertz
    """\
def transform(data, network, time_factor=1):
    x = data.placeholder
    import tensorflow as tf
    # summary("features", x)
    step = network.global_train_step
    step1 = tf.compat.v1.where(tf.greater_equal(step, 1000), 1, 0)
    step2 = tf.compat.v1.where(tf.greater_equal(step, 2000), 1, 0)
    def get_masked():
        x_masked = x
        x_masked = random_mask(
          x_masked, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
          min_num=step1 + step2, max_num=tf.maximum(tf.shape(input=x)[data.time_dim_axis] // 100, 2) * (1 + step1 + step2 * 2),
          max_dims=20 // time_factor)
        x_masked = random_mask(
          x_masked, batch_axis=data.batch_dim_axis, axis=data.feature_dim_axis,
          min_num=step1 + step2, max_num=2 + step1 + step2 * 2,
          max_dims=data.dim // 5)
        #summary("features_mask", x_masked)
        return x_masked
    x = network.cond_on_train(get_masked, lambda: x)
    return x
    """
)

_MASK_CODE = textwrap.dedent( # code from albertz
    """
def _mask(x, batch_axis, axis, pos, max_amount):
    import tensorflow as tf
    ndim = x.get_shape().ndims
    n_batch = tf.shape(input=x)[batch_axis]
    dim = tf.shape(input=x)[axis]
    amount = tf.random.uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
    pos2 = tf.minimum(pos + amount, dim)
    idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
    pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
    pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
    cond = tf.logical_and(tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc))  # (batch,dim)
    if batch_axis > axis:
        cond = tf.transpose(a=cond)  # (dim,batch)
    cond = tf.reshape(cond, [tf.shape(input=x)[i] if i in (batch_axis, axis) else 1 for i in range(ndim)])
    from TFUtil import where_bc
    x = where_bc(cond, 0.0, x)
    return x
    """
)

RANDOM_MASK_CODE = textwrap.dedent(
    """
def random_mask(x, batch_axis, axis, min_num, max_num, max_dims):
    import tensorflow as tf
    n_batch = tf.shape(input=x)[batch_axis]
    if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
        num = min_num
    else:
        num = tf.random.uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    z = -tf.math.log(-tf.math.log(tf.random.uniform((n_batch, tf.shape(input=x)[axis]), 0, 1)))
    _, indices = tf.nn.top_k(z, num if isinstance(num, int) else tf.reduce_max(input_tensor=num))
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
    # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])
    if isinstance(num, int):
        for i in range(num):
            x = _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims)
    else:
        _, x = tf.while_loop(
            cond=lambda i, _: tf.less(i, tf.reduce_max(input_tensor=num)),
            body=lambda i, x: (
                i + 1,
                tf.compat.v1.where(
                    tf.less(i, num),
                    _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims),
                    x)),
            loop_vars=(0, x))
    return x    
    """
)

GET_SPRINT_DATASET_CODE = textwrap.dedent( # Code from the old pipeline
    """
import os
def get_sprint_dataset(data):
    assert data in {"train", "cv", "dev", "hub5e_01", "rt03s"}
    epoch_split = {"train": EpochSplit}.get(data, 1)
    corpus_name = {"cv": "train"}.get(data, data)  # train, dev, hub5e_01, rt03s

    # see /u/tuske/work/ASR/switchboard/corpus/readme
    # and zoltans mail https://mail.google.com/mail/u/0/#inbox/152891802cbb2b40
    files = {}
    files["config"] = "/u/schupp/setups/dynamic_sis/datasets/swb/old_pipeline/config/training.config"
    files["corpus"] = "/work/asr3/irie/data/switchboard/corpora/%s.corpus.gz" % corpus_name
    if data in {"train", "cv"}:
        files["segments"] = "/u/schupp/setups/dynamic_sis/datasets/swb/old_pipeline/dependencies/seg_%s" % {"train":"train", "cv":"cv_head3000"}[data]
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
        "--*.segment-order-sort-by-time-length-chunk-size=%i" % {"train": (EpochSplit or 1) * 1000}.get(data, -1),
        lambda: "--*.feature-cache-path=" + cf(files["features"]),
        "--*.log-channel.file=/dev/null",
        "--*.window-size=1",
    ]
    d = {
        "class": "ExternSprintDataset", "sprintTrainerExecPath": "/u/schupp/setups/dynamic_sis/datasets/swb/old_pipeline/sprint-executables/nn-trainer",
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
    """
)

CACHE_MANAGER_CODE = textwrap.dedent(
"""
from subprocess import check_output, CalledProcessError
_cf_cache = {}

def cf(filename):
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
"""
)

CUSTOM_CONSTRUCTION_ALGO_CODE = textwrap.dedent(
"""
def custom_construction_algo(idx, net_dict):
    # pool = True
    # For debugging, use: python3 ./crnn/Pretrain.py config... Maybe set repetitions=1 below.
    StartNumLayers = 2
    InitialDimFactor = 0.5
    orig_num_lstm_layers = 0
    while "lstm%i_fw" % orig_num_lstm_layers in net_dict:
        orig_num_lstm_layers += 1
    assert orig_num_lstm_layers >= 2
    orig_red_factor = 1
    if pool:
        for i in range(orig_num_lstm_layers - 1):
            orig_red_factor *= net_dict["lstm%i_pool" % i]["pool_size"][0]
    net_dict["#config"] = {}
    if idx < 4:
        #net_dict["#config"]["batch_size"] = 15000
        net_dict["#config"]["accum_grad_multiple_step"] = 4
    idx = max(idx - 4, 0)  # repeat first
    num_lstm_layers = idx + StartNumLayers  # idx starts at 0. start with N layers
    if num_lstm_layers > orig_num_lstm_layers:
        # Finish. This will also use label-smoothing then.
        return None
    if pool:
        if num_lstm_layers == 2:
            net_dict["lstm0_pool"]["pool_size"] = (orig_red_factor,)
    # Skip to num layers.
    net_dict["encoder"]["from"] = ["lstm%i_fw" % (num_lstm_layers - 1), "lstm%i_bw" % (num_lstm_layers - 1)]
    # Delete non-used lstm layers. This is not explicitly necessary but maybe nicer.
    for i in range(num_lstm_layers, orig_num_lstm_layers):
        del net_dict["lstm%i_fw" % i]
        del net_dict["lstm%i_bw" % i]
        if pool:
            del net_dict["lstm%i_pool" % (i - 1)]
    # Thus we have layers 0 .. (num_lstm_layers - 1).
    layer_idxs = list(range(0, num_lstm_layers))
    layers = ["lstm%i_fw" % i for i in layer_idxs] + ["lstm%i_bw" % i for i in layer_idxs]
    grow_frac = 1.0 - float(orig_num_lstm_layers - num_lstm_layers) / (orig_num_lstm_layers - StartNumLayers)
    dim_frac = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac
    for layer in layers:
        net_dict[layer]["n_out"] = int(net_dict[layer]["n_out"] * dim_frac)
        if "dropout" in net_dict[layer]:
            net_dict[layer]["dropout"] *= dim_frac
    net_dict["enc_value"]["dims"] = (AttNumHeads, int(EncValuePerHeadDim * dim_frac * 0.5) * 2)
    # Use label smoothing only at the very end.
    net_dict["output"]["unit"]["output_prob"]["loss_opts"]["label_smoothing"] = 0
    return net_dict
"""
)