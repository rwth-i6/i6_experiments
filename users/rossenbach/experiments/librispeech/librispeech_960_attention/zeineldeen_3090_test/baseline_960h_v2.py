import copy, os

import numpy

from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.attention_asr_config import \
  create_config, ConformerEncoderArgs, TransformerDecoderArgs, RNNDecoderArgs, ConformerDecoderArgs
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.additional_config import \
  apply_fairseq_init_to_conformer, apply_fairseq_init_to_transformer_decoder, reset_params_init
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.data import \
  build_training_datasets, build_test_dataset
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.default_tools import \
  RETURNN_EXE, RETURNN_ROOT, RETURNN_CPU_EXE
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.feature_extraction_net import \
  log10_net_10ms, log10_net_10ms_long_bn
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.pipeline import \
  training, search, get_average_checkpoint, get_best_checkpoint, search_single
from i6_experiments.users.zeineldeen.models.lm import generic_lm
from i6_experiments.users.zeineldeen.models.lm.transformer_lm import TransformerLM
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960 import ilm_helpers
from i6_experiments.users.rossenbach.experiments.librispeech.kazuki_lm.experiment import get_lm, ZeineldeenLM

train_jobs_map = {}  # dict[str, ReturnnTrainJob]
train_job_avg_ckpt = {}
train_job_best_epoch = {}

BPE_10K = 10000
BPE_5K = 5000
BPE_1K = 1000

# --------------------------- LM --------------------------- #

lstm_10k_lm_opts = {
  'lm_subnet': generic_lm.libri_lstm_bpe10k_net,
  'lm_model': generic_lm.libri_lstm_bpe10k_model,
  'name': 'lstm',
}

lstm_lm_opts_map = {
  BPE_10K: lstm_10k_lm_opts,
}

trafo_lm_net = TransformerLM(source='prev:output', num_layers=24, vocab_size=10025, use_as_ext_lm=True)
trafo_lm_net.create_network()
trafo_10k_lm_opts = {
  'lm_subnet': trafo_lm_net.network.get_net(),
  'load_on_init_opts': {
    'filename': '/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/transfo_24_d00.4096_1024.sgd.lr1.8_heads/bk-net-model/network.023',
    'params_prefix': '',
    'load_if_prefix': 'lm_output/',
  },
  'name': 'trafo'
}

bpe5k_lm = get_lm('ls960_trafo24_bs3000_5ep_5kbpe')  # type: ZeineldeenLM
trafo_5k_lm_opts = {
  'lm_subnet': bpe5k_lm.combination_network,
  'load_on_init_opts': {
    'filename': get_best_checkpoint(bpe5k_lm.train_job, key='dev_score_output/output'),
    'params_prefix': '',
    'load_if_prefix': 'lm_output/',
  },
  'name': 'trafo'
}

trafo_lm_opts_map = {
  BPE_10K: trafo_10k_lm_opts,
  BPE_5K: trafo_5k_lm_opts,
}


# ----------------------------------------------------------- #

def conformer_baseline():
  prefix_name = "experiments/librispeech/librispeech_960_attention/zeineldeen_3090_test/"

  def get_test_dataset_tuples(bpe_size):
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
      test_dataset_tuples[testset] = build_test_dataset(
        testset, use_raw_features=True, bpe_size=bpe_size,
      )
    return test_dataset_tuples

  def run_train(exp_name, train_args, train_data, feature_extraction_net, num_epochs, recog_epochs, **kwargs):
    exp_prefix = os.path.join(prefix_name, exp_name)
    returnn_config = create_config(
      training_datasets=train_data,
      **train_args,
      feature_extraction_net=feature_extraction_net,
      recog_epochs=recog_epochs,
    )
    train_job = training(exp_prefix, returnn_config, RETURNN_CPU_EXE, RETURNN_ROOT, num_epochs=num_epochs)
    return train_job

  def run_single_search(
          exp_name, train_data, search_args, checkpoint, feature_extraction_net, recog_dataset, recog_ref,
          mem_rqmt=8, time_rqmt=4, **kwargs):

    exp_prefix = os.path.join(prefix_name, exp_name)
    returnn_search_config = create_config(
      training_datasets=train_data,
      **search_args,
      feature_extraction_net=feature_extraction_net,
      is_recog=True,
    )
    search_single(
      exp_prefix,
      returnn_search_config,
      checkpoint,
      recognition_dataset=recog_dataset,
      recognition_reference=recog_ref,
      returnn_exe=RETURNN_CPU_EXE,
      returnn_root=RETURNN_ROOT,
      mem_rqmt=mem_rqmt,
      time_rqmt=time_rqmt,
    )

  def run_lm_fusion(
          lm_type, exp_name, epoch, test_set_names, lm_scales, train_job, train_data, feature_net, bpe_size, args,
          beam_size=12, prior_scales=None, prior_type=None, mini_lstm_ckpt=None, length_norm=True,
          prior_type_name=None, coverage_scale=None, coverage_threshold=None, **kwargs,
  ):
    assert lm_type in ['lstm', 'trafo'], 'lm type should be lstm or trafo'

    if isinstance(lm_scales, float):
      lm_scales = [lm_scales]
    if prior_scales and isinstance(prior_scales, float):
      prior_scales = [prior_scales]
    if isinstance(test_set_names, str):
      test_set_names = [test_set_names]
    assert isinstance(test_set_names, list)

    if epoch == 'avg':
      search_checkpoint = train_job_avg_ckpt[exp_name]
    elif epoch == 'best':
      search_checkpoint = train_job_best_epoch[exp_name]
    else:
      assert isinstance(epoch, int), 'epoch must be either a defined integer or a string in {avg, best}.'
      search_checkpoint = train_job.out_checkpoints[epoch]

    ext_lm_opts = lstm_lm_opts_map[bpe_size] if lm_type == 'lstm' else trafo_lm_opts_map[bpe_size]

    time_rqmt = 1.0

    search_args = copy.deepcopy(args)

    if lm_type == 'lstm':
      if beam_size > 128:
        search_args['batch_size'] = 4000 * 160

    if lm_type == 'trafo':
      search_args['batch_size'] = 4000 * 160 if beam_size <= 32 else 2000 * 160
      time_rqmt = 1

    search_args['beam_size'] = beam_size
    if kwargs.get('batch_size', None):
      search_args['batch_size'] = kwargs['batch_size']

    if not length_norm:
      search_args['decoder_args'].length_normalization = False

    if 'decoder_args' in kwargs:
      for k, v in kwargs['decoder_args'].items():
        setattr(search_args['decoder_args'], k, v)

    scales = [(e,) for e in lm_scales]

    for test_set in test_set_names:

      if prior_scales:
        import itertools
        scales = itertools.product(lm_scales, prior_scales)

      for scale in scales:
        lm_scale = scale[0]
        prior_scale = scale[1] if len(scale) == 2 else None
        if prior_scale and prior_scale > lm_scale:
          continue

        # External LM opts
        ext_lm_opts['lm_scale'] = lm_scale
        search_args['ext_lm_opts'] = ext_lm_opts

        # ILM opts
        if prior_scale:
          ilm_opts = {
            'scale': prior_scale,
            'type': prior_type,
            'ctx_dim': search_args['encoder_args'].enc_key_dim,  # this is needed for mini-lstm
          }
          # this is needed for mini-self-att
          if hasattr(search_args['decoder_args'], 'num_layers'):
            ilm_opts['num_dec_layers'] = search_args['decoder_args'].num_layers
            search_args['decoder_args'].create_ilm_decoder = True
            search_args['decoder_args'].ilm_type = prior_type

          ilm_opts.update(kwargs.get('ilm_train_opts', {}))  # example for FFN, etc

          search_args['prior_lm_opts'] = ilm_opts
          search_args['preload_from_files'] = {
            'prior_lm': {
              'filename': search_checkpoint,  # copy ASR decoder to be used as ILM decoder
              'prefix': 'prior_'
            }
          }
          if prior_type == 'mini_lstm' or prior_type == 'ffn':
            assert mini_lstm_ckpt, 'Mini-LSTM checkpoint not set.'
            search_args['preload_from_files'].update(
              {
                'mini_lstm': {
                  'filename': mini_lstm_ckpt,
                  'prefix': 'mini_'
                }
              })

        if prior_type_name is None:
          prior_type_name = prior_type

        lm_desc = f'lm-scale-{lm_scale}'
        if prior_scale:
          lm_desc += f'-prior-{prior_scale}-{prior_type_name}'
        lm_desc += f'-beam-{beam_size}'
        if length_norm is False:
          lm_desc += '-woLenNorm'

        if coverage_scale and coverage_threshold:
          assert isinstance(search_args['decoder_args'], RNNDecoderArgs)
          search_args['decoder_args'].coverage_scale = coverage_scale
          search_args['decoder_args'].coverage_threshold = coverage_threshold
          lm_desc += f'_coverage-thre{coverage_threshold}-scale{coverage_scale}'

        name = f'{exp_name}/recog-{lm_type}-lm/ep-{epoch}/{lm_desc}/{test_set}'

        test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)

        run_single_search(
          exp_name=name,
          train_data=train_data,
          search_args=search_args,
          checkpoint=search_checkpoint,
          feature_extraction_net=feature_net,
          recog_dataset=test_dataset_tuples[test_set][0],
          recog_ref=test_dataset_tuples[test_set][1],
          time_rqmt=kwargs.get('time_rqmt', time_rqmt),
        )

  def run_search(
          exp_name, train_args, train_data, train_job, feature_extraction_net, num_epochs, search_args, recog_epochs,
          bpe_size, **kwargs):

    exp_prefix = os.path.join(prefix_name, exp_name)

    search_args = search_args if search_args is not None else train_args

    returnn_search_config = create_config(
      training_datasets=train_data,
      **search_args,
      feature_extraction_net=feature_extraction_net,
      is_recog=True,
    )

    num_avg = kwargs.get('num_avg', 4)
    averaged_checkpoint = get_average_checkpoint(
      train_job, returnn_exe=RETURNN_CPU_EXE, returnn_root=RETURNN_ROOT, num_average=num_avg,
    )
    if num_avg == 4:  # TODO: just for now to not break hashes
      train_job_avg_ckpt[exp_name] = averaged_checkpoint

    best_checkpoint = get_best_checkpoint(train_job)
    train_job_best_epoch[exp_name] = best_checkpoint

    if recog_epochs is None:
      default_recog_epochs = [80 * i for i in range(1, int(num_epochs / 80) + 1)]
      if num_epochs % 80 != 0:
        default_recog_epochs += [num_epochs]
    else:
      default_recog_epochs = recog_epochs

    test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)

    for ep in default_recog_epochs:
      search(
        exp_prefix + f"/recogs/ep-{ep}", returnn_search_config, train_job.out_checkpoints[ep],
        test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

    search(
      exp_prefix + "/default_last", returnn_search_config, train_job.out_checkpoints[num_epochs],
      test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

    search(
      exp_prefix + "/default_best", returnn_search_config, best_checkpoint,
      test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

    search(
      exp_prefix + f"/average_{num_avg}", returnn_search_config, averaged_checkpoint,
      test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

  def run_exp(
          exp_name, train_args, feature_extraction_net=log10_net_10ms, num_epochs=300, search_args=None,
          recog_epochs=None, bpe_size=10000, **kwargs
  ):
    if train_args.get('retrain_checkpoint', None):
      assert kwargs.get('epoch_wise_filter', None) is None, 'epoch_wise_filter should be disabled for retraining.'
    train_data = build_training_datasets(
      bpe_size=bpe_size,
      use_raw_features=True,
      epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
      link_speed_perturbation=train_args.get("speed_pert", True),
      seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
    )
    train_job = run_train(exp_name, train_args, train_data, feature_extraction_net, num_epochs, recog_epochs, **kwargs)
    train_jobs_map[exp_name] = train_job

    run_search(
      exp_name, train_args, train_data, train_job, feature_extraction_net, num_epochs, search_args, recog_epochs,
      bpe_size=bpe_size, **kwargs
    )
    return train_job, train_data

  # --------------------------- General Settings --------------------------- #

  conformer_enc_args = ConformerEncoderArgs(
    num_blocks=12,
    input_layer="conv-6",
    att_num_heads=8,
    ff_dim=2048,
    enc_key_dim=512,
    conv_kernel_size=32,
    pos_enc="rel",
    dropout=0.1,
    att_dropout=0.1,
    l2=0.0001,
  )
  apply_fairseq_init_to_conformer(conformer_enc_args)
  conformer_enc_args.ctc_loss_scale = 1.0

  rnn_dec_args = RNNDecoderArgs()

  trafo_dec_args = TransformerDecoderArgs(
    num_layers=6,
    embed_dropout=0.1,
    label_smoothing=0.1,
    apply_embed_weight=True,
    pos_enc="rel",
  )
  apply_fairseq_init_to_transformer_decoder(trafo_dec_args)

  conformer_dec_args = ConformerDecoderArgs()
  apply_fairseq_init_to_conformer(conformer_dec_args)

  training_args = dict()

  # LR scheduling
  training_args["const_lr"] = [42, 100]  # use const LR during pretraining
  training_args["wup_start_lr"] = 0.0002
  training_args["wup"] = 20
  training_args["with_staged_network"] = True
  training_args["speed_pert"] = True

  trafo_training_args = copy.deepcopy(training_args)
  trafo_training_args["pretrain_opts"] = {"variant": 3, "initial_batch_size": 20000 * 160}
  trafo_training_args["pretrain_reps"] = 5
  trafo_training_args["batch_size"] = 12000 * 160  # frames * samples per frame

  trafo_dec_exp_args = copy.deepcopy(
    {
      **trafo_training_args,
      "encoder_args": conformer_enc_args,
      "decoder_args": trafo_dec_args,
    }
  )

  conformer_dec_exp_args = copy.deepcopy(trafo_dec_exp_args)
  conformer_dec_exp_args['decoder_args'] = conformer_dec_args

  lstm_training_args = copy.deepcopy(training_args)
  lstm_training_args["pretrain_opts"] = {"variant": 3, "initial_batch_size": 22500 * 160 * 2}
  lstm_training_args["pretrain_reps"] = 5
  lstm_training_args["batch_size"] = 15000 * 160 * 2  # frames * samples per frame
  lstm_training_args["accum_grad"] = 1  # frames * samples per frame

  lstm_dec_exp_args = copy.deepcopy(
    {
      **lstm_training_args,
      "encoder_args": conformer_enc_args,
      "decoder_args": rnn_dec_args,
    }
  )

  # --------------------------- Experiments --------------------------- #

  oclr_args = copy.deepcopy(lstm_dec_exp_args)
  oclr_args["oclr_opts"] = {
    "peak_lr": 9e-4,
    "final_lr": 1e-6,
    "cycle_ep": 915,
    "total_ep": 2035,  # 20 epochs
    "n_step": 675,
  }
  oclr_args["encoder_args"].input_layer = "conv-6"
  oclr_args['encoder_args'].use_sqrd_relu = True

  # Old Wo LM:
  #
  # dev-clean  2.28
  # dev-other  5.63
  # test-clean  2.48
  # test-other  5.71

  name = "base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009"
  train_j, train_data = run_exp(name, train_args=oclr_args, num_epochs=2035)

  # TODO: retrain

  # TODO: LM + ILM
  for beam_size in [12]:
    run_lm_fusion(
      lm_type='trafo', exp_name=name, epoch='avg',
      test_set_names=['dev-clean', 'dev-other'],
      lm_scales=[0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52],
      train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=oclr_args,
      beam_size=beam_size, bpe_size=BPE_10K,
    )

