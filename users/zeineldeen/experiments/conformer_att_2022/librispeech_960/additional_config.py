from i6_experiments.users.zeineldeen.models.lm.transformer_lm import TransformerLM
from .attention_asr_config import ConformerEncoderArgs, TransformerDecoderArgs, ConformerDecoderArgs


def get_lm_opts():
    transf_lm_net = TransformerLM(
        source='prev:output', num_layers=24, vocab_size=2051, use_as_ext_lm=True, prefix_name='lm_')
    transf_lm_net.create_network()
    transf_lm_opts = {
        'lm_subnet': transf_lm_net.network.get_net(),
        'lm_output_prob_name': 'lm_output',
        'is_recurrent': True,
        'preload_from_files': {
            'lm_model': {
                'filename': '/work/asr4/zeineldeen/setups-data/librispeech/2021-02-21--lm-bpe/dependencies/lm_models/transf/epoch.017',
                'prefix': 'lm_'
            }
        },
        'name': 'trafo',
    }
    return transf_lm_opts


fairseq_ff_init = "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"  # limit = sqrt(6 / (fan_in + fan_out))
fairseq_mhsa_init = "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=0.5)"  # limit = sqrt(6 * 0.5 / (fan_in + fan_out)) = sqrt(3 / (fan_in + fan_out))


def apply_fairseq_init_to_conformer(conformer_args: [ConformerEncoderArgs,ConformerDecoderArgs]):
    # fairseq init
    conformer_args.ff_init = fairseq_ff_init
    conformer_args.mhsa_init = fairseq_mhsa_init
    conformer_args.mhsa_out_init = fairseq_ff_init
    conformer_args.conv_module_init = fairseq_ff_init


def apply_fairseq_init_to_transformer_decoder(transformer_dec_args: TransformerDecoderArgs):
    # fairseq init
    transformer_dec_args.ff_init = fairseq_ff_init
    transformer_dec_args.mhsa_init = fairseq_mhsa_init
    transformer_dec_args.mhsa_out_init = fairseq_ff_init


def reset_params_init(args: [ConformerEncoderArgs,TransformerDecoderArgs]):
    # reset parameters init
    args.ff_init = None
    args.mhsa_init = None
    args.mhsa_out_init = None
    if isinstance(args, ConformerEncoderArgs):
        args.conv_module_init = None