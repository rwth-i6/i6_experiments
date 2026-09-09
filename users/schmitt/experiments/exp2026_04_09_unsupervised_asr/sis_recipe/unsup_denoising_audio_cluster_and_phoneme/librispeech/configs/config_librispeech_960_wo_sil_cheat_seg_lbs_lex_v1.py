"""
Cheating-segmentation setup (zyang's oracle GMM segment clusters, k512) with the *text* phonemized in the
phoneme set of the GMM alignment those clusters come from, instead of our fairseq/g2p_en set.

Why: `config_librispeech_960_wo_sil_cheat_seg_v1` pairs cluster sequences cut from a GMM alignment over the
39-phoneme LibriSpeech inventory (no silence -- silence is implicit in the neighbouring segments -- and no
``'``) with text phonemized by a *different* lexicon (g2p_en over the LM corpus, 41 symbols incl. ``'`` and
``<SIL>``). Measured against the alignment's own segment phonemes, that text disagrees on 6.6% of the tokens
and matches the segment count on only 25% of the utterances -- pronunciation variants (``and``: AE vs AH,
``was``: AH vs AA, ``to``: T UW/T IH/T AH) and adjacent-repeat merging (``some mysterious`` -> one M). Here the
text uses the official LibriSpeech lexicon with the aligner's preferred variants plus repeat collapsing
(`data.librispeech.text.get_lbs_lexicon`): 3.8% disagreement / 38% equal length. The vocab is
``[SILENCE]=0, AA=1, ..., ZH=39`` (size 40; index 0 never occurs in text), the index space of zyang's
``gmm_segment_phonemes.*.hdf`` and lkleppel's ``phoneme.lex.xml.gz``, so outputs are directly comparable to
theirs. All 281k train utterances are kept (no LID filter, OOVs G2P'd).

NB the eval "dev-other" is a fresh 3000-utt sample of train-other-960 (different pool, see
`build_test_datasets_w_cheating_clusters`) in the new phoneme set -- PERs are not comparable to the
historical cheat-seg numbers.
"""

import copy
import functools

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from i6_core.returnn.config import ReturnnConfig
from i6_core.serialization import Collection

from ....train_exp import run_experiment
from ..data.common import build_training_datasets_w_cheating_clusters, build_test_datasets_w_cheating_clusters
from ....data.librispeech.text import get_lbs_lexicon
from ... import __setup_base_name__

from .config_librispeech_960_wo_sil_cheat_seg_v1 import (
    base_config as cheat_seg_base_config,
    settings,
    base_num_epochs,
    get_keep_epochs,
    alternate_batching,
)
from .config_librispeech_960_wo_sil_v1 import _text_recon_sweep


# the aligner-preferred pronunciation variants use the GMM alignment only for corpus-level counts (which variant
# of "to"/"the"/"was"/... it realized most often), no utterance pairing. `variant_selection="first"` is the
# alignment-free alternative (5.7% disagreement instead of 3.8%).
lexicon = get_lbs_lexicon(variant_selection="aligner", collapse_repeats=True)

train_data = build_training_datasets_w_cheating_clusters(
    sil_prob=0.0, surround_w_sil=False, settings=settings, lexicon=lexicon
)
test_data_dict_wo_sil = build_test_datasets_w_cheating_clusters(sil_prob=0.0, surround_w_sil=False, lexicon=lexicon)

base_config = dict_update_deep(
    copy.deepcopy(cheat_seg_base_config),
    {
        # 40 instead of 41 (39 phonemes + the [SILENCE] index placeholder)
        "model_args.text_out_dim": train_data.datastreams["phon_indices"].vocab_size,
    },
)
assert base_config["model_args"]["text_out_dim"] == 40, base_config["model_args"]["text_out_dim"]


run_experiment = functools.partial(
    run_experiment,
    train_data=train_data,
    test_data_dict=test_data_dict_wo_sil,
    keep_epochs=get_keep_epochs(base_num_epochs),
    additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    analysis_opts={
        "checkpoints": get_keep_epochs(base_num_epochs),
        "max_plotted_seqs": 20,
        "cosine_similarity_summary": True,
    },
    cross_att_opts={
        "checkpoints": get_keep_epochs(base_num_epochs),
        "input_modality": "audio",
        "output_modality": "text",
        "max_plotted_seqs": 20,
    },
    ppl_opts={
        "checkpoints": get_keep_epochs(base_num_epochs),
        "input_modality": "audio",
        "test_data_dict": test_data_dict_wo_sil,
    },
    recog_variants=[
        {
            "recog_name": "recon_audio",
            "input_modality": "audio",
            "output_modality": "audio",
            "mask_input": True,
            "masking_opts": copy.deepcopy(base_config["train_args"]["audio_masking_opts"]),
            "keep_epochs": get_keep_epochs(base_num_epochs),
        },
        {
            "recog_name": "recon_text",
            "input_modality": "text",
            "output_modality": "text",
            "mask_input": True,
            "masking_opts": copy.deepcopy(base_config["train_args"]["text_masking_opts"]),
            "keep_epochs": get_keep_epochs(base_num_epochs),
        },
        *_text_recon_sweep(base_num_epochs),
    ],
)


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    # plain shared denoising autoencoder -- the model whose per-symbol encoder states were the useful ones
    # in the embedding-geometry analyses (the GAN one degraded them)
    run_experiment(
        training_name=f"{prefix_name}/baseline",
        config=copy.deepcopy(base_config),
    )

    run_experiment(
        training_name=f"{prefix_name}/baseline_lstm-gan",
        config=dict_update_deep(
            copy.deepcopy(base_config),
            {
                "model_args.discriminator_type": "lstm",
                "train_args": {
                    "adv_loss_scale": 0.1,
                },
            },
        ),
    )
