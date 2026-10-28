"""Decoding with more than one HMM state per phoneme.

The model is the colleague's free-segmentation run that reached 21.0% PER
(``vq_gmm_tau_bonus``, arm ``bonus-0.5`` with the segment codebook, epoch 200):
a mixture model over the frozen k512 codebook, trained with a 2-state
left-to-right topology (tied emissions, loop 0.2061, transition scale 0.35) by
his CUDA forward-backward op. He decoded it with the 1-state linear search
(loop 0.5, transition scale 1, i.e. no boundary price), so training and decoding
topology differ. Here the same model is decoded with ``num_hmm_states`` 1, 2 and 3
in the linear search (``tools.RASR_PATH_HMM_STATES``): every state loops with the
loop probability and moves on with 1 - p, and only the last state leaves the
phoneme, so a phoneme lasts at least S frames.

Decoded on the same data as his number: the silence-free cv frames (not pooled),
restricted to the same seeded 10% of cv, AM scale 0.35, LM scale 1.0, 5-gram.

* ``repro``: his decode exactly (1 state, silence allowed), once with the old
  linear-search build and once with the new one; the two must agree, and match
  his 21.0%.
* ``free``: loop 0.5 at transition scale 1, the topology charges nothing per
  boundary, so S only adds the minimum duration.
* ``train-topology``: loop 0.2061 at transition scale 0.35, the transitions he
  trained with (for S=2 exactly his training topology).

Silence is forbidden outside ``repro``: blank keeps a single state, so with S > 1
a one-frame silence would be a way around the minimum duration.

**Tied against untied emissions.** The colleague's ``vq_gmm_tau_untied`` trains the
same S=2 bonus-0.5 setup on the frame-level codebook twice: tied, and untied with
one table row per state (80 rows, phoneme-major: rows 2a and 2a+1 are phoneme a's
two states). Both arms are decoded with his training topology (S=2, loop 0.2061,
transition scale 0.35), the untied one with ``tie_hmm_states=False`` so that each
state reads its own score column.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    COLLEAGUE_CENTROIDS_K512,
    GMM_ALIGNMENT_CV,
    INPUT_DATA as input_data,
)
from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import (
    SegmentedFeaturesFromAlignmentJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_lexicon,
    create_recog_rasr_config,
)
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import (
    decode_and_score,
    DecodeConfig,
)
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import (
    DatasetConfig,
    SegmentFile,
)
from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob
from i6_experiments.example_setups.guided_kmeans.config.vq_unsupervised import (
    silence_free_cv_features,
)

exp_dir = "vq_hmm_states_decode"
version = 1

#: Epoch 200 of the colleague's vq_gmm_tau_bonus bonus-0.5 segment-codebook arm
#: (21.0% PER free-segmentation decode, 11.8% with oracle segmentation). Lives in
#: his work directory.
MODEL_DIR = tk.Path(
    "/u/mann/experiments/2026-06-09--guided-k-means/work/i6_experiments/example_setups/"
    "guided_kmeans/setup/chunked_clustering/GuidedClusteringEpochJob.Kgbrdei8HIuC/output/model"
)

LM_ORDER = 5
LM_SCALE = 1.0
AM_SCALE = 0.35
BEAM_SIZE = 1000
#: Share of cv decoded, seeded as in his config, so the subset is the same.
CV_FRACTION = 0.1
DECODE_WORKERS = 8

_MANN_EPOCH_JOBS = (
    "/work/asr4/mann/experiments/2026-06-09--guided-k-means/work/i6_experiments/example_setups/"
    "guided_kmeans/setup/chunked_clustering"
)
#: vq_gmm_tau_untied's two arms at UNTIED_EPOCH: name -> (model directory, tied emissions)
UNTIED_EPOCH = 50
UNTIED_MODELS = {
    "frames-tied": (tk.Path(f"{_MANN_EPOCH_JOBS}/GuidedClusteringEpochJob.Ae4peBV0q7WZ/output/model"), True),
    "frames-untied": (tk.Path(f"{_MANN_EPOCH_JOBS}/GuidedClusteringEpochJob.wLzGD1zceI6O/output/model"), False),
}

#: name -> (loop probability, transition scale)
TOPOLOGIES = {
    "free": (0.5, 1.0),
    "train-topology": (0.2061, 0.35),
}
NUM_HMM_STATES = [1, 2, 3]


def _recog_config(num_hmm_states, loop_prob, transition_scale, forbid_blank, tie_hmm_states=True):
    return create_recog_rasr_config(
        lm_scale=LM_SCALE,
        emission_scale=1.0,
        transition_scale=transition_scale,
        loop_probability=loop_prob,
        silence_loop_probability=loop_prob,
        lm_order=LM_ORDER,
        use_eow_phonemes=False,
        max_beam_size=BEAM_SIZE,
        forbid_blank=forbid_blank,
        num_hmm_states=num_hmm_states,
        tie_hmm_states=tie_hmm_states,
    )


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)

    cv_frames = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["cv"]["features"],
        alignment=GMM_ALIGNMENT_CV,
        exclude_labels=(0,),
        pooling="none",
    )
    decode_segments = ShuffleAndSplitSegmentsJob(
        silence_free_cv_features().out_segments,
        split={"decode": CV_FRACTION, "rest": 1.0 - CV_FRACTION},
    ).out_segments["decode"]
    cv_dataset = DatasetConfig(
        audio_hdf_path=cv_frames.out_features,
        sampling_method=SegmentFile(decode_segments),
        precomputed=True,
    )

    # decode name -> (model directory, num HMM states, tied emissions, loop probability,
    #                 transition scale, forbid blank, RASR build)
    decodes = {
        "bonus-0.5_ep-200_repro_old-build": (MODEL_DIR, 1, True, 0.5, 1.0, False, tools.RASR_PATH),
        "bonus-0.5_ep-200_repro": (MODEL_DIR, 1, True, 0.5, 1.0, False, tools.RASR_PATH_HMM_STATES),
    }
    for topology, (loop_prob, transition_scale) in TOPOLOGIES.items():
        for num_hmm_states in NUM_HMM_STATES:
            decodes[f"bonus-0.5_ep-200_{topology}_states-{num_hmm_states}"] = (
                MODEL_DIR, num_hmm_states, True, loop_prob, transition_scale, True, tools.RASR_PATH_HMM_STATES,
            )
    loop_prob, transition_scale = TOPOLOGIES["train-topology"]
    for arm, (model_dir, tied) in UNTIED_MODELS.items():
        decodes[f"{arm}_ep-{UNTIED_EPOCH}_train-topology_states-2"] = (
            model_dir, 2, tied, loop_prob, transition_scale, True, tools.RASR_PATH_HMM_STATES,
        )

    recog_results = []
    for name, (model_dir, num_hmm_states, tied, loop_prob, transition_scale, forbid_blank, rasr_path) in decodes.items():
        decode_name = f"{name}_loop-{loop_prob}_ts-{transition_scale}"
        decode_config = DecodeConfig(
            centroids=COLLEAGUE_CENTROIDS_K512,
            model_dir=model_dir,
            recog_rasr_config=_recog_config(num_hmm_states, loop_prob, transition_scale, forbid_blank, tied),
            distance_scale=AM_SCALE,
            write_frame_labels=True,
            num_workers=DECODE_WORKERS,
        )
        res = decode_and_score(
            decode_name, "cv", decode_config, cv_dataset,
            rasr_path=rasr_path, device="cpu", corpus_key="train-other-960",
            alias_prefix=f"guided_kmeans/{exp_dir}",
        )
        res.fer = FrameErrorRateJob(res.frame_labels, cv_frames.out_labels, lexicon).out_fer
        tk.register_output(f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per)
        tk.register_output(f"guided_kmeans/{exp_dir}/eval/{decode_name}_fer", res.fer)
        recog_results.append(res)

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )


def py():
    run()
