"""Supervised vector-quantized baseline - a diagnostic, not an experiment.

Measures the ceiling: given the *right* labels, how well can a discrete model
over a fixed codebook do? Everything unsupervised in this setup is trying to
reach that number, and without it a failure is not diagnosable.

It exists because the unsupervised mixture runs failed in a way that needed
explaining. Three measurements, taken before any of this was built:

* a trained :class:`...chunked.models.GaussianMixtureModel` from
  ``codebook_mixture`` scores **1.7%** frame accuracy on its own label set with
  no language model and no transitions - *below* the 2.5% chance level. Its
  acoustic scores carry no label information at all, which is why sweeping the
  LM scale (1.0 as well as 30) changed nothing: there was nothing to weight.
* the codebooks are nevertheless fine. With an oracle table over the same
  frames, our stage-1 codebook reaches 65.6% at K=512 and a colleague's FAISS
  codebook 67.2% - a difference that matters far less than the gap to 1.7%.
* pooled over the reference segmentation rather than per frame, a counted table
  reaches **82.4% held out**.

So the codebook is not the problem, the model class is not the problem, and the
search scale is not the problem: the unsupervised weight estimation is. This
config pins everything except that, by replacing the learned table with a
counted one.

What it does
    1. :class:`...setup.vq_baseline.SegmentedFeaturesFromAlignmentJob` pools the
       raw features into the reference alignment's phoneme segments and drops
       the silence ones. With ``exclude_labels=()`` this reproduces the
       setup's existing ``segmented_features_*.hdf`` byte for byte (verified:
       344,839 segments for ls-cv, values equal to float32 rounding), so the
       silence-free file differs from the established input in exactly one way.
    2. :class:`...setup.vq_baseline.SupervisedVQTableJob` counts
       ``p(codeword | label)`` on 80% of the sequences and reports accuracy on
       the other 20%. It writes a model directory, so the result decodes
       through the ordinary path with no decode-side change.
    3. The held-out split is decoded and scored.

Caveats, all deliberate
    The segmentation is the reference alignment's, so this is an oracle-segment
    experiment - it measures labelling, not segmentation. Silence is removed
    because the colleague's codebook was trained without it; ours was not, which
    is one reason to run both. And there is no ls-100 alignment in this setup,
    only ls-cv (2786 sequences) and a 120-sequence debug subset, so the
    held-out split of cv is where generalization has to be measured.

If this decodes badly *despite* an 82% held-out table, the problem is the search
or the decode configuration rather than anything upstream of it - which is the
one thing the unsupervised runs could not tell us.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.constants import (
    INPUT_DATA as input_data,
    GMM_ALIGNMENT_CV,
    COLLEAGUE_CENTROIDS_K512 as COLLEAGUE_CENTROIDS,
)
from i6_experiments.example_setups.guided_kmeans.setup.vq_baseline import (
    SegmentedFeaturesFromAlignmentJob,
    SupervisedVQTableJob,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_recog_rasr_config,
    create_lexicon,
)
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import (
    decode_and_score,
    DecodeConfig,
)
from i6_experiments.example_setups.guided_kmeans.setup.dataset_config import (
    DatasetConfig,
    SegmentFile,
)
from i6_experiments.example_setups.guided_kmeans.setup.report import create_report
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.setup.score import FrameErrorRateJob

exp_dir = "vq_supervised"
version = 1



def run():
    use_eow_phonemes = False
    num_labels = 40 if not use_eow_phonemes else 79
    lm_order = 3

    # The floor is not cosmetic: 74.6% of a counted table's entries are zero,
    # every zero is an +inf score, and a codeword no label admits leaves a frame
    # with no viable label. It is also a scale knob - the label contrast the
    # search sees was measured at 13.0 nats at 1e-3 and 8.4 at 1e-1 - so it is
    # swept alongside the acoustic scale rather than fixed.
    table_floors = [1e-3, 1e-2, 1e-1]
    heldout_fraction = 0.2
    split_seed = 42

    # One vector per phoneme segment, so a segment is one label and there is no
    # self-loop to pay for - the same pairing the pre-segmented training configs
    # use. Kept at 0.0 deliberately; a nonzero loop here would model a
    # transition that cannot occur.
    decode_loop_prob = 0.0
    decode_lm_scales = [0.0, 1.0, 5.0, 20.0]
    distance_scales = [1.0]
    decode_beam_size = 100_000

    lexicon = create_lexicon(use_eow_phonemes=use_eow_phonemes, add_unknown_phoneme=False)

    # Silence-free, segmented. exclude_labels=() would reproduce the existing
    # segmented_features_ls-cv.hdf exactly; (0,) is that minus the silence
    # segments, which is what a codebook trained without silence needs.
    features = SegmentedFeaturesFromAlignmentJob(
        features_hdf=input_data["cv"]["features"],
        alignment=GMM_ALIGNMENT_CV,
        exclude_labels=(0,),
        pooling="mean",
    )
    features.add_alias(f"guided_kmeans/{exp_dir}/features_cv_nosil")
    tk.register_output(
        f"guided_kmeans/{exp_dir}/features/statistics.json", features.out_statistics
    )

    recog_results = []
    for table_floor in table_floors:
        table = SupervisedVQTableJob(
            features_hdf=features.out_features,
            labels=features.out_labels,
            centroids=COLLEAGUE_CENTROIDS,
            num_labels=num_labels,
            table_floor=table_floor,
            heldout_fraction=heldout_fraction,
            split_seed=split_seed,
        )
        name = f"floor-{table_floor}_heldout-{heldout_fraction}_seed-{split_seed}"
        table.add_alias(f"guided_kmeans/{exp_dir}/table_{name}")
        tk.register_output(
            f"guided_kmeans/{exp_dir}/table/{name}_diagnostics.json", table.out_diagnostics
        )
        # The number this whole config exists to produce, available without
        # decoding anything: what the model gets right on sequences it never saw.
        tk.register_output(
            f"guided_kmeans/{exp_dir}/table/{name}_heldout_accuracy", table.out_accuracy
        )

        # Decode the half the table was not counted on.
        heldout_dataset = DatasetConfig(
            audio_hdf_path=features.out_features,
            sampling_method=SegmentFile(table.out_heldout_segments),
            precomputed=True,
        )
        for lm_scale in decode_lm_scales:
            for distance_scale in distance_scales:
                decode_name = f"{name}_lm-{lm_scale}_am-{distance_scale}"
                recognition_config = create_recog_rasr_config(
                    lm_scale=lm_scale,
                    emission_scale=1.0,
                    transition_scale=None,
                    loop_probability=decode_loop_prob,
                    silence_loop_probability=decode_loop_prob,
                    lm_order=lm_order,
                    use_eow_phonemes=use_eow_phonemes,
                    max_beam_size=decode_beam_size,
                )
                decode_config = DecodeConfig(
                    # Passed for the interface's sake and ignored for scoring:
                    # model_dir carries the manifest, so the callback loads a
                    # VectorQuantizedModel and gets the table with it.
                    centroids=COLLEAGUE_CENTROIDS,
                    model_dir=table.out_model,
                    recog_rasr_config=recognition_config,
                    distance_scale=distance_scale,
                    write_frame_labels=True,
                )
                res = decode_and_score(
                    decode_name,
                    "cv",
                    decode_config,
                    heldout_dataset,
                    rasr_path=tools.RASR_PATH,
                    device="cpu",
                    corpus_key="train-other-960",
                )
                if res.frame_labels is not None:
                    res.fer = FrameErrorRateJob(
                        res.frame_labels, GMM_ALIGNMENT_CV, lexicon
                    ).out_fer
                    tk.register_output(
                        f"guided_kmeans/{exp_dir}/eval/{decode_name}_fer", res.fer
                    )
                tk.register_output(
                    f"guided_kmeans/{exp_dir}/per/{decode_name}_per", res.per
                )
                recog_results.append(res)

    tk.register_report(
        f"guided_kmeans/{exp_dir}/recognition/report_{version}.txt",
        values=create_report(recog_results),
        required=True,
    )


def py():
    run()
