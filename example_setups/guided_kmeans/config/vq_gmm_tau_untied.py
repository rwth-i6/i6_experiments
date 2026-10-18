"""Untied HMM states: a separate emission model for a phoneme's onset and offset.

``docs/free_segmentation/findings_and_directions.md`` §6.8. The free-segmentation
runs stall at a fixed point where a true boundary does not pay for itself: the
acoustics gain too little by splitting a merged segment (``boundary_payback``)
to cover the price of one more phoneme. With both states of a phoneme sharing
one row of the table, the coarticulated frames around a boundary count against
whichever label they land in. Untied, A's offset row and B's onset row can
absorb them - the acoustic-side counterpart of the boundary bonus in
``vq_gmm_tau_bonus``.

**The frame-level codebook, not the segment one.** Untying only helps if the
densities can tell a phoneme's onset from its offset. The colleague's k512 was
fit on mean-pooled segments, so its cells sit where whole segments average
out: a phoneme's onset and offset frames tend to fall into the same cells, and
two rows over those cells have little to learn apart. ``vq_gmm_tau_bonus``'s
frame-level codebook (unguided k-means on these frames, under the model's
covariance - ``frame_codebook``) also covers the transitional frames between
phonemes, which is where the states should differ.

**Two arms, one difference.** Everything is ``vq_gmm_tau_bonus``'s 0.5-bonus
frame-codebook arm - S=2 at a bonus of half the LM's phoneme cost (p=0.206 at
ts=0.35), LM 1.0, am=0.35, tau=4, the frozen frame k512 codebook, an exactly
uniform initial table - imported from that config so the two cannot drift apart.
The tied arm *is* that arm, job for job: it costs nothing here. Read against the
segment-codebook arm at the same bonus in ``vq_gmm_tau_bonus``, it separates the
codebook's effect from the states'. The untied arm gives each state its own row, 80 instead of 40
(phoneme-major: rows 2a and 2a+1 are phoneme a's onset and offset, see
``backoff_fb_graph.LabelLayout``). From a uniform table the two rows of a
phoneme start identical; the topology orders them in time, so the first E-step
already separates them.

**Evaluation without a decoder.** RASR has no topology with per-state
emissions, so the standard decode cannot read an untied model. Instead both
arms are scored on the training search's own output: the frame-wise argmax of
the forward-backward posteriors each epoch already computes, folded to
phonemes (a new phoneme where the argmax phoneme changes or, untied, where its
state index drops back), scored against the training transcription
(``chunked_clustering(score_reference=...)``). That is a diagnostic of what the
training search believes, over the training corpus - not a decode, and not
comparable to the cv tables - but it is the same measurement for both arms,
which is what the question needs: whether untying moves the plateau.

Epochs run before posterior hypotheses existed wrote empty ones, which would
score as 100% deletions; ``TIED_ARGMAX_SINCE`` guards the shared tied arm
against that (see there).

**Readout.** ``boundary_payback`` against the control's, and the guided PER,
deletions and hyp/ref beside it. Untying that helps shows as payback and the
payable share rising above the control's while the guided deletions fall.
"""

from sisyphus import tk

from i6_experiments.example_setups.guided_kmeans.setup.chunked_clustering import (
    DuplicateCovsJob,
    GlobalCovarianceJob,
    NormalTableJob,
    ScaleCovsJob,
    chunked_clustering,
    mixture_flavor,
)
from i6_experiments.example_setups.guided_kmeans.lib.guided_kmeans.chunked.flavors import (
    BackoffFBSearch,
)
from i6_experiments.example_setups.guided_kmeans.setup.librasr_recognition import (
    create_lexicon,
    create_fb_lexicon,
    phonetic_lm_dict,
)
from i6_experiments.example_setups.guided_kmeans.setup.decode_config import (
    phoneme_reference,
)
from i6_experiments.example_setups.guided_kmeans.setup.latex_report import (
    TextTableReport,
    clustering_statistics_per_epoch,
    param,
)
from i6_experiments.example_setups.guided_kmeans.setup.emission_balance import (
    BalanceFramesJob,
    EmissionBalanceJob,
    PhonemeLMCostJob,
    transition_boundary_penalty,
)
from i6_experiments.example_setups.guided_kmeans import tools
from i6_experiments.example_setups.guided_kmeans.config.vq_gmm_tau_lm import (
    frame_level_ls100,
)
from i6_experiments.example_setups.guided_kmeans.config import vq_gmm_tau_bonus as bonus

exp_dir = "vq_gmm_tau_untied"
version = 1

#: The bonus arm both arms run at, and the tied arm is vq_gmm_tau_bonus's
#: frame-codebook arm at it. 0.5 since 2026-10-14 (before any epoch ran): the
#: zero-bonus control had plateaued at 79.8% PER by epoch 25, while the 0.5 arm
#: was at 72.2% and still falling at 30, so untying is tested where training
#: still moves. The epoch counts follow that config's for this fraction.
BONUS_FRACTION = 0.5

NUM_STATES = bonus.NUM_STATES
NUM_PHONEMES = bonus.NUM_LABELS  # 40: silence + 39 phonemes, the tied label count

if BONUS_FRACTION in bonus.EXTENDED_FRACTIONS:
    WEIGHTS_EPOCHS = bonus.EXTENDED_EPOCHS
else:
    WEIGHTS_EPOCHS = bonus.WEIGHTS_EPOCHS
BALANCE_EPOCHS = [e for e in bonus.BALANCE_EPOCHS + bonus.EXTENDED_BALANCE_EPOCHS
                  if e <= WEIGHTS_EPOCHS]

#: First recognition pass of the shared tied arm whose hypotheses were read off
#: its posteriors. Passes before it ran on code that wrote empty hypotheses for
#: a forward-backward search; scoring those would report 100% deletions. The
#: untied arm is new, so all of its passes have them.
#:
#: 1: the change landed on 2026-10-11, and the 0.5-bonus frame-codebook arm was
#: added on 2026-10-14, so every pass of it runs on the new code. (So would the
#: zero-bonus frame arm's: the frame codebook was still in its third k-means
#: epoch when the change landed.)
TIED_ARGMAX_SINCE = 1

#: Unhashed operational settings, as vq_gmm_tau_bonus. An untied S=2 epoch does
#: the same op work as a tied one - the graph is identical, only the emission
#: lookup widens to 80 columns - so the batch that fits an 11 GB card still does.
NUM_CHUNKS = bonus.NUM_CHUNKS
EPOCH_RQMT = bonus.EPOCH_RQMT


def run():
    lexicon = create_lexicon(use_eow_phonemes=False, add_unknown_phoneme=False)
    fb_lexicon = create_fb_lexicon(use_eow_phonemes=False)
    ls100 = frame_level_ls100()
    ls100.add_alias(f"guided_kmeans/{exp_dir}/features_ls100_frames_nosil")

    global_cov = GlobalCovarianceJob(ls100.out_features).out_cov
    base_covs = DuplicateCovsJob(global_cov, bonus.NUM_CODEWORDS).out_covs
    covs = ScaleCovsJob(base_covs, bonus.TAU).out_covs
    centroids = bonus.frame_codebook(ls100.out_features, base_covs).out_centroids[
        bonus.CODEBOOK_EPOCHS]

    # Tagged phoneme reference for exactly the sequences in the feature file.
    reference = phoneme_reference("train-clean-100", ls100.out_segments)
    train_config = bonus._recog_config(bonus.LM_SCALE, 0.0, 1.0, use_fb=True)

    balance_frames = BalanceFramesJob(
        features_hdf=ls100.out_features,
        frame_labels=ls100.out_labels,
        centroids=centroids,
        covs=covs,
    ).out_frames
    lm_cost = PhonemeLMCostJob(
        arpa_path=phonetic_lm_dict[bonus.LM_ORDER],
        lexicon=fb_lexicon,
        backoff_fb_root=tools.BACKOFF_FB,
    ).out_cost

    balance_columns = ["segment_gap", "boundary_payback", "boundary_price",
                       "payable_boundaries"]
    guided_columns = ["guided_per", "guided_del", "guided_ins", "guided_sub",
                      "guided_hyp_ref"]
    # epoch_offset=0: row k is recognition pass k, which ran with model k - 1.
    # Its statistics and its guided score describe that pass; the balance
    # columns are measured on model k, the one the pass produced.
    report = TextTableReport(
        columns=[param("states", header=("HMM", "states")), "epoch"]
                + guided_columns
                + ["posterior_entropy", "segment_duration_argmax", "boundary_rate",
                   "dead_clusters"] + balance_columns,
        sort_by=["states"],
        drop_empty_rows=True,
        title=(
            f"Untied HMM states against the tied control: S={NUM_STATES}, bonus "
            f"{BONUS_FRACTION:g} of the LM's phoneme cost, LM {bonus.LM_SCALE}, "
            f"am={bonus.AM_SCALE}, uniform init, frame-level codebook. 'Guided' scores "
            f"each pass's frame-wise posterior argmax, folded to phonemes, on the "
            f"training corpus: the training search's belief, not a decode. Read "
            f"boundary payback against price; the segment-codebook arm at the same "
            f"bonus in vq_gmm_tau_bonus is the reference for the codebook's own effect. "
            f"Frequencies, entropy and durations are per phoneme for both arms; "
            f"dead clusters counts table rows, i.e. states when untied."
        ),
    )

    lam = bonus.bonus_lambda(BONUS_FRACTION)
    p = bonus.loop_for_lambda(lam, bonus.TRANSITION_SCALE)
    for states in ("tied", "untied"):
        untied = states == "untied"
        num_labels = NUM_PHONEMES * NUM_STATES if untied else NUM_PHONEMES
        init_table = NormalTableJob(
            num_labels, bonus.NUM_CODEWORDS, sigma=bonus.INIT_SIGMA, seed=bonus.SEED
        ).out_table
        # The tied name is vq_gmm_tau_bonus's, so both configs alias one run.
        name = (f"weights_tau-{bonus.TAU}_s-{NUM_STATES}_bonus-{BONUS_FRACTION}"
                f"_loop-{round(p, 4)}_ts-{bonus.TRANSITION_SCALE}_lm-{bonus.LM_SCALE}"
                f"_am-{bonus.AM_SCALE}_init-uniform_cb-frames"
                + ("_untied" if untied else ""))
        backoff_fb = BackoffFBSearch(
            backoff_fb_root=tools.BACKOFF_FB,
            arpa_path=phonetic_lm_dict[bonus.LM_ORDER],
            lexicon=fb_lexicon,
            lm_scale=bonus.LM_SCALE,
            emission_scale=1.0,
            loop_probability=p,
            silence_loop_probability=None,
            transition_scale=bonus.TRANSITION_SCALE,
            num_states=NUM_STATES,
            tie_states=not untied,
            apply_sentence_end=True,
            batch_size=bonus.BATCH_SIZE,
            max_frames=bonus.MAX_FRAMES,
            checkpoint_interval=-1,
            storage_dtype="float32",
        )
        flavor = mixture_flavor(
            centroids=centroids,
            covs=covs,
            mixtures=init_table,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=num_labels,
            distance_scale=bonus.AM_SCALE,
            use_forward_backward=True,
            update_densities=False,
            update_covariances=None,
            mixture_floor=1e-2,
            num_workers=bonus.NUM_WORKERS,
            backoff_fb=backoff_fb,
        )
        result = chunked_clustering(
            num_epochs=WEIGHTS_EPOCHS,
            features_hdf=ls100.out_features,
            recognition_config=train_config,
            lexicon=lexicon,
            num_clusters=num_labels,
            flavor=flavor,
            distance_scale=bonus.AM_SCALE,
            use_forward_backward=True,
            rasr_path=tools.RASR_PATH_FORWARD_BACKWARD,
            num_chunks=NUM_CHUNKS,
            num_workers=bonus.NUM_WORKERS,
            parent_threads=bonus.PARENT_THREADS,
            rqmt=EPOCH_RQMT,
            score_reference=reference,
            alias_prefix=f"guided_kmeans/{exp_dir}/{name}",
        )
        tk.register_output(
            f"guided_kmeans/{exp_dir}/statistics/{name}.json", result.out_statistics,
        )
        stats = clustering_statistics_per_epoch(
            result.out_epoch_statistics, name=name, epoch_offset=0, lexicon=lexicon,
        )

        penalty = transition_boundary_penalty(NUM_STATES, p, bonus.TRANSITION_SCALE)
        balance = {}
        for e in BALANCE_EPOCHS:
            job = EmissionBalanceJob(
                frames=balance_frames,
                mixtures=result.out_artifacts["mixtures"][e],
                am_scale=bonus.AM_SCALE,
                lm_cost=lm_cost,
                lm_scale=bonus.LM_SCALE,
                transition_penalty=penalty,
                num_untied_states=NUM_STATES if untied else 1,
            )
            job.add_alias(f"guided_kmeans/{exp_dir}/{name}/balance/epoch_{e:03d}")
            balance[e] = {
                "segment_gap": job.out_segment_gap,
                "boundary_payback": job.out_payback,
                "boundary_price": job.out_price,
                "payable_boundaries": job.out_payable,
            }

        first_scored = 1 if untied else TIED_ARGMAX_SINCE
        for k in sorted(int(e) for e in result.out_epoch_statistics):
            values = dict(balance.get(k, {}))
            if first_scored is not None and k >= first_scored:
                # Pass k recognized with model k - 1, which is how
                # out_guided_scores is keyed.
                values.update(result.guided_score_row(k - 1))
            report.add_row(params={"states": states}, epoch=k, statistics=stats,
                           values=values)

    report.register(
        f"guided_kmeans/{exp_dir}/convergence_{version}.txt",
        required=report, update_frequency=60,
    )


def py():
    run()
