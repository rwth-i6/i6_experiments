"""
TTS-encoder project -- MAIN RZ recipe (single-GPU).
Owns all RZ experiments:
the audio-only baselines (base-ls*),
and the single-GPU / nep=100 TTS-encoder runs (ref-match + pseudo-enc),
the clean A/B vs Nick's single-GPU 3.53 reference.
Imports ``train_ls_base`` + ``DbMelFeatureExtractor`` from the ``exp2026_05_28_tts_encoder`` library,
and ``_train_tts_encoder`` from the ``_fzj`` recipe.
"""

from __future__ import annotations

import returnn.frontend as rf

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder import train_ls_base, DbMelFeatureExtractor
from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder_fzj import _train_tts_encoder
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.muon import Muon

__all__ = ["py"]
__setup_root_prefix__ = "exp2026_05_28_tts_encoder_rz"


def py():
    prefix = get_setup_prefix_for_module(__name__)
    # WER comments above each call are %, joint AED+CTC first-pass (aed+ctc/recog-1stpass-res.txt),
    # as {"dev-clean", "dev-other", "test-clean", "test-other"}.
    # --- Audio-only single-GPU baselines (moved here from the exp2026_05_28_tts_encoder library). ---
    # base-ls: standard log-mel, bhv 24 -> same hash as the imported base-librispeech baseline (not re-run).
    # {"dev-clean": 1.87, "dev-other": 4.06, "test-clean": 2.06, "test-other": 4.38}
    train_ls_base("base-ls", prefix=prefix)
    # base-ls-dbmel: DbMel front-end (== frozen GlowTTS space); bhv 25 is behavior-neutral here.
    # {"dev-clean": 1.82, "dev-other": 4.19, "test-clean": 2.06, "test-other": 4.49}
    train_ls_base(
        "base-ls-dbmel", prefix=prefix, feature_extraction=rf.build_dict(DbMelFeatureExtractor), behavior_version=25
    )
    # base-ls-newrtrn: base-ls re-run under the current RZ RETURNN (isolates the RETURNN-version effect).
    # {"dev-clean": 1.86, "dev-other": 4.14, "test-clean": 2.03, "test-other": 4.28}
    train_ls_base("base-ls-newrtrn", prefix=prefix, config_updates_extra={"_meta_hash_trigger": "new-returnn-2026-06"})
    # base-ls-muon: muon-lr5e3-wdbl on the single-GPU log-mel baseline (best FZJ optimizer setting, at RZ).
    # {"dev-clean": 1.91, "dev-other": 4.28, "test-clean": 2.09, "test-other": 4.46}
    # muon ~= AdamW single-GPU (newrtrn 4.14 / newtorch 4.26): no single-GPU gain (a 4-GPU-regime fix).
    train_ls_base(
        "base-ls-muon",
        prefix=prefix,
        base_lr=1.0,
        peak_lr=5e-3,
        config_updates_extra={"optimizer.class": rf.build_dict(Muon)["class"]},
        config_deletes_extra=["optimizer.epsilon"],
    )
    # --- TTS-encoder single-GPU runs ---
    # Reference-match TTS-encoder, SINGLE-GPU / nep=100 (the regime of Nick's 3.53 reference).
    # Identical synth / data to the FZJ tts-enc-ref-match; only num_processes=1 / nep=100 differ.
    # {"dev-clean": 1.71, "dev-other": 4.17, "test-clean": 1.88, "test-other": 4.40}  (base-ls-dbmel 4.19 -> ~0 text gain)
    _train_tts_encoder(
        "tts-enc-dbmel-plain-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        # reduced from 120k/25k: 80GB c25g GPUs OOM with real TTS durations
        # (audio batches set the memory high-water mark); 100k = the offline reference batch.
        batch_size_audio_frames=90_000,  # 100k peaked ~78GB in the long laplace buckets -> OOM at 80GB
        # 15k still OOMs on 80GB for this variant: the audio-step pool (~53GB at audio 100k)
        # plus the text-step pool must fit together; 10k keeps the text step small enough.
        batch_size_phon=4_000,  # synth is ~167ms audio/phoneme (incl silence), so 8k phon -> ~21M samples,
        # 1.47x the 90k-frame audio cap -> OOM at 80GB in the long-phon buckets;
        # 4k keeps text-step encoder load <= audio-step load
        max_phon_len=300,
        tts_waveform=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Faithful reference match: same as ref-match-1gpu but the LOG-MEL ASR front-end (asr_logmel=True).
    # Nick's reference re-extracts at log-mel,
    # so ref-match-1gpu (DbMel default) had a front-end mismatch.
    # tts_waveform_peak_norm=True is required on the log-mel+waveform path
    # (raw log-mel + feature_batch_norm NaNs without peak-normalizing the injected waveform;
    # the DbMel path is fixed-norm and immune).
    _train_tts_encoder(
        "tts-enc-logmel-plain-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        # reduced from 120k/25k: 80GB c25g GPUs OOM with real TTS durations
        # (audio batches set the memory high-water mark); 100k = the offline reference batch.
        batch_size_audio_frames=90_000,  # 100k peaked ~78GB in the long laplace buckets -> OOM at 80GB
        # 15k still OOMs on 80GB for this variant: the audio-step pool (~53GB at audio 100k)
        # plus the text-step pool must fit together; 10k keeps the text step small enough.
        batch_size_phon=4_000,  # synth is ~167ms audio/phoneme (incl silence), so 8k phon -> ~21M samples,
        # 1.47x the 90k-frame audio cap -> OOM at 80GB in the long-phon buckets;
        # 4k keeps text-step encoder load <= audio-step load
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # FULL reference match: ref-match-logmel-1gpu + the remaining config diffs vs the offline-TTS
    # reference (full config diff 2026-07-13): SamplingBPE(0.01) train targets (bpeSample001, both streams),
    # shared no-bias aux CTC heads (auxShared/auxNoBias), random train audio padding (AudioPadRnd100).
    # Remaining known diffs vs the reference: online-vs-offline TTS mixing (that is the research question)
    # and behavior_version/torch (judged irrelevant).
    # Base for the mixing experiments (single-stream, gumbel-interleave).
    _train_tts_encoder(
        "tts-enc-logmel-refcfg-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        # reduced from 120k/25k: 80GB c25g GPUs OOM with real TTS durations
        # (audio batches set the memory high-water mark); 100k = the offline reference batch.
        batch_size_audio_frames=90_000,  # 100k peaked ~78GB in the long laplace buckets -> OOM at 80GB
        # 15k still OOMs on 80GB for this variant: the audio-step pool (~53GB at audio 100k)
        # plus the text-step pool must fit together; 10k keeps the text step small enough.
        batch_size_phon=4_000,  # synth is ~167ms audio/phoneme (incl silence), so 8k phon -> ~21M samples,
        # 1.47x the 90k-frame audio cap -> OOM at 80GB in the long-phon buckets;
        # 4k keeps text-step encoder load <= audio-step load
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Mixing experiment 1 (single-stream): the full reference match, but real-audio + TTS-synth MIXED
    # in ONE batch (no alternate_batching, one loss set) via aed_glowtts_single_stream_train_step --
    # the offline-TTS reference's batch structure with our online TTS.
    # Smoke-tested 2026-07-14 (mixed batches, losses sane, ~75GB peak, ~2.3 s/step incl per-step TTS+GL).
    _train_tts_encoder(
        "tts-enc-logmel-refcfg-single-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        # reduced from 120k/25k: 80GB c25g GPUs OOM with real TTS durations
        # (audio batches set the memory high-water mark); 100k = the offline reference batch.
        batch_size_audio_frames=90_000,  # 100k peaked ~78GB in the long laplace buckets -> OOM at 80GB
        # 15k still OOMs on 80GB for this variant: the audio-step pool (~53GB at audio 100k)
        # plus the text-step pool must fit together; 10k keeps the text step small enough.
        batch_size_phon=4_000,  # synth is ~167ms audio/phoneme (incl silence), so 8k phon -> ~21M samples,
        # 1.47x the 90k-frame audio cap -> OOM at 80GB in the long-phon buckets;
        # 4k keeps text-step encoder load <= audio-step load
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        single_stream=True,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Mixing experiment 2 (+gumbel): single-stream + Gumbel-max softened interleave
    # (RETURNN CombinedDataset interleave_gumbel_scale: samples the next dataset
    # from softmax(log(expected_remaining_seqs) / scale)).
    # Constant scale 1.0 = pick proportional to the remaining seqs = sampling-without-replacement,
    # i.e. EXACTLY a uniform random shuffle of the audio+text union -- the offline reference's ordering
    # (balance + joint finish are automatic; the annealed "remaining" schedule stays an unused option).
    _train_tts_encoder(
        "tts-enc-logmel-refcfg-single-gumbel-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        # reduced from 120k/25k: 80GB c25g GPUs OOM with real TTS durations
        # (audio batches set the memory high-water mark); 100k = the offline reference batch.
        batch_size_audio_frames=90_000,  # 100k peaked ~78GB in the long laplace buckets -> OOM at 80GB
        # 15k OOMs at 80GB once batches truly mix audio+text (the earlier 64.9GB reading
        # was from the audio-only sampling-bug phase); 10k like the single-stream sibling.
        batch_size_phon=4_000,  # synth is ~167ms audio/phoneme (incl silence), so 8k phon -> ~21M samples,
        # 1.47x the 90k-frame audio cap -> OOM at 80GB in the long-phon buckets;
        # 4k keeps text-step encoder load <= audio-step load
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        single_stream=True,
        interleave_gumbel_scale=1.0,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Full base, DbMel DIRECT: TTS log-mel fed straight into the DbMel ASR front-end space
    # (no GL-net, no Griffin-Lim, no waveform round-trip; the cheapest TTS path).
    _train_tts_encoder(
        "tts-enc-dbmel-direct-refcfg-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        # reduced from 120k/25k: 80GB c25g GPUs OOM with real TTS durations
        # (audio batches set the memory high-water mark); 100k = the offline reference batch.
        batch_size_audio_frames=90_000,  # 100k peaked ~78GB in the long laplace buckets -> OOM at 80GB
        # 15k still OOMs on 80GB for this variant: the audio-step pool (~53GB at audio 100k)
        # plus the text-step pool must fit together; 10k keeps the text step small enough.
        batch_size_phon=4_000,  # synth is ~167ms audio/phoneme (incl silence), so 8k phon -> ~21M samples,
        # 1.47x the 90k-frame audio cap -> OOM at 80GB in the long-phon buckets;
        # 4k keeps text-step encoder load <= audio-step load
        max_phon_len=300,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Full base + SHORT random durations: w = w_pred * U(0.2, 0.5) per phoneme
    # (ceil keeps >= 1 frame; ~2-3 frames/phoneme, close to what the preload bug accidentally ran,
    # but with REAL acoustics -- tests whether compressed synth speech suffices for text-util).
    _train_tts_encoder(
        "tts-enc-logmel-refcfg-rnddur-short-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        # reduced from 120k/25k: 80GB c25g GPUs OOM with real TTS durations
        # (audio batches set the memory high-water mark); 100k = the offline reference batch.
        batch_size_audio_frames=100_000,
        batch_size_phon=15_000,
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        glow_tts_random_durations_jitter_mult=(0.2, 0.5),
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Full base + FIXED 1 frame per phoneme (no duration predictor):
    # the extreme duration ablation, bridging to the pseudo-enc label-dur-1 regime with REAL acoustics.
    _train_tts_encoder(
        "tts-enc-logmel-refcfg-dur1-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        # reduced from 120k/25k: 80GB c25g GPUs OOM with real TTS durations
        # (audio batches set the memory high-water mark); 100k = the offline reference batch.
        batch_size_audio_frames=100_000,
        batch_size_phon=15_000,
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        glow_tts_fixed_duration=1,
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        enc_aux_logits_share_weights=True,
        enc_aux_logits_with_bias=False,
        pad_audio_rnd=100,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Best FZJ mechanism (layer-split pseudo-enc at enc-layer 4 = 3.79 dev-other with muon-nep38 4-GPU),
    # run in the reference's OWN regime: single-GPU / nep=100 / AdamW (NO muon),
    # so it is directly comparable to Nick's 3.53 offline-TTS baseline (same optimizer + regime).
    # Tests whether our layer-split injection beats the reference apples-to-apples.
    # NB muon lifted layer4 a lot on FZJ (nep25 AdamW 4.24 -> muon 3.79),
    # so this AdamW run will likely land above 3.79;
    # the muon version stays the FZJ headline.
    _train_tts_encoder(
        "pseudo-enc-layer4-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        batch_size_phon=8_000,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_start_layer=4,
        pseudo_enc_specaug_max_width=6,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Same as pseudo-enc-layer4-1gpu but NO blanks (pseudo_enc_blank_duration_range=(0, 0)):
    # at layer8 the no-blank / single-blank variants beat interleaved blanks by ~0.15
    # (FZJ muon-nep38: 3.95/3.93 vs 4.11),
    # so applying that at the best depth (layer4) in the reference regime is our most promising unrun config.
    _train_tts_encoder(
        "pseudo-enc-layer4-noblank-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        batch_size_phon=8_000,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_start_layer=4,
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Muon version of pseudo-enc-layer4-1gpu (muon-lr5e3-wdbl):
    # best FZJ mechanism + best optimizer in the single-GPU nep100 regime,
    # the best-number chase, vs the AdamW #1 which is the fair 3.53 comparison.
    _train_tts_encoder(
        "pseudo-enc-layer4-muon-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        batch_size_phon=8_000,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_start_layer=4,
        pseudo_enc_specaug_max_width=6,
        num_processes=1,
        gpu_mem=96,
        nep=100,
        base_lr=1.0,
        peak_lr=5e-3,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
    # Same as pseudo-enc-layer4-noblank-1gpu but batch_size_phon 8k -> 25k (ref-match's default):
    # noblank is ~1.0 enc frame/phoneme (< ref-match's ~1.33),
    # so the 8k throttle (a blanket OOM workaround for the blank-heavy variants) is unnecessary here.
    # 25k halves the steps (~7k, like ref-match) and matches ref-match's text batch,
    # for a clean + fast layer-split-vs-front-end comparison.
    _train_tts_encoder(
        "pseudo-enc-layer4-noblank-25kphon-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        batch_size_phon=25_000,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_start_layer=4,
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
