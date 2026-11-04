"""OOV graph checks and CPU regressions; no installed model/GPU required."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sisyphus import tk
from sisyphus.hash import sis_hash_helper
from i6_core.util import write_xml

from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_decoder_smoke import smoke_lexicon
from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_diagnostics import (
    audit_config,
    ctc_viterbi_alignment,
    lexical_piece_statistics,
    read_lexicon,
    read_tracebacks,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_extended_lexicon import (
    ExtendGraniteOovLexiconJob,
    RankOovLexiconCandidatesJob,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import oov_length
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.oov_guard import (
    GuardedOovTracebackFormatter,
)
from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.recog_rasr_config import (
    TreeTimesyncRecogParams,
    get_tree_timesync_recog_config,
)


class OovLengthTest(unittest.TestCase):
    def test_approach_grids_and_guards(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.lm import (
            ArpaLmParams,
            TransformerLmParams,
        )

        lexicons = {
            name: tk.Path(name) for name in ("closed", "A-100k", "A-300k", "A-full", "B", "C-100k", "C-300k", "C-full")
        }
        conditions = {
            "none": None,
            "4gram-lm0.3": ArpaLmParams(scale=0.3),
            "trafo-lm0.3": TransformerLmParams(scale=0.3),
        }
        variants = oov_length.length_variants(
            "granite", lexicons, tk.Path("binary"), lm_conditions=conditions, repo=tk.Path("repo")
        )
        self.assertEqual(len(variants), 48)
        for variant in variants:
            params = variant.search_algorithm_params
            self.assertEqual(variant.rasr_binary_path_override, tk.Path("binary"))
            self.assertEqual(params.unknown_word_penalty, 0)
            self.assertEqual(params.word_end_score_threshold, 16)
            if variant.prepared_lexicon_file in (
                lexicons["closed"],
                *(lexicons[f"A-{size}"] for size in ("100k", "300k", "full")),
            ):
                self.assertIsNone(params.unknown_word_fallback)
                self.assertIsNone(params.unknown_piece_penalty)
            else:
                self.assertEqual(params.unknown_piece_penalty, 0.0)
                self.assertIsNotNone(variant.prototype_traceback_formatter_serializers)

    def test_identical_beta_axis_for_bpe_modes(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.lm import ArpaLmParams

        for kind in ("bpe", "transducer"):
            variants = oov_length.length_variants(
                kind,
                {k: tk.Path("shared-lexicon") for k in ("closed", "legacy", "known-excluding")},
                tk.Path("binary"),
                lm_conditions={"4gram-lm0.6": ArpaLmParams(scale=0.6)},
                alpha0_by_lm={"4gram-lm0.6": 1.0},
                phase="sweep",
                betas=(0.0, 10.0, 20.0),
            )
            for mode in ("legacy", "known-excluding"):
                matches = [v for v in variants if v.search_algorithm_params.unknown_word_fallback == mode]
                self.assertEqual({v.search_algorithm_params.unknown_word_penalty for v in matches}, {0, 10, 20})
                self.assertTrue(all(v.prototype_traceback_formatter_serializers for v in matches))
                self.assertTrue(
                    all(v.search_algorithm_params.collapse_repeated_labels == (kind == "bpe") for v in matches)
                )

    def test_calibration_required(self):
        with self.assertRaises(KeyError):
            oov_length.length_variants(
                "bpe",
                {"known-excluding": tk.Path("lex")},
                tk.Path("binary"),
                lm_conditions={"uncalibrated": None},
                alpha0_by_lm={},
                repo=tk.Path("repo"),
            )

    def test_preflight_separate_build(self):
        jobs = oov_length.run_preflight()
        clone = jobs["binary"].creator.source_folder.creator
        self.assertEqual(clone.commit, "6e07a2e503fb18cc4fcddc89fa01e2d9fbd4a5d9")
        self.assertIn("unknown-word-start+nonword", clone.patches[0])
        self.assertIsNone(jobs["binary"].hash_overwrite)
        self.assertEqual(len(jobs["correctness"].cases), 34)
        self.assertEqual(len(jobs["delayed_penalty_probe"].configs), 4)
        compile(jobs["delayed_penalty_probe"].runner, "probe", "exec")
        self.assertEqual(jobs["pruning"].probe_report, jobs["delayed_penalty_probe"].out_report)
        self.assertEqual(jobs["pruning"].first_piece_report, jobs["first_piece_probe"].out_report)
        self.assertEqual(len(jobs["first_piece_probe"].cases), 8)
        compile(jobs["first_piece_probe"].runner, "first-piece-probe", "exec")
        self.assertIn('scores = np.full((3, case["vocab_size"])', jobs["first_piece_probe"].runner)
        for case in jobs["first_piece_probe"].cases:
            self.assertEqual(case["expected_score"], 20.0)
            self.assertEqual(case["expected_text"], "DOG")
            cfg = case["config"].creator.config.lib_rasr
            self.assertEqual(cfg.acoustic_model.state_tying.type, "monophone")
            self.assertEqual(cfg.search_algorithm.unknown_word_penalty, 100.0)
            self.assertEqual(cfg.search_algorithm.unknown_piece_penalty, 0.0)
        self.assertNotEqual(jobs["margins"].correctness_report, jobs["correctness"].out_report)
        self.assertIn("GraniteWordStartMarginJob.RiigwjieBGmV", jobs["margins"].out_report.get_path())
        self.assertEqual(jobs["state_tying"].decoder_smoke_report, jobs["correctness"].out_report)
        dump = jobs["state_tying"].state_tying.creator
        self.assertEqual(dump.config.allophone_tool.acoustic_model.state_tying.type, "monophone")
        self.assertFalse(jobs["margins"].data_config.peak_normalization)
        self.assertIsNone(jobs["margins"].data_config.preemphasis)
        cases = {c["name"]: c for c in jobs["correctness"].cases}
        self.assertEqual(cases["bpe-alpha-unknown"]["expected_score"], 5)
        self.assertEqual(cases["granite-alpha-known-refund"]["expected_score"], 0)
        self.assertEqual(cases["granite-extended-unk-refund"]["expected_score"], 2)
        self.assertEqual(cases["bpe-legacy-beta"]["expected_score"], 2)

    def test_granite_sweep_defaults_to_zero_alpha(self):
        kwargs = dict(
            kind="granite",
            lexicons={"B": tk.Path("lex")},
            binary=tk.Path("binary"),
            lm_conditions={"4gram": None},
            phase="sweep",
            betas=(0.0, 10.0, 100.0),
            repo=tk.Path("repo"),
        )
        variants = oov_length.length_variants(**kwargs)
        self.assertEqual(len(variants), 3)
        self.assertEqual({v.search_algorithm_params.unknown_piece_penalty for v in variants}, {0.0})
        with self.assertRaisesRegex(ValueError, "Use explicit alphas for Granite"):
            oov_length.length_variants(**kwargs, alpha0_by_lm={"4gram": 0.327})

    def test_fallback_lexicons_depend_on_pruning_acceptance(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_extended_lexicon import PrepareOovLengthLexiconJob

        preflight = oov_length.run_preflight()
        _, lexicons = oov_length.setup_granite_lexicons(preflight, approaches=("closed", "A-100k", "B", "C-100k"))
        for name in ("B", "C-100k"):
            gate = lexicons[name].creator.lexicon.creator
            self.assertIsInstance(gate, PrepareOovLengthLexiconJob)
            self.assertEqual(gate.pruning_report, preflight["pruning"].out_report)
        self.assertIsNone(lexicons["closed"].creator.lexicon.creator.pruning_report)
        self.assertIsNone(lexicons["A-100k"].creator.lexicon.creator.lexicon.creator.pruning_report)

    def test_approach_a_entrypoint_excludes_fallback(self):
        with patch.object(oov_length, "run_granite_phase", return_value=[]) as run:
            self.assertEqual(oov_length.run_approach_a(), [])
        self.assertEqual(run.call_args.kwargs["approaches"], ("closed", "A-100k"))
        self.assertEqual(run.call_args.kwargs["lm_conditions"]["4gram-lm0.3"].scale, 0.3)

    def test_pruning_acceptance_requires_all_four_cases(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_length_smoke import RequireOovPruningProbeJob

        with tempfile.TemporaryDirectory() as d:
            report, out = Path(d) / "probe.json", Path(d) / "accepted.json"
            results = {
                name: dict(correct=True, cost=35.0)
                for name in ("closed", "open-narrow", "open-large-beam", "open-wide-threshold")
            }
            report.write_text(json.dumps(dict(results=results)))
            job = RequireOovPruningProbeJob(tk.Path(str(report)))
            job.out_report = tk.Path(str(out))
            job.run()
            self.assertTrue(json.loads(out.read_text())["passed"])
            for name in results:
                broken = {k: dict(v) for k, v in results.items()}
                broken[name] = dict(correct=False, cost=None)
                report.write_text(json.dumps(dict(results=broken)))
                with self.assertRaisesRegex(ValueError, "Pruning regression"):
                    job.run()

    def test_limited_fallback_grid(self):
        for kind, modes, expected_betas, expected_scales, count in (
            ("bpe", ("closed", "known-excluding"), {10, 12, 20, 100}, {0.6}, 5),
            ("granite", ("B",), {0, 2, 4, 8, 16, 100}, {None, 0.3}, 12),
        ):
            variants = oov_length.limited_fallback_variants(
                kind, {m: tk.Path("lex") for m in modes}, tk.Path("binary"), repo=tk.Path("repo")
            )
            self.assertEqual(len(variants), count)
            self.assertEqual(len({v.descriptor for v in variants}), count)
            fallback = [v for v in variants if v.search_algorithm_params.unknown_word_fallback == "known-excluding"]
            self.assertEqual({v.search_algorithm_params.unknown_word_penalty for v in fallback}, expected_betas)
            self.assertEqual({v.search_algorithm_params.unknown_piece_penalty for v in fallback}, {0})
            self.assertEqual(
                {
                    v.search_algorithm_params.word_lm_params.scale if v.search_algorithm_params.word_lm_params else None
                    for v in fallback
                },
                expected_scales,
            )
            for variant in variants:
                params = variant.search_algorithm_params
                self.assertEqual(params.max_beam_sizes, [256])
                self.assertEqual(params.score_thresholds, [16.0])
                self.assertEqual(params.max_word_end_beam_size, 256)
                self.assertEqual(params.word_end_score_threshold, 16.0)
                self.assertFalse(params.sentence_end_fallback)
                self.assertEqual(params.recombination_mode, "on")
                self.assertEqual(variant.prior_scale, 0)
                self.assertEqual(variant.blank_penalty, 0)
                if variant in fallback:
                    self.assertIsNotNone(variant.prototype_traceback_formatter_serializers)

    def test_combined_pruning_acceptance_rejects_first_piece_failure(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_length_smoke import RequireOovPruningProbeJob

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            budgets = ("closed", "open-narrow", "open-large-beam", "open-wide-threshold")
            (p / "delayed.json").write_text(
                json.dumps(
                    dict(
                        results={name: dict(correct=True, cost=35.0) for name in budgets},
                    )
                )
            )
            first = dict(
                passed=True,
                num_cases=8,
                results=[
                    dict(case=dict(name=f"{kind}-{name}"), score=20.0, text="DOG")
                    for kind in ("bpe", "granite")
                    for name in budgets
                ],
            )
            (p / "first.json").write_text(json.dumps(first))
            job = RequireOovPruningProbeJob(tk.Path(str(p / "delayed.json")), tk.Path(str(p / "first.json")))
            old_job = RequireOovPruningProbeJob(tk.Path(str(p / "delayed.json")))
            self.assertNotEqual(job.out_report.get_path(), old_job.out_report.get_path())
            job.out_report = tk.Path(str(p / "accepted.json"))
            job.run()
            self.assertTrue(json.loads((p / "accepted.json").read_text())["passed"])
            for index in (1, 5):  # Narrow open, both tokenizations; old probe still passes.
                broken = json.loads(json.dumps(first))
                broken["results"][index].update(score=100.0, text="X")
                (p / "first.json").write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError, "First-piece pruning regression"):
                    job.run()
            first["results"].pop()
            (p / "first.json").write_text(json.dumps(first))
            with self.assertRaisesRegex(ValueError, "Incomplete or failed"):
                job.run()

    def test_beta_only_rejects_nonzero_alpha(self):
        with self.assertRaisesRegex(ValueError, "beta-only phase fixes alpha=0"):
            oov_length.length_variants(
                "bpe",
                {"known-excluding": tk.Path("lex")},
                tk.Path("binary"),
                lm_conditions={"4gram": None},
                alpha0_by_lm={"4gram": 1.0},
                phase="beta-only",
            )

    def test_limited_entrypoint_is_gated_and_dev_only(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.recognition import (
            ctc_bpe,
            ctc_granite,
        )

        model = SimpleNamespace(model_config=SimpleNamespace(target_size=185))
        with (
            patch.object(ctc_bpe, "run", side_effect=lambda *args, **kwargs: list(kwargs["variants"])) as bpe_run,
            patch.object(ctc_granite, "run", side_effect=lambda **kwargs: list(kwargs["variants"])) as granite_run,
        ):
            variants = oov_length.run_limited_fallback(model)
        self.assertEqual(len(variants), 17)
        self.assertEqual(bpe_run.call_args.kwargs["corpora"], ["dev-other"])
        self.assertEqual(granite_run.call_args.kwargs["corpora"], ["dev-other"])
        bpe = bpe_run.call_args.kwargs["variants"]
        self.assertEqual(len({v.prepared_lexicon_file for v in bpe}), 1)
        self.assertTrue(all(v.descriptor.endswith("_lexicon-baseline") for v in bpe))
        self.assertEqual(sum(v.search_algorithm_params.unknown_word_fallback == "disabled" for v in bpe), 1)
        pruning = oov_length.run_preflight()["pruning"].out_report
        for variant in variants:
            gate = variant.prepared_lexicon_file.creator.lexicon.creator
            self.assertEqual(gate.pruning_report, pruning)

    def test_tuning_grid_and_reused_points(self):
        for kind, modes, total, added in (
            ("bpe", ("known-excluding",), 25, 23),
            ("granite", ("B",), 11, 7),
        ):
            kwargs = dict(
                kind=kind, lexicons={m: tk.Path("lex") for m in modes}, binary=tk.Path("binary"), repo=tk.Path("repo")
            )
            full = oov_length.tuning_variants(**kwargs, include_completed=True)
            new = oov_length.tuning_variants(**kwargs)
            old = {v.descriptor: v for v in oov_length.limited_fallback_variants(**kwargs)}
            self.assertEqual(len(full), total)
            self.assertEqual(len(new), added)
            self.assertEqual(len({v.descriptor for v in full}), total)
            self.assertFalse(set(old) & {v.descriptor for v in new})
            for variant in full:
                params = variant.search_algorithm_params
                self.assertEqual(params.max_beam_sizes, [256])
                self.assertEqual(params.score_thresholds, [16.0])
                self.assertEqual(params.word_end_score_threshold, 16.0)
                self.assertEqual(params.unknown_word_fallback, "known-excluding")
                self.assertFalse(params.sentence_end_fallback)
                self.assertEqual(variant.blank_penalty, 0.0)
                self.assertEqual(variant.prior_scale, 0.0)
                self.assertIsNotNone(variant.prototype_traceback_formatter_serializers)
                if variant.descriptor in old:
                    self.assertEqual(params, old[variant.descriptor].search_algorithm_params)
                    self.assertEqual(variant.prepared_lexicon_file, old[variant.descriptor].prepared_lexicon_file)
            if kind == "bpe":
                self.assertEqual({v.search_algorithm_params.unknown_word_penalty for v in full}, {6, 8, 10, 12, 14})
                self.assertEqual({v.search_algorithm_params.unknown_piece_penalty for v in full}, {0, 0.25, 0.5, 1, 2})
                self.assertTrue(all(v.search_algorithm_params.word_lm_params.scale == 0.6 for v in full))
            else:
                self.assertTrue(all(v.search_algorithm_params.unknown_piece_penalty == 0 for v in full))
                for scale, betas in ((None, {1, 2, 3, 4, 5, 6}), (0.3, {4, 6, 8, 10, 12})):
                    self.assertEqual(
                        {
                            v.search_algorithm_params.unknown_word_penalty
                            for v in full
                            if (
                                v.search_algorithm_params.word_lm_params.scale
                                if v.search_algorithm_params.word_lm_params
                                else None
                            )
                            == scale
                        },
                        betas,
                    )

    def test_explicit_alpha_grid_validation(self):
        kwargs = dict(
            kind="bpe",
            lexicons={"known-excluding": tk.Path("lex")},
            binary=tk.Path("binary"),
            lm_conditions={"lm": None},
            phase="sweep",
        )
        for alphas in ((), (-1,), (float("nan"),), (float("inf"),)):
            with self.assertRaises(ValueError):
                oov_length.length_variants(**kwargs, alphas=alphas)
        with self.assertRaises(ValueError):
            oov_length.length_variants(**kwargs, alphas=(0, 1), alpha0_by_lm={"lm": 1})

    def test_tuning_entrypoint_is_gated_dev_only_and_shares_lexicon(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.recognition import (
            ctc_bpe,
            ctc_granite,
        )

        model = SimpleNamespace(model_config=SimpleNamespace(target_size=185))
        with (
            patch.object(ctc_bpe, "run", side_effect=lambda *args, **kwargs: list(kwargs["variants"])) as bpe_run,
            patch.object(ctc_granite, "run", side_effect=lambda **kwargs: list(kwargs["variants"])) as granite_run,
        ):
            old = oov_length.run_limited_fallback(model)
            new = oov_length.run_tuning(model)
        self.assertEqual(len(new), 30)
        self.assertEqual(len(bpe_run.call_args.kwargs["variants"]), 23)
        self.assertEqual(len(granite_run.call_args.kwargs["variants"]), 7)
        self.assertEqual(bpe_run.call_args.kwargs["corpora"], ["dev-other"])
        self.assertEqual(granite_run.call_args.kwargs["corpora"], ["dev-other"])
        self.assertFalse({v.descriptor for v in new} & {v.descriptor for v in old})
        self.assertEqual({v.prepared_lexicon_file for v in new}, {v.prepared_lexicon_file for v in old})
        preflight = oov_length.run_preflight()
        for variant in new:
            self.assertEqual(variant.rasr_binary_path_override, preflight["binary"])
            self.assertEqual(
                variant.prepared_lexicon_file.creator.lexicon.creator.pruning_report, preflight["pruning"].out_report
            )

    def test_granite_alpha_grid_and_reused_controls(self):
        from sisyphus.hash import sis_hash_helper

        kwargs = dict(lexicons={"B": tk.Path("lex")}, binary=tk.Path("binary"), repo=tk.Path("repo"))
        full = oov_length.granite_alpha_tuning_variants(**kwargs, include_completed=True)
        new = oov_length.granite_alpha_tuning_variants(**kwargs)
        old = {
            v.descriptor: v
            for v in (
                oov_length.limited_fallback_variants("granite", **kwargs)
                + oov_length.tuning_variants("granite", **kwargs, include_completed=True)
            )
        }
        self.assertEqual(len(full), 50)
        self.assertEqual(len(new), 40)
        self.assertEqual(len({v.descriptor for v in full}), 50)
        self.assertFalse(set(old) & {v.descriptor for v in new})
        for scale, betas in ((None, {0, 1, 2, 3, 4}), (0.3, {0, 2, 4, 6, 8})):
            self.assertEqual(
                {
                    (p.unknown_word_penalty, p.unknown_piece_penalty)
                    for v in full
                    for p in [v.search_algorithm_params]
                    if (p.word_lm_params.scale if p.word_lm_params else None) == scale
                },
                {(beta, alpha) for beta in betas for alpha in (0, 0.25, 0.5, 1, 2)},
            )
        for v in full:
            p = v.search_algorithm_params
            self.assertEqual(p.max_beam_sizes, [256])
            self.assertEqual(p.max_word_end_beam_size, 256)
            self.assertEqual(p.score_thresholds, [16.0])
            self.assertEqual(p.word_end_score_threshold, 16.0)
            self.assertEqual(p.unknown_word_fallback, "known-excluding")
            self.assertEqual(p.unknown_word_tokenization, "word-start-marked")
            self.assertFalse(p.sentence_end_fallback)
            self.assertEqual(v.blank_penalty, 0)
            self.assertEqual(v.prior_scale, 0)
            self.assertIsNotNone(v.prototype_traceback_formatter_serializers)
            if v.descriptor in old:
                self.assertEqual(sis_hash_helper(vars(v)), sis_hash_helper(vars(old[v.descriptor])))
            else:
                self.assertGreater(p.unknown_piece_penalty, 0)

    def test_granite_alpha_entrypoint_reuses_gated_lexicon_and_is_dev_only(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.recognition import ctc_granite

        preflight = oov_length.run_preflight()
        repo, lexicons = oov_length.setup_granite_lexicons(preflight, approaches=("B",))
        with patch.object(ctc_granite, "run", side_effect=lambda **kwargs: list(kwargs["variants"])) as run:
            variants = oov_length.run_granite_alpha_tuning()
        self.assertEqual(len(variants), 40)
        self.assertEqual(run.call_args.kwargs["corpora"], ["dev-other"])
        self.assertEqual(run.call_args.kwargs["huggingface_repo_dir"], repo)
        for v in variants:
            self.assertEqual(v.rasr_binary_path_override, preflight["binary"])
            self.assertEqual(v.prepared_lexicon_file, lexicons["B"])
            self.assertEqual(
                v.prepared_lexicon_file.creator.lexicon.creator.pruning_report, preflight["pruning"].out_report
            )

    def test_granite_beam_grid_and_reused_controls(self):
        from dataclasses import replace
        from sisyphus.hash import sis_hash_helper

        kwargs = dict(lexicons={"B": tk.Path("lex")}, binary=tk.Path("binary"), repo=tk.Path("repo"))
        full = oov_length.granite_beam_tuning_variants(**kwargs, include_completed=True)
        new = oov_length.granite_beam_tuning_variants(**kwargs)
        old = {v.descriptor: v for v in oov_length.granite_alpha_tuning_variants(**kwargs, include_completed=True)}
        self.assertEqual(len(full), 24)
        self.assertEqual(len(new), 22)
        self.assertEqual(len({v.descriptor for v in full}), 24)
        self.assertFalse(set(old) & {v.descriptor for v in new})
        for lm_name, beta in (("no-lm", 1), ("4gram-lm0.3", 6)):
            group = [v for v in full if f"_B_{lm_name}_" in v.descriptor]
            control = old[f"oov-{oov_length.REVISION}_B_{lm_name}_beta{beta}_alpha2_beam256_thr16"]
            self.assertEqual(
                {
                    (v.search_algorithm_params.max_beam_sizes[0], v.search_algorithm_params.score_thresholds[0])
                    for v in group
                },
                {(b, t) for b in (64, 256, 1024) for t in (8, 16, 32, 64)},
            )
            for v in group:
                p = v.search_algorithm_params
                self.assertEqual(p.max_word_end_beam_size, p.max_beam_sizes[0])
                self.assertEqual(p.word_end_score_threshold, p.score_thresholds[0])
                reverted = replace(
                    v,
                    descriptor=control.descriptor,
                    search_algorithm_params=replace(
                        p,
                        max_beam_sizes=[256],
                        max_word_end_beam_size=256,
                        score_thresholds=[16.0],
                        word_end_score_threshold=16.0,
                    ),
                )
                self.assertEqual(sis_hash_helper(vars(reverted)), sis_hash_helper(vars(control)))

    def test_granite_beam_entrypoint_is_gated_dev_only(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.recognition import ctc_granite

        preflight = oov_length.run_preflight()
        with patch.object(ctc_granite, "run", side_effect=lambda **kwargs: list(kwargs["variants"])) as run:
            variants = oov_length.run_granite_beam_tuning()
        self.assertEqual(len(variants), 22)
        self.assertEqual(run.call_args.kwargs["corpora"], ["dev-other"])
        for v in variants:
            self.assertEqual(v.rasr_binary_path_override, preflight["binary"])
            self.assertIsNotNone(v.prototype_traceback_formatter_serializers)
            self.assertEqual(
                v.prepared_lexicon_file.creator.lexicon.creator.pruning_report, preflight["pruning"].out_report
            )

    def test_granite_transformer_grid(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.lm import TransformerLmParams

        params = TransformerLmParams(
            onnx_model=tk.Path("model"), vocab_file=tk.Path("validated-vocab"), use_gpu=True, use_kv_cache=False
        )
        variants = oov_length.granite_transformer_tuning_variants(
            {"closed": tk.Path("closed"), "B": tk.Path("open")},
            tk.Path("binary"),
            repo=tk.Path("repo"),
            lm_params=params,
        )
        self.assertEqual(len(variants), 18)
        self.assertEqual(len({v.descriptor for v in variants}), 18)
        for scale in (0.3, 0.5):
            group = [v for v in variants if v.search_algorithm_params.word_lm_params.scale == scale]
            closed = [v for v in group if v.search_algorithm_params.unknown_word_fallback == "disabled"]
            self.assertEqual(len(closed), 1)
            self.assertIsNone(closed[0].prototype_traceback_formatter_serializers)
            self.assertEqual(closed[0].search_algorithm_params.unknown_word_penalty, 0)
            self.assertEqual(
                {
                    (v.search_algorithm_params.unknown_word_penalty, v.search_algorithm_params.unknown_piece_penalty)
                    for v in group
                    if v.search_algorithm_params.unknown_word_fallback == "known-excluding"
                },
                {(b, a) for b in (0, 4, 8, 12) for a in (0, 2)},
            )
        for v in variants:
            p = v.search_algorithm_params
            self.assertEqual(p.max_beam_sizes, [256])
            self.assertEqual(p.max_word_end_beam_size, 256)
            self.assertEqual(p.score_thresholds, [16.0])
            self.assertEqual(p.word_end_score_threshold, 16)
            self.assertFalse(p.sentence_end_fallback)
            self.assertEqual(p.word_lm_params.onnx_model, params.onnx_model)
            self.assertEqual(p.word_lm_params.vocab_file, params.vocab_file)
            self.assertFalse(p.word_lm_params.use_kv_cache)
            self.assertTrue(p.word_lm_params.use_gpu)
            self.assertTrue(v.descriptor.endswith("_gpu-stateless"))
            self.assertEqual(v.search_mode_params.gpu_mem_rqmt, 24)
            self.assertEqual(v.prior_scale, 0)
            self.assertEqual(v.blank_penalty, 0)

    def test_transformer_entrypoint_and_retained_references(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.recognition import (
            ctc_bpe,
            ctc_granite,
        )
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_neural_lm import ValidateOovNeuralLmJob

        model = SimpleNamespace(model_config=SimpleNamespace(target_size=185))
        with (
            patch.object(ctc_bpe, "run", side_effect=lambda *args, **kwargs: list(kwargs["variants"])) as bpe_run,
            patch.object(ctc_granite, "run", side_effect=lambda **kwargs: list(kwargs["variants"])) as run,
        ):
            refs = oov_length.run_retained_comparisons(model)
            new = oov_length.run_granite_transformer_tuning()
        self.assertEqual(len(refs), 9)
        self.assertEqual(len(bpe_run.call_args.kwargs["variants"]), 2)
        self.assertEqual(len(new), 18)
        self.assertEqual(run.call_args.kwargs["corpora"], ["dev-other"])
        self.assertFalse({v.descriptor for v in refs} & {v.descriptor for v in new})
        self.assertTrue(all("thr64" not in v.descriptor for v in refs + new))
        validation_jobs = {v.search_algorithm_params.word_lm_params.vocab_file.creator for v in new}
        self.assertEqual(len(validation_jobs), 1)
        (job,) = validation_jobs
        self.assertIsInstance(job, ValidateOovNeuralLmJob)
        self.assertEqual(set(job.lexicons), {v.prepared_lexicon_file for v in new})
        for v in new:
            self.assertEqual(v.search_algorithm_params.word_lm_params.onnx_model, job.onnx_model)
        self.assertFalse(job.use_kv_cache)
        self.assertIn("AddOnnxMetadataJob.wJhl1BGaB8Lo", job.reference_model.get_path())
        export = job.onnx_model.creator
        self.assertIn("ReturnnTrainingJob.UCMntNBI6vDq/output/models/epoch.030.pt", str(export.checkpoint))
        self.assertEqual(export.input_names, ["tokens", "tokens:size1"])
        self.assertEqual(export.output_names, ["scores"])
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.lm import get_word_lm_config

        cfg = get_word_lm_config(new[0].prepared_lexicon_file, new[0].search_algorithm_params.word_lm_params)
        self.assertEqual(cfg.type, "onnx-stateless")
        self.assertEqual(cfg.onnx_model.session.execution_provider_type, "cuda")

    def test_completed_sweeps_are_not_registered_by_run_all(self):
        import ast
        import inspect
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import run_all

        calls = {
            ast.unparse(n.func) for n in ast.walk(ast.parse(inspect.getsource(run_all))) if isinstance(n, ast.Call)
        }
        self.assertIn("oov_length.run_retained_comparisons", calls)
        self.assertIn("oov_length.run_granite_transformer_pruning_tuning", calls)
        for driver in (
            "run_approach_a",
            "run_limited_fallback",
            "run_tuning",
            "run_granite_alpha_tuning",
            "run_granite_beam_tuning",
            "run_granite_transformer_tuning",
        ):
            self.assertNotIn(f"oov_length.{driver}", calls)
        self.assertNotIn("oov_prototype.run", calls)
        self.assertNotIn("recognition.ctc_granite.lm_scale_tuning_variants", calls)
        self.assertNotIn("recognition.ctc_granite.text_normalized_lm_variants", calls)

    def test_transformer_pruning_grid(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.lm import TransformerLmParams

        kwargs = dict(
            lexicons={"B": tk.Path("open")},
            binary=tk.Path("binary"),
            repo=tk.Path("repo"),
            lm_params=TransformerLmParams(
                onnx_model=tk.Path("model"), vocab_file=tk.Path("vocab"), use_gpu=True, use_kv_cache=False
            ),
        )
        variants = oov_length.granite_transformer_pruning_variants(**kwargs)
        self.assertEqual(len(variants), 11)
        self.assertEqual(len({v.descriptor for v in variants}), 11)
        self.assertEqual(
            {
                (v.search_algorithm_params.max_beam_sizes[0], v.search_algorithm_params.max_word_end_beam_size)
                for v in variants
            },
            {(b, w) for b in (64, 128, 256) for w in (4, 16, 64, 256)} - {(256, 256)},
        )
        for v in variants:
            p = v.search_algorithm_params
            self.assertEqual((p.word_lm_params.scale, p.unknown_word_penalty, p.unknown_piece_penalty), (0.3, 4, 2))
            self.assertEqual(p.score_thresholds, [16.0])
            self.assertEqual(p.word_end_score_threshold, 16.0)
            self.assertEqual(p.unknown_word_fallback, "known-excluding")
            self.assertFalse(p.sentence_end_fallback)
            self.assertFalse(p.word_lm_params.use_kv_cache)
            self.assertTrue(p.word_lm_params.use_gpu)
            self.assertEqual(v.search_mode_params.gpu_mem_rqmt, 24)
            self.assertIsNotNone(v.prototype_traceback_formatter_serializers)
        complete = oov_length.granite_transformer_pruning_variants(**kwargs, include_completed=True)
        self.assertEqual(len(complete), 12)
        old = next(
            v
            for v in oov_length.granite_transformer_tuning_variants(**kwargs)
            if v.search_algorithm_params.word_lm_params.scale == 0.3
            and v.search_algorithm_params.unknown_word_penalty == 4
            and v.search_algorithm_params.unknown_piece_penalty == 2
        )
        self.assertEqual(sis_hash_helper(complete[-1]), sis_hash_helper(old))

    def test_transformer_pruning_entrypoint_disables_queue_with_new_identities(self):
        from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech.recognition import ctc_granite

        with patch.object(ctc_granite, "run", side_effect=lambda **kw: kw):
            old = oov_length.run_granite_transformer_tuning()
            new = oov_length.run_granite_transformer_pruning_tuning()
        self.assertEqual(new["corpora"], ["dev-other"])
        self.assertEqual(len(new["variants"]), 13)
        from dataclasses import replace
        from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.lm import get_word_lm_config
        from i6_core.rasr import WriteRasrConfigJob

        old_by_name = {v.descriptor: v for v in old["variants"]}
        self.assertFalse(set(old_by_name) & {v.descriptor for v in new["variants"]})
        controls = []
        for v in new["variants"]:
            self.assertTrue(v.descriptor.endswith("_lm-batch1"))
            p = v.search_algorithm_params.word_lm_params
            self.assertEqual(p.max_batch_size, 1)
            self.assertFalse(p.use_kv_cache)
            self.assertTrue(p.use_gpu)
            cfg = get_word_lm_config(v.prepared_lexicon_file, p)
            default = get_word_lm_config(v.prepared_lexicon_file, replace(p, max_batch_size=None))
            self.assertEqual(cfg.max_batch_size, 1)
            self.assertEqual(default.max_batch_size, 3)
            self.assertNotEqual(WriteRasrConfigJob(cfg, None).out_config, WriteRasrConfigJob(default, None).out_config)
            old_name = v.descriptor.removesuffix("_lm-batch1")
            if old_name in old_by_name:
                controls.append(old_name)
                self.assertNotEqual(sis_hash_helper(v), sis_hash_helper(old_by_name[old_name]))
        self.assertEqual(len(controls), 2)
        self.assertEqual(
            {v.search_algorithm_params.word_lm_params.vocab_file for v in new["variants"]},
            {v.search_algorithm_params.word_lm_params.vocab_file for v in old["variants"]},
        )

    def test_alpha_is_opt_in_and_hashed(self):
        paths = [
            get_tree_timesync_recog_config(
                tk.Path("lex"),
                [],
                TreeTimesyncRecogParams(
                    collapse_repeated_labels=True, max_beam_sizes=[256], unknown_piece_penalty=alpha
                ),
            )
            for alpha in (None, 0.0, 0.5)
        ]
        self.assertNotIn("unknown-piece-penalty", str(paths[0].creator.config))
        self.assertEqual(paths[2].creator.config.lib_rasr.search_algorithm.unknown_piece_penalty, 0.5)
        self.assertEqual(len(set(p.get_path() for p in paths)), 3)

    def test_ctc_alignment_repeated_labels(self):
        scores = np.full((5, 3), -20.0)
        scores[np.arange(5), [0, 1, 0, 1, 0]] = 0
        self.assertEqual(ctc_viterbi_alignment(scores, [1, 1]).tolist(), [0, 1, 2, 3, 4])
        with self.assertRaises(ValueError):
            ctc_viterbi_alignment(scores[:1], [1, 1])
        with self.assertRaises(ValueError):
            ctc_viterbi_alignment(scores, [0])

    def test_parser_apostrophes_and_no_alignment_mix(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log"
            path.write_text(
                "Recognize sequence 'dev/id/id'\n    Ground truth: \"IT'S\"\n"
                '    Recognized: "IT\'S"\n    Search traceback:\n'
                "<TracebackItem(lemma='IT'S', am_score=1, lm_score=2, start_time=0, end_time=1)>\n"
                "    Alignment traceback:\n"
                "<TracebackItem(lemma='WRONG', am_score=1, lm_score=2, start_time=0, end_time=1)>\n"
            )
            (seg,) = list(read_tracebacks(str(path)))
            self.assertEqual(seg["items"][0]["lemma"], "IT'S")
            self.assertEqual(len(seg["items"]), 1)

    def test_audit_dotted_scorer_and_costs(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rasr.config"
            path.write_text(
                "[lib-rasr]\nlabel-scorer.type = no-op\n[lib-rasr.lm]\nscale=0.3\n"
                "[lib-rasr.search-algorithm]\nunknown-word-penalty=0\n"
            )
            report = audit_config(tk.Path(str(path)))
            self.assertEqual(report["word_exit_cost"], 0)
            self.assertEqual(report["lm_scale"], 0.3)
            path.write_text(path.read_text().replace("no-op", "transition"))
            with self.assertRaises(ValueError):
                audit_config(tk.Path(str(path)))

    def test_extension_candidates_and_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            lex, _ = smoke_lexicon("granite")
            lex.lemmata = [l for l in lex.lemmata if not (l.special or "").startswith("unknown-")]
            write_xml(str(p / "lex.xml"), lex.to_xml())
            (p / "train").write_text("KAT DOG KAX KAX X LMWORD\n")
            (p / "lm").write_text("\\data\\\nngram 1=3\n\\1-grams:\n-1 <UNK>\n-1 DOG\n-2 LMWORD\n\\end\\\n")
            rank = SimpleNamespace(
                training_text=tk.Path(str(p / "train")),
                recognition_lexicon=tk.Path(str(p / "lex.xml")),
                arpa_lm=tk.Path(str(p / "lm")),
                out_words=tk.Path(str(p / "words")),
                out_report=tk.Path(str(p / "rank.json")),
            )
            RankOovLexiconCandidatesJob.run(rank)
            self.assertEqual((p / "words").read_text(), "KAX\t2\nX\t1\n")
            table = [dict(token="▁ka", label="▁ka", role="start"), dict(token="x", label="x", role="internal")]
            (p / "table").write_text(json.dumps(table))
            extend = SimpleNamespace(
                lexicon=rank.recognition_lexicon,
                candidates=rank.out_words,
                tokenizer_table=tk.Path(str(p / "table")),
                repo=tk.Path("unused"),
                max_candidates=1,
                out_lexicon=tk.Path(str(p / "extended.xml")),
                out_report=tk.Path(str(p / "extend.json")),
                out_rejected=tk.Path(str(p / "rejected")),
            )
            tokenizer = SimpleNamespace(tokenize=lambda text: ["▁ka", "x"])
            with patch(
                "i6_experiments.example_setups.seq2seq_rasr_2025.data.oov_extended_lexicon._load_tokenizer",
                return_value=tokenizer,
            ):
                ExtendGraniteOovLexiconJob.run(extend)
            root = read_lexicon(extend.out_lexicon)
            extra = root.findall("lemma")[-1]
            self.assertEqual(extra.findtext("orth"), "KAX")
            self.assertEqual(extra.findtext("synt/tok"), "<UNK>")
            self.assertIsNone(extra.get("special"))
            self.assertEqual(
                [x.findtext("symbol") for x in root.findall("phoneme-inventory/phoneme")], list(lex.phonemes)
            )

    def test_guard_detects_long_unknown_but_not_attested_word(self):
        with tempfile.TemporaryDirectory() as d:
            lex, _ = smoke_lexicon("bpe")
            path = Path(d) / "lex.xml"
            write_xml(str(path), lex.to_xml())
            guard = GuardedOovTracebackFormatter(tk.Path(str(path)), "continuation-marked", max_unknown_pieces=3)

            def item(orth):
                return SimpleNamespace(lemma=orth, start_time=0, end_time=1, confidence_score=None)

            with patch("builtins.print"):
                guard.validate_traceback([item("ka@@"), item("t&&")], "KAT", "known")
                with self.assertRaisesRegex(ValueError, "piece limit"):
                    guard.validate_traceback([item("ka@@")] * 4 + [item("x&&")], "KAX", "bad")
                with self.assertRaisesRegex(ValueError, "collapsed"):
                    guard.validate_traceback([item("x&&")], "WORD " * 12, "bad-boundaries")


if __name__ == "__main__":
    unittest.main()
