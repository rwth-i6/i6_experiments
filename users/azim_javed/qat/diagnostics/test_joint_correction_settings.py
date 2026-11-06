"""Focused hash/routing checks; run as a module from the QAT setup root."""

import copy
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from sisyphus import tk
from i6_core.returnn import PtCheckpoint
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings
from synaptogen_ml.synaptogen import rng as device_rng

from recipe.experiments.librispeech.recognition.memristor import qat_ffnn_transducer_bpe as recog
from recipe.experiments.librispeech.training.qat_ffnn_transducer_bpe import get_model_config
from recipe.model_pipelines.common.memrecog import MemristorModelConversionJob
from recipe.model_pipelines.common.memristor_layers import ActivationQuantizer, LinearQuant
from recipe.model_pipelines.qat_ffnn_transducer.pytorch_modules import QATFFNNTransducerModel


def ideal_settings():
    return CycleCorrectionSettings(None, None, None, True)


def hash_args():
    return dict(
        checkpoint=PtCheckpoint(tk.Path("/tmp/joint-correction-hash-check.pt")),
        model_config=get_model_config(
            weight_bit_prec=4, activation_bit_prec=8, weight_dropout=0.0, weight_pruning_config=None
        ),
        model_class=QATFFNNTransducerModel,
        replace_params=dict(
            converter_hardware_settings=DacAdcHardwareSettings(8, 4, 4, 0.6, 8020.0),
            pos_enc_converter_hardware_settings=DacAdcHardwareSettings(8, 1, 7, 0.6, 8020.0),
            correction_settings=None,
            num_cycles=1,
        ),
        regrid_embedding=True,
    )


def calibrated_linear():
    quant = ActivationQuantizer(8, torch.qint8, "per_tensor_symmetric", 1, None)
    linear = LinearQuant(4, 4, 4, torch.qint8, "per_tensor_symmetric", bias=True)
    quant(torch.randn(3, 4))
    linear(torch.randn(3, 4))
    return quant, linear


def small_conversion_model():
    # Exercise the real full-offload prep_quant/linear programming on tiny layers.
    model = QATFFNNTransducerModel.__new__(QATFFNNTransducerModel)
    torch.nn.Module.__init__(model)
    model.conformer = Mock()
    model.token_embedding = torch.nn.Identity()
    model._convert_embedding = lambda embedding: embedding
    model.converter_hardware_settings = DacAdcHardwareSettings(8, 4, 4, 0.6, 8020.0)
    model.correction_settings = None
    model.joint_correction_settings = None
    model.joint_correction_layer = 3
    model.num_cycles = 0
    qpred, pred = calibrated_linear()
    q1, linear1 = calibrated_linear()
    q2, linear2 = calibrated_linear()
    model.prediction_net = torch.nn.Sequential(torch.nn.Identity(), qpred, pred,
                                              torch.nn.Identity(), torch.nn.Tanh())
    model.joint_net = torch.nn.Sequential(torch.nn.Identity(), q1, linear1,
                                         torch.nn.Identity(), torch.nn.Tanh(),
                                         torch.nn.Identity(), q2, linear2)
    return model


class JointCorrectionSettingsTest(unittest.TestCase):
    def test_previous_hashes_and_override_hash(self):
        args = hash_args()
        # Recorded from the working tree before adding this parameter.
        self.assertEqual(MemristorModelConversionJob.hash(args), "SiEzx8jZlEPv")
        self.assertEqual(MemristorModelConversionJob.hash(dict(args, joint_correction_settings=None)),
                         "SiEzx8jZlEPv")
        self.assertEqual(MemristorModelConversionJob.hash(dict(args, joint_correction_layer=3)),
                         "SiEzx8jZlEPv")
        self.assertEqual(MemristorModelConversionJob.hash(dict(args, regrid_embedding=False,
                                                              joint_correction_settings=None)),
                         "xPTyyCDACBHY")
        ideal_hash = MemristorModelConversionJob.hash(dict(args, joint_correction_settings=ideal_settings()))
        self.assertEqual(ideal_hash, "Jn26zS1G4IH4")
        self.assertEqual(MemristorModelConversionJob.hash(dict(
            args, joint_correction_settings=ideal_settings(), joint_correction_layer=3)), ideal_hash)
        self.assertNotEqual(ideal_hash, "SiEzx8jZlEPv")
        realistic = CycleCorrectionSettings(2, 0.6, 0.05, False)
        self.assertNotEqual(MemristorModelConversionJob.hash(dict(args, joint_correction_settings=realistic)),
                            ideal_hash)

    def test_layer_selection_hashes_separately_with_supplied_settings(self):
        args = hash_args()
        for settings in (ideal_settings(), CycleCorrectionSettings(2, 0.6, 0.05, False)):
            hashes = [MemristorModelConversionJob.hash(dict(
                args, joint_correction_settings=settings, joint_correction_layer=layer)) for layer in (1, 2, 3)]
            self.assertEqual(len(set(hashes)), 3)

    def test_selective_ideal_programming(self):
        torch.manual_seed(17)
        original = small_conversion_model()
        rng_state = copy.deepcopy(device_rng.bit_generator.state)
        reference = copy.deepcopy(original)
        reference.prep_quant()
        for flag in (1, 2, 3):
            with self.subTest(flag=flag):
                model = copy.deepcopy(original)
                model.joint_correction_settings = ideal_settings()
                model.joint_correction_layer = flag
                device_rng.bit_generator.state = copy.deepcopy(rng_state)
                model.prep_quant()
                for key, value in reference.prediction_net.state_dict().items():
                    self.assertTrue(torch.equal(value, model.prediction_net.state_dict()[key]), key)
                for index, bit in ((2, 1), (7, 2)):
                    arrays = [array.r for pair in model.joint_net[index].memristors
                              for array in (pair.pos, pair.neg)]
                    is_ideal = all(bool(torch.all((r == 0) | (r == 1))) for r in arrays)
                    self.assertEqual(is_ideal, bool(flag & bit), (flag, index))

    def test_realistic_settings_are_not_replaced_with_ideal_settings(self):
        model = small_conversion_model()
        model.joint_correction_settings = CycleCorrectionSettings(0, 0.6, 0.05, False)
        model.joint_correction_layer = 3
        model.prep_quant()
        for index in (2, 7):
            r = model.joint_net[index].memristors[0].pos.r
            self.assertTrue(torch.any((r != 0) & (r != 1)))

    def test_only_joiner_programming_changes(self):
        torch.manual_seed(17)
        original = small_conversion_model()
        reference = copy.deepcopy(original)
        overridden = copy.deepcopy(original)
        overridden.joint_correction_settings = ideal_settings()
        device_rng_state = copy.deepcopy(device_rng.bit_generator.state)
        np.random.seed(19)
        reference.prep_quant()
        device_rng.bit_generator.state = device_rng_state
        np.random.seed(19)
        overridden.prep_quant()
        left, right = reference.prediction_net.state_dict(), overridden.prediction_net.state_dict()
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)
        for index in (2, 7):
            programmed = overridden.joint_net[index]
            for pair in programmed.memristors:
                for array in (pair.pos, pair.neg):
                    self.assertTrue(torch.all((array.r == 0) | (array.r == 1)))
            self.assertFalse(torch.equal(reference.joint_net[index].memristors[0].pos.r,
                                         programmed.memristors[0].pos.r))
        overridden.conformer.prep_quant.assert_called_once_with()

    def test_selected_none_is_passed_directly_without_global_fallback(self):
        torch.manual_seed(17)
        model = small_conversion_model()
        model.correction_settings = ideal_settings()
        model.joint_correction_settings = None
        model.joint_correction_layer = 1
        model.prep_quant()
        for layer in (model.prediction_net[2], model.joint_net[7]):
            for pair in layer.memristors:
                self.assertTrue(torch.all((pair.pos.r == 0) | (pair.pos.r == 1)))
        r = model.joint_net[2].memristors[0].pos.r
        self.assertTrue(torch.any((r != 0) & (r != 1)))

    def test_exact_settings_objects_are_routed_to_selected_layers(self):
        for selected in (1, 2, 3):
            model = small_conversion_model()
            global_settings = CycleCorrectionSettings(2, 0.6, 0.05, False)
            joint_settings = CycleCorrectionSettings(4, 0.4, 0.1, False)
            model.correction_settings = global_settings
            model.joint_correction_settings = joint_settings
            model.joint_correction_layer = selected
            with patch.object(model, "_convert_linear",
                              side_effect=lambda linear, quant, *args, **kw: linear) as convert:
                model.prep_quant()
            self.assertIs(convert.call_args_list[0].args[2], global_settings)
            for call, layer in zip(convert.call_args_list[1:], (1, 2)):
                expected = joint_settings if selected in (layer, 3) else global_settings
                self.assertIs(call.kwargs["correction_settings"], expected)

    def test_job_passes_override_before_conversion(self):
        settings = ideal_settings()
        model = Mock()
        model.prep_quant.side_effect = lambda: self.assertIs(model.joint_correction_settings, settings)
        job = SimpleNamespace(
            checkpoint=SimpleNamespace(path="unused.pt"), out_config=object(),
            model_class=Mock(return_value=model), regrid_embedding=True,
            joint_correction_settings=settings, joint_correction_layer=3,
            out_checkpoint=SimpleNamespace(path="unused-output.pt"),
        )
        with patch("torch.load", return_value={"model": {}}), patch("torch.save"):
            MemristorModelConversionJob.run(job)
        model.prep_quant.assert_called_once_with()

    def test_job_passes_layer_selection_before_conversion(self):
        for flag in (1, 2, 3):
            model = Mock()
            model.prep_quant.side_effect = lambda: self.assertEqual(model.joint_correction_layer, flag)
            job = SimpleNamespace(
                checkpoint=SimpleNamespace(path="unused.pt"), out_config=object(),
                model_class=Mock(return_value=model), regrid_embedding=True,
                joint_correction_settings=ideal_settings(), joint_correction_layer=flag,
                out_checkpoint=SimpleNamespace(path="unused-output.pt"),
            )
            with patch("torch.load", return_value={"model": {}}), patch("torch.save"):
                MemristorModelConversionJob.run(job)
            model.prep_quant.assert_called_once_with()

    def test_recognition_forwards_settings_and_distinguishes_aliases(self):
        # Stop before constructing jobs; verify the public API and descriptor routing.
        for settings, suffix in ((None, ""), (ideal_settings(), "_joint_ideal_programming"),
                                 (CycleCorrectionSettings(2, 0.6, 0.05, False),
                                  "_joint_correction_2cycles_0.6input_0.05dev")):
            variant = SimpleNamespace(cycles=1, search_mode_params=SimpleNamespace(batch_size_seconds=1),
                                      descriptor="lexfree")
            with patch.object(recog, "_run_single_variant", return_value=(0, [])) as run_single, \
                    patch.object(recog, "post_recog_memristor_results", return_value=[]):
                recog.run(SimpleNamespace(descriptor="w4"), variants=[variant], corpora=[], max_runs=1,
                          joint_correction_settings=settings)
            self.assertIs(run_single.call_args.kwargs["joint_correction_settings"], settings)
            self.assertEqual(run_single.call_args.kwargs["suffix"], suffix)

    def test_recognition_routes_supplied_settings_and_layer_selection(self):
        for settings, layer, suffix in (
            (None, 3, ""),
            (ideal_settings(), 1, "_joint_l1_ideal_programming"),
            (ideal_settings(), 2, "_joint_l2_ideal_programming"),
            (ideal_settings(), 3, "_joint_ideal_programming"),
            (CycleCorrectionSettings(2, 0.6, 0.05, False), 1,
             "_joint_correction_2cycles_0.6input_0.05dev_joint_l1"),
        ):
            variant = SimpleNamespace(cycles=1, search_mode_params=SimpleNamespace(batch_size_seconds=1),
                                      descriptor="lexfree")
            with patch.object(recog, "_run_single_variant", return_value=(0, [])) as run_single, \
                    patch.object(recog, "post_recog_memristor_results", return_value=[]):
                recog.run(SimpleNamespace(descriptor="w4"), variants=[variant], corpora=[], max_runs=1,
                          joint_correction_settings=settings, joint_correction_layer=layer)
            self.assertIs(run_single.call_args.kwargs["joint_correction_settings"], settings)
            self.assertEqual(run_single.call_args.kwargs["joint_correction_layer"], layer)
            self.assertEqual(run_single.call_args.kwargs["suffix"], suffix)

    def test_invalid_layer_flag_is_rejected(self):
        for flag in (-1, 0, 4):
            with self.assertRaisesRegex(ValueError, "joint_correction_layer must be"):
                recog.run(SimpleNamespace(descriptor="w4"), variants=[], corpora=[],
                          joint_correction_layer=flag)

    def test_actual_experiment_calls_produce_six_distinct_descriptors(self):
        path = Path("recipe/experiments/librispeech/__init__.py")
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_mem")
        blocks = [n for n in function.body if isinstance(n, ast.Expr) and any(
            isinstance(k, ast.keyword) and k.arg == "joint_correction_layer" for k in ast.walk(n))]
        self.assertEqual(len(blocks), 6)
        env = dict(
            recog_results=[], memristor_recognition=SimpleNamespace(qat_ffnn_transducer_bpe=recog),
            models={"qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8": SimpleNamespace(descriptor="w4_a8")},
            converter_hardware_settings=DacAdcHardwareSettings(8, 4, 4, 0.6, 8020.0),
            converter_hardware_settings_4prec_8range=DacAdcHardwareSettings(8, 4, 8, 0.6, 8020.0),
            pos_enc_converter_hardware_settings=DacAdcHardwareSettings(8, 1, 7, 0.6, 8020.0),
            correction_settings=None, CycleCorrectionSettings=CycleCorrectionSettings,
        )
        with patch.object(recog, "_run_single_variant", return_value=(0, [])) as run_single, \
                patch.object(recog, "post_recog_memristor_results", return_value=[]) as report:
            exec(compile(ast.Module(body=blocks, type_ignores=[]), str(path), "exec"), env)
        self.assertEqual(run_single.call_count, 6)
        self.assertEqual(len({call.kwargs["descriptor"] for call in report.call_args_list}), 6)
        self.assertEqual(sorted(call.kwargs["joint_correction_layer"] for call in run_single.call_args_list),
                         [1, 1, 2, 2, 3, 3])
        for call in run_single.call_args_list:
            self.assertTrue(call.kwargs["joint_correction_settings"].ideal_programming)
            self.assertEqual(call.kwargs["variant"].search_mode_params.num_shards, 3)


if __name__ == "__main__":
    unittest.main()
