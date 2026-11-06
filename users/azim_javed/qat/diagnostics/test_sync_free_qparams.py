"""Run from the QAT setup: python -m diagnostics.test_sync_free_qparams.

Uses the original observer-based forwards as references. Runs CUDA cases too
when available; CPU equivalence does not establish GPU speedup.
"""

import copy
import sys
import unittest
from unittest.mock import patch
import warnings

import torch

from recipe.model_pipelines.common.memristor_layers import ActivationQuantizer, WeightQuantizer
from recipe.model_pipelines.common.memristor_layers_mem_inited import ActivationQuantizer as MemActivationQuantizer
from .original_quantizers import (
    OriginalWeightQuantizer, OriginalActivationQuantizer, OriginalMemActivationQuantizer,
    prepare_reference, reference_forward,
)
from .test_sync_free_qparams_edges import SyncFreeQparamsEdgesTest


class SyncFreeQparamsTest(unittest.TestCase):
    def assert_state_equal(self, a, b):
        left, right = a.state_dict(), b.state_dict()
        self.assertEqual(list(left), list(right))
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)

    def compare(self, fast, old, device):
        fast = fast.to(device)
        self.assertIsInstance(fast.observer.quant_min, int)
        self.assertIsInstance(fast.observer.quant_max, int)
        old = prepare_reference(old, device)
        self.assertIsNot(type(fast), type(old))
        self.assertFalse(hasattr(old, "_quant_min_int"))
        self.assert_state_equal(fast, old)
        for training, magnitude, input_dtype in [
            (True, 1.0, torch.float32),
            (True, 3.0, torch.float32),
            (True, 0.0, torch.float32),
            (True, 2.0, torch.bfloat16),
            (False, 8.0, torch.float32),
            (False, 8.0, torch.bfloat16),
        ]:
            fast.train(training)
            old.train(training)
            x = (torch.randn(4, 7, device=device) * magnitude).to(input_dtype).requires_grad_()
            y = x.detach().clone().requires_grad_()
            out_fast, out_old = fast(x), reference_forward(old, y)
            self.assertTrue(torch.equal(out_fast, out_old), "outputs")
            self.assertTrue(torch.equal(fast.scale, old.scale), "scale")
            self.assertTrue(torch.equal(fast.zero_point, old.zero_point), "zero point")
            upstream = torch.randn_like(out_fast)
            out_fast.backward(upstream)
            out_old.backward(upstream)
            self.assertTrue(torch.equal(x.grad, y.grad), "gradients")
            self.assert_state_equal(fast, old)
        # Strict load in both directions, including fresh eval derived attributes.
        fresh = copy.deepcopy(fast)
        fresh.scale = fresh.zero_point = None
        fresh.load_state_dict(old.state_dict(), strict=True)
        old.load_state_dict(fast.state_dict(), strict=True)
        x = torch.randn(4, 7, device=device)
        self.assertTrue(torch.equal(fresh(x), reference_forward(old, x)), "loaded eval")
        self.assert_state_equal(fresh, old)
        # Explicit export/conversion qparam path must remain identical.
        fast.set_scale_and_zp()
        old.set_scale_and_zp()
        self.assertTrue(torch.equal(fast.scale, old.scale))
        self.assertTrue(torch.equal(fast.zero_point, old.zero_point))

    def test_equivalence(self):
        torch.manual_seed(19)
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.qint8, torch.quint8):
                for method in ("per_tensor", "per_tensor_symmetric"):
                    for bits in (1.5, 4, 8):
                        for reduced in (False, True):
                            # PyTorch disallows reduced-range unsigned symmetric observers.
                            if reduced and dtype == torch.quint8 and method == "per_tensor_symmetric":
                                continue
                            with self.subTest(device=device, dtype=dtype, method=method, bits=bits, reduced=reduced):
                                self.compare(
                                    WeightQuantizer(bits, dtype, method, reduced),
                                    OriginalWeightQuantizer(bits, dtype, method, reduced), device,
                                )
                                for cls, original in (
                                    (ActivationQuantizer, OriginalActivationQuantizer),
                                    (MemActivationQuantizer, OriginalMemActivationQuantizer),
                                ):
                                    for moving in (None, 0.01):
                                        self.compare(
                                            cls(bits, dtype, method, None, moving, reduced),
                                            original(bits, dtype, method, None, moving, reduced), device,
                                        )
            for cls, original in (
                (ActivationQuantizer, OriginalActivationQuantizer),
                (MemActivationQuantizer, OriginalMemActivationQuantizer),
            ):
                for moving in (None, 0.01):
                    with self.subTest(device=device, cls=cls, moving=moving, method="per_channel"):
                        self.compare(
                            cls(8, torch.qint8, "per_channel", 1, moving),
                            original(8, torch.qint8, "per_channel", 1, moving), device,
                        )

    def test_uncalibrated_activation_defaults(self):
        # Regular activations historically use calculate_qparams' (1, 0) fallback.
        fast = ActivationQuantizer(8, torch.qint8, "per_tensor", None, None).eval()
        old = OriginalActivationQuantizer(8, torch.qint8, "per_tensor", None, None).eval()
        x = torch.randn(4, 7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertTrue(torch.equal(fast(x), reference_forward(old, x)))
        self.assert_state_equal(fast, old)

    def test_forward_bypasses_calculate_qparams(self):
        for module in (
            WeightQuantizer(8, torch.qint8, "per_tensor_symmetric"),
            ActivationQuantizer(8, torch.qint8, "per_tensor", None, None),
            MemActivationQuantizer(8, torch.qint8, "per_tensor", None, None),
        ):
            with patch.object(module.observer, "calculate_qparams", side_effect=AssertionError("slow path")):
                module(torch.randn(4, 7))
                module.eval()
                module(torch.randn(4, 7))

    def test_original_initialization_errors(self):
        for cls in (MemActivationQuantizer, OriginalMemActivationQuantizer):
            module = cls(8, torch.qint8, "per_tensor", None, None).eval()
            with self.assertRaisesRegex(AssertionError, "Need to init observer first"):
                module(torch.randn(4, 7))
        for cls in (WeightQuantizer, OriginalWeightQuantizer):
            module = cls(8, torch.qint8, "per_tensor")
            with self.assertRaisesRegex(AssertionError, "Need to init observer first"):
                module(torch.empty(0))

    def test_eval_observer_freezing_and_weight_scale_cache(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for cls, original in (
                (WeightQuantizer, OriginalWeightQuantizer),
                (ActivationQuantizer, OriginalActivationQuantizer),
                (MemActivationQuantizer, OriginalMemActivationQuantizer),
            ):
                args = (8, torch.qint8, "per_tensor")
                if cls is not WeightQuantizer:
                    args += (None, None)
                fast, old = cls(*args).to(device), prepare_reference(original(*args), device)
                fast(torch.ones(4, 7, device=device))
                old(torch.ones(4, 7, device=device))
                fast.eval()
                old.eval()
                # Change observer state externally after scale has been initialized.
                # Weights historically cache eval qparams; activations recompute them.
                fast.observer.max_val.fill_(2)
                old.observer.max_val.fill_(2)
                x = torch.full((4, 7), 8.0, device=device)
                self.assertTrue(torch.equal(fast(x), old(x)))
                self.assert_state_equal(fast, old)
                self.assertTrue(torch.equal(fast.scale, old.scale))


if __name__ == "__main__":
    if "--require-cuda" in sys.argv:
        sys.argv.remove("--require-cuda")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required but unavailable; run on an allocated GPU node.")
        print(f"CUDA equivalence tests enabled: {torch.cuda.get_device_name(0)}")
    unittest.main()
