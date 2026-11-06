"""Independent edge-case checks; included by test_sync_free_qparams."""

import io
import unittest
import warnings

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from recipe.model_pipelines.common.memristor_layers import ActivationQuantizer, WeightQuantizer
from recipe.model_pipelines.common.memristor_layers_mem_inited import ActivationQuantizer as MemActivationQuantizer
from recipe.model_pipelines.common.quantization import observer_qparams
from .original_quantizers import (
    OriginalActivationQuantizer, OriginalWeightQuantizer, OriginalMemActivationQuantizer,
    prepare_reference,
)


PAIRS = (
    (WeightQuantizer, OriginalWeightQuantizer),
    (ActivationQuantizer, OriginalActivationQuantizer),
    (MemActivationQuantizer, OriginalMemActivationQuantizer),
)


class OperationRecorder(TorchDispatchMode):
    """Record dispatched ops without profiler CUDA initialization side effects."""

    def __init__(self):
        super().__init__()
        self.operations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(str(func))
        return func(*args, **(kwargs or {}))


class SyncFreeQparamsEdgesTest(unittest.TestCase):
    @staticmethod
    def devices():
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def make_pair(self, pair, device, method="per_tensor", dtype=torch.qint8, bits=8, axis=None, moving=None):
        cls, original = pair
        args = (bits, dtype, method)
        if cls is not WeightQuantizer:
            args += (axis, moving)
        return cls(*args).to(device), prepare_reference(original(*args), device)

    def assert_tensor_equal(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.device, b.device)
        self.assertTrue(torch.equal(a, b))
        # torch.equal alone treats positive and negative zero as equal.
        self.assertTrue(torch.equal(
            a.detach().contiguous().reshape(-1).view(torch.uint8),
            b.detach().contiguous().reshape(-1).view(torch.uint8),
        ))

    def compare_step(self, fast, old, values):
        x = values.detach().requires_grad_()
        y = values.detach().clone().requires_grad_()
        out_fast, out_old = fast(x), old(y)
        self.assert_tensor_equal(out_fast, out_old)
        self.assert_tensor_equal(fast.scale, old.scale)
        self.assert_tensor_equal(fast.zero_point, old.zero_point)
        upstream = torch.randn_like(out_fast)
        out_fast.backward(upstream)
        out_old.backward(upstream)
        self.assert_tensor_equal(x.grad, y.grad)
        a, b = fast.state_dict(), old.state_dict()
        self.assertEqual(list(a), list(b))
        for key in a:
            self.assert_tensor_equal(a[key], b[key])

    def test_fresh_constant_small_and_one_sided_ranges(self):
        cases = [
            [0.0, 0.0], [-0.0, 0.0], [1.0, 1.0], [-1.0, -1.0],
            [1.0, 3.0], [-3.0, -1.0], [-1.0e-10, 2.0e-10],
            [-1.0e-38, 1.0e-38], [-1.0e30, 1.0e30],
        ]
        for device in self.devices():
            for pair in PAIRS:
                for method in ("per_tensor", "per_tensor_symmetric"):
                    for values in cases:
                        with self.subTest(device=device, cls=pair[0], method=method, values=values):
                            fast, old = self.make_pair(pair, device, method)
                            self.compare_step(fast, old, torch.tensor(values, device=device))

    def test_rounding_clipping_and_ste_boundaries(self):
        for device in self.devices():
            for pair in PAIRS:
                for method in ("per_tensor", "per_tensor_symmetric"):
                    for dtype in (torch.qint8, torch.quint8):
                        with self.subTest(device=device, cls=pair[0], method=method, dtype=dtype):
                            fast, old = self.make_pair(pair, device, method, dtype)
                            # Freeze known observer ranges before probing half-step ties,
                            # adjacent representable values, and clipping endpoints.
                            for module in (fast, old):
                                module.observer.min_val.fill_(-1)
                                module.observer.max_val.fill_(3)
                                module.set_scale_and_zp()
                                module.eval()
                            qmin, qmax = int(fast.quant_min), int(fast.quant_max)
                            levels = torch.arange(qmin - 2, qmax + 3, device=device, dtype=torch.float32)
                            ties = (levels + 0.5 - old.zero_point) * old.scale
                            values = torch.cat([
                                ties,
                                torch.nextafter(ties, torch.full_like(ties, -float("inf"))),
                                torch.nextafter(ties, torch.full_like(ties, float("inf"))),
                                (levels - old.zero_point) * old.scale,
                            ])
                            self.compare_step(fast, old, values)

    def test_input_layouts_dtypes_and_autocast(self):
        for device in self.devices():
            for pair in PAIRS:
                for input_dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    for layout in ("scalar", "transposed", "sliced", "channels_last"):
                        with self.subTest(device=device, cls=pair[0], dtype=input_dtype, layout=layout):
                            fast, old = self.make_pair(pair, device)
                            x = torch.randn(2, 3, 4, 5, device=device, dtype=input_dtype)
                            if layout == "scalar":
                                x = x[0, 0, 0, 0]
                            elif layout == "transposed":
                                x = x.transpose(0, 2)
                            elif layout == "sliced":
                                x = x[:, :, ::2, :]
                            else:
                                x = x.contiguous(memory_format=torch.channels_last)
                            self.compare_step(fast, old, x)
            for pair in PAIRS:
                for autocast_dtype in (torch.bfloat16, torch.float16):
                    fast, old = self.make_pair(pair, device)
                    with torch.autocast(device_type=device, dtype=autocast_dtype):
                        self.compare_step(fast, old, torch.randn(3, 7, device=device))

    def test_per_channel_axes_and_observer_updates(self):
        for device in self.devices():
            for pair in PAIRS[1:]:
                for axis in (0, 1, 2):
                    for dtype in (torch.qint8, torch.quint8):
                        for moving in (None, 0.0, 0.01, 1.0):
                            with self.subTest(device=device, cls=pair[0], axis=axis, dtype=dtype, moving=moving):
                                fast, old = self.make_pair(pair, device, "per_channel", dtype, axis=axis, moving=moving)
                                for magnitude in (1.0, 3.0, 0.0):
                                    self.compare_step(fast, old, torch.randn(2, 3, 4, device=device) * magnitude)
                                fast.eval()
                                old.eval()
                                self.compare_step(fast, old, torch.randn(2, 3, 4, device=device) * 10)

    def test_empty_inputs_before_and_after_calibration(self):
        for device in self.devices():
            for pair in PAIRS:
                for training in (True, False):
                    for calibrated in (False, True):
                        with self.subTest(device=device, cls=pair[0], training=training, calibrated=calibrated):
                            fast, old = self.make_pair(pair, device)
                            if calibrated:
                                self.compare_step(fast, old, torch.randn(3, 7, device=device))
                            fast.train(training)
                            old.train(training)
                            x = torch.empty(0, device=device)
                            if not calibrated and pair[0] in (WeightQuantizer, MemActivationQuantizer):
                                for module in (fast, old):
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        with self.assertRaises(AssertionError):
                                            module(x)
                            else:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    self.compare_step(fast, old, x)

    def test_qparam_math_randomized_and_all_precision_labels(self):
        # Directly compare independent observer math, including vector ranges.
        generator = torch.Generator().manual_seed(732)
        ranges = torch.randn(1024, 2, generator=generator)
        ranges *= torch.logspace(-30, 30, 1024).unsqueeze(1)
        ranges = ranges.sort(dim=1).values
        for device in self.devices():
            for bits in (1.5, 2, 2.5, 3, 3.5, 4, 8):
                for dtype in (torch.qint8, torch.quint8):
                    for method in ("per_tensor", "per_tensor_symmetric"):
                        with self.subTest(device=device, bits=bits, dtype=dtype, method=method):
                            fast, old = self.make_pair(PAIRS[0], device, method, dtype, bits)
                            for values in (ranges.to(device), ranges[:1].to(device)):
                                lo, hi = values.unbind(dim=1)
                                if len(values) == 1:
                                    lo, hi = lo.squeeze(0), hi.squeeze(0)
                                fast.observer.min_val = lo
                                fast.observer.max_val = hi
                                scale, zp = observer_qparams(fast.observer, fast._observer_quant_min_int, fast._observer_quant_max_int)
                                old_scale, old_zp = old.observer._calculate_qparams(lo, hi)
                                self.assert_tensor_equal(scale, old_scale.float())
                                self.assert_tensor_equal(zp, old_zp.int())

    def test_real_checkpoint_roundtrip_and_fresh_load(self):
        for device in self.devices():
            for pair in PAIRS:
                for method in (["per_tensor", "per_tensor_symmetric"] if pair[0] is WeightQuantizer else ["per_tensor", "per_channel"]):
                    axis = 1 if method == "per_channel" else None
                    with self.subTest(device=device, cls=pair[0], method=method):
                        fast, old = self.make_pair(pair, device, method, axis=axis)
                        self.compare_step(fast, old, torch.randn(3, 7, device=device))
                        # Serialize rather than only handing in-memory state between modules.
                        streams = [io.BytesIO(), io.BytesIO()]
                        torch.save(fast.state_dict(), streams[0])
                        torch.save(old.state_dict(), streams[1])
                        fresh_fast, fresh_old = self.make_pair(pair, device, method, axis=axis)
                        for stream in streams:
                            stream.seek(0)
                        fresh_fast.load_state_dict(torch.load(streams[1], map_location=device, weights_only=True), strict=True)
                        fresh_old.load_state_dict(torch.load(streams[0], map_location=device, weights_only=True), strict=True)
                        fresh_fast.eval()
                        fresh_old.eval()
                        self.compare_step(fresh_fast, fresh_old, torch.randn(3, 7, device=device))
                        fresh_fast.train()
                        fresh_old.train()
                        self.compare_step(fresh_fast, fresh_old, torch.randn(3, 7, device=device) * 3)

    def test_qparams_no_scalar_reads_or_autograd_history(self):
        for device in self.devices():
            fast, old = self.make_pair(PAIRS[0], device)
            fast(torch.randn(4, 7, device=device, requires_grad=True))
            with OperationRecorder() as recorder:
                scale, zp = observer_qparams(fast.observer, fast._observer_quant_min_int, fast._observer_quant_max_int)
            self.assertFalse(any("_local_scalar_dense" in op or "aten.item" in op for op in recorder.operations))
            self.assertFalse(scale.requires_grad)
            self.assertIsNone(scale.grad_fn)
            self.assertFalse(zp.requires_grad)

    def test_ordinary_minmax_training_forwards_have_no_scalar_reads(self):
        for device in self.devices():
            for pair in PAIRS:
                fast, old = self.make_pair(pair, device)
                x = torch.randn(3, 7, device=device)
                with OperationRecorder() as recorder:
                    fast(x)
                self.assertFalse(any("_local_scalar_dense" in op or "aten.item" in op for op in recorder.operations))
                # Positive control: the old qparam path really performs scalar reads.
                with OperationRecorder() as old_recorder:
                    old(x)
                self.assertTrue(any("_local_scalar_dense" in op for op in old_recorder.operations))

    def test_dtype_moves_and_cpu_to_cuda_before_first_forward(self):
        for device in self.devices():
            for pair in PAIRS:
                fast, old = self.make_pair(pair, "cpu")
                fast = fast.to(device).double()
                old = prepare_reference(old, device).double()
                self.compare_step(fast, old, torch.randn(3, 7, device=device, dtype=torch.float64))

    def test_empty_per_channel_and_uncalibrated_eval(self):
        for device in self.devices():
            for pair in PAIRS[1:]:
                for calibrated in (False, True):
                    for shape in ((0, 3, 4), (2, 3, 4)):
                        with self.subTest(device=device, cls=pair[0], calibrated=calibrated, shape=shape):
                            fast, old = self.make_pair(pair, device, "per_channel", axis=1)
                            if calibrated:
                                self.compare_step(fast, old, torch.randn(2, 3, 4, device=device))
                            fast.eval()
                            old.eval()
                            x = torch.empty(shape, device=device) if shape[0] == 0 else torch.randn(shape, device=device)
                            if calibrated:
                                self.compare_step(fast, old, x)
                            else:
                                # Original code cannot quantize multiple channels
                                # with uncalibrated single-element default qparams.
                                errors = []
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    for module in (fast, old):
                                        try:
                                            module(x)
                                        except (AssertionError, RuntimeError) as error:
                                            errors.append(type(error))
                                self.assertEqual(len(errors), 2)
                                self.assertIs(errors[0], errors[1])

    def test_optimizer_steps_in_quantized_linear_pipeline(self):
        for device in self.devices():
            for moving in (None, 0.01):
                weight_fast, weight_old = self.make_pair(PAIRS[0], device, "per_tensor_symmetric")
                act_fast, act_old = self.make_pair(PAIRS[1], device, moving=moving)
                w_fast = torch.nn.Parameter(torch.randn(5, 7, device=device))
                w_old = torch.nn.Parameter(w_fast.detach().clone())
                b_fast = torch.nn.Parameter(torch.randn(5, device=device))
                b_old = torch.nn.Parameter(b_fast.detach().clone())
                optimizers = [torch.optim.AdamW([w_fast, b_fast]), torch.optim.AdamW([w_old, b_old])]
                for _ in range(4):
                    x = torch.randn(3, 7, device=device)
                    target = torch.randn(3, 5, device=device)
                    outputs = []
                    for w, b, wq, aq, optimizer in (
                        (w_fast, b_fast, weight_fast, act_fast, optimizers[0]),
                        (w_old, b_old, weight_old, act_old, optimizers[1]),
                    ):
                        optimizer.zero_grad(set_to_none=True)
                        output = torch.nn.functional.linear(aq(x), wq(w), b)
                        torch.nn.functional.mse_loss(output, target).backward()
                        outputs.append(output)
                        optimizer.step()
                    self.assert_tensor_equal(*outputs)
                    self.assert_tensor_equal(w_fast.grad, w_old.grad)
                    self.assert_tensor_equal(b_fast.grad, b_old.grad)
                    self.assert_tensor_equal(w_fast, w_old)
                    self.assert_tensor_equal(b_fast, b_old)

    def test_inverted_range_validation_difference(self):
        # Record the intentional validation divergence for corrupt observer state.
        for device in self.devices():
            fast, old = self.make_pair(PAIRS[0], device)
            for module in (fast, old):
                module.observer.min_val.fill_(2.0)
                module.observer.max_val.fill_(1.0)
            with self.assertRaises(AssertionError):
                old.set_scale_and_zp()
            observer_qparams(fast.observer, fast._observer_quant_min_int, fast._observer_quant_max_int)

    def test_nonfinite_qparam_math(self):
        # PyTorch's scalar validation does NOT reject NaN (NaN > max is false).
        # Compare NaNs explicitly rather than using torch.equal on NaN tensors.
        for device in self.devices():
            for method in ("per_tensor", "per_tensor_symmetric"):
                for lo, hi in (
                    (float("nan"), 1.0), (-1.0, float("nan")),
                    (-float("inf"), float("inf")),
                    (-1.0, float("inf")), (-float("inf"), 1.0),
                ):
                    fast, old = self.make_pair(PAIRS[0], device, method)
                    fast.observer.min_val.fill_(lo)
                    fast.observer.max_val.fill_(hi)
                    old.observer.min_val.fill_(lo)
                    old.observer.max_val.fill_(hi)
                    scale, zp = observer_qparams(fast.observer, fast._observer_quant_min_int, fast._observer_quant_max_int)
                    old.set_scale_and_zp()
                    torch.testing.assert_close(scale, old.scale, rtol=0, atol=0, equal_nan=True)
                    self.assert_tensor_equal(zp, old.zero_point)

    def test_original_constructor_rejections(self):
        cases = [
            (1, torch.qint8, "per_tensor", False),
            (0, torch.qint8, "per_tensor", False),
            (8, torch.float32, "per_tensor", False),
            (8, torch.qint8, "unknown", False),
            (8, torch.quint8, "per_tensor_symmetric", True),
        ]
        for pair in PAIRS:
            for bits, dtype, method, reduced in cases:
                with self.subTest(cls=pair[0], bits=bits, dtype=dtype, method=method, reduced=reduced):
                    errors = []
                    for cls in pair:
                        args = (bits, dtype, method)
                        if pair[0] is not WeightQuantizer:
                            args += (None, None)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                cls(*args, reduce_range=reduced)
                            except (AssertionError, ValueError, NotImplementedError) as error:
                                errors.append(type(error))
                    self.assertEqual(len(errors), 2)
                    self.assertIs(errors[0], errors[1])

    def test_loading_into_already_used_modules_preserves_original_cache_policy(self):
        for device in self.devices():
            for pair in PAIRS:
                fast, old = self.make_pair(pair, device)
                self.compare_step(fast, old, torch.tensor([-1.0, 1.0], device=device))
                fast.eval()
                old.eval()
                source_fast, source_old = self.make_pair(pair, device)
                self.compare_step(source_fast, source_old, torch.tensor([-5.0, 5.0], device=device))
                fast.load_state_dict(source_old.state_dict(), strict=True)
                old.load_state_dict(source_fast.state_dict(), strict=True)
                self.compare_step(fast, old, torch.tensor([-2.0, 2.0], device=device))
                fast.train()
                old.train()
                self.compare_step(fast, old, torch.tensor([-8.0, 8.0], device=device))


if __name__ == "__main__":
    unittest.main()
