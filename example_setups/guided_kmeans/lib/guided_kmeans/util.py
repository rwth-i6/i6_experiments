import time
from time import strftime, gmtime
import timeit
from datetime import timedelta
import json
from functools import partial

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def window_time_np(x: np.ndarray, stride: int, kernel_size: int | None = None) -> np.ndarray:
    """
    x: (T, D) numpy array
    returns: (T', D) max-pooled over time
    """
    if x.ndim != 2:
        raise ValueError(f"expected (T, D), got {x.shape}")

    k = stride if kernel_size is None else kernel_size
    T, D = x.shape
    if k <= 0 or stride <= 0:
        raise ValueError("kernel_size and stride must be positive")
    if T < k:
        return x.max(axis=0, keepdims=True)  # (1, D)

    # (T-k+1, D, k)
    w = sliding_window_view(x, window_shape=k, axis=0)
    # (T', D, k)
    w = w[::stride]

    # max over the window axis -> (T', D)
    return w

class PoolingRegistry:
    @staticmethod
    def maxpool_time_np(x: np.ndarray, stride: int, kernel_size: int | None = None) -> np.ndarray:
        w = window_time_np(x, stride, kernel_size)
        return w.max(axis=-1)

    @staticmethod
    def avg_time_np(x: np.ndarray, stride: int, kernel_size: int | None = None) -> np.ndarray:
        w = window_time_np(x, stride, kernel_size)
        return w.mean(axis=-1)

    @staticmethod
    def normed_avg_time_np(x: np.ndarray, stride: int, kernel_size: int | None = None) -> np.ndarray:
        x_norm = np.linalg.norm(x, axis=-1).mean()
        w = window_time_np(x, stride, kernel_size)
        wdw_avg = w.mean(axis=-1)
        return x_norm * wdw_avg / np.linalg.norm(wdw_avg, axis=-1, keepdims=True)
    
    @staticmethod
    def cosine_medoid_pool_np(x, stride, kernel_size):
        w = window_time_np(x, stride, kernel_size)
        w = np.swapaxes(w, 1, 2)  # (T', k, D)
        C = w @ np.swapaxes(w, 1, 2)  # (T', k, k)
        idx = C.mean(axis=2).argmax(axis=1)  # (T',)
        y = w[np.arange(w.shape[0]), idx]    # (T', D)
        # if inputs are normalized, y already is
        return y

    @staticmethod
    def sample_pool_time_np(x: np.ndarray, stride: int, kernel_size: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        w = window_time_np(x, stride, kernel_size)
        if rng is None:
            rng = np.random.default_rng()
        sample_idxs = rng.integers(0, kernel_size, size=x.shape[0])
        w_sampled = np.take_along_axis(
            w,
            sample_idxs[:, None, None],
            axis=-1
        ).squeeze(-1)
        return w_sampled

    @staticmethod
    def maxpool_time_np_pad(x: np.ndarray, stride: int, kernel_size: int | None = None, pad_value=-np.inf):
        k = stride if kernel_size is None else kernel_size
        T, D = x.shape
        # pad so we can take an integer number of strides of windows
        # enough padding so that (T_pad - k) is divisible by stride
        rem = (T - k) % stride
        pad = 0 if rem == 0 else (stride - rem)
        if pad > 0:
            x = np.pad(x, ((0, pad), (0, 0)), constant_values=pad_value)
        return PoolingRegistry.maxpool_time_np(x, stride=stride, kernel_size=k)
    
    @classmethod
    def select(cls, name: str, **pooling_opts):
        if not hasattr(cls, name):
            raise ValueError("No pooling function with this name")
        pooling_func = getattr(cls, name)
        return partial(
            pooling_func, **pooling_opts
        )


# Source - https://stackoverflow.com/a
# Posted by Aravind Voggu, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-18, License - CC BY-SA 4.0
def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)


def segments_to_array(segments: np.ndarray) -> np.ndarray:
    """
    Turn segment information into an array expressing the segments.
    
    :param segments: [N, 3] array
    """
    values = segments[:, 0]
    repeats = segments[:, 2] - segments[:, 1]
    return np.repeat(values, repeats)

def _test_segments_to_array():
    segments = [
        (2, 0, 3),
        (4, 3, 4),
        (42, 4, 10)
    ]
    out_list = [2] * 3 + [4] * 1 + [42] * 6

    res = segments_to_array(np.asarray(segments))

    print(res)
    print(out_list)
    assert out_list == res.tolist()

class ProgressLogger:
    def __init__(self, total_num_steps, logging_step=512, bar_length=20):
        self.start_time: float = 0.0
        self.total_num_steps = total_num_steps
        self.logging_step = logging_step
        self.bar_length = bar_length
    
    def start(self):
        self.start_time = timeit.default_timer()
    
    def progress(self, step):
        if step < self.total_num_steps - 1 and step % self.logging_step != 0:
            # print("Not pringing")
            return
        
        fraction = (step + 1) / self.total_num_steps

        arrow = int(fraction * self.bar_length - 1) * '-' + '>'
        padding = int(self.bar_length - len(arrow)) * ' '

        ending = '\n' if fraction == 1.0 else '\r'

        elapsed_time = str(timedelta(seconds=timeit.default_timer() - self.start_time))
        elapsed_time = timeit.default_timer() - self.start_time

        num_steps_str = str(self.total_num_steps)
        num_steps_str_len = len(num_steps_str)
        abs_step = step + 1
        abs_str = f"{abs_step:>{num_steps_str_len}}/{num_steps_str}"

        time_str = str(strftime("%H:%M:%S", gmtime(elapsed_time)))
        print(f'Progress: [{arrow}{padding}] {int(fraction*100):>3}% | {abs_str} | {time_str}', end=ending)
    
class TracebackLogger:
    def __init__(self):
        self.data = []
        self.seq_id = 0

    def feed(self, traceback: list):
        for it in traceback:
            data_line = {
                "seq_id": self.seq_id,
                "lemma": it.lemma,
                "am": it.am_score,
                "lm": it.lm_score,
                "start": it.start_time,
                "end": it.end_time,
            }

            self.data.append(data_line)
            self.seq_id += 1
        
    def finalize(self):
        with open("traceback.log", "w+") as fp:
            json.dump(self.data, fp)


def _test_progress_logger():
    num_steps = 1_000_000
    prog = ProgressLogger(num_steps)
    prog.start()

    for i in range(num_steps):
        time.sleep(0.000001)
        prog.progress(i)

if __name__ == "__main__":
    _test_segments_to_array()
    _test_progress_logger()