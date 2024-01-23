import gzip
import json
import logging
import math
import os
import random
import shutil
import tempfile
from typing import Generator, Iterator, List, Optional, Tuple, Union

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, ArrayLike, Float, Int, PyTree
import optax
from simple_pytree import dataclass, Pytree
from sisyphus import Job, Path, Task

from i6_core.lib.rasr_cache import FileArchiveBundle
from i6_core.util import chunks

from .model import ConformerEncoder
from .timer import Timer


@dataclass
class Batch(Pytree):
    device_types: Int[Array, "B"]
    seq_mask: Int[Array, "B T"]
    targets: Int[Array, "B T"]
    x: Float[Array, "B T X Y"]

    def log(self, name: str = "batch") -> "Batch":
        logging.info(f"{name}: {self.shapes()}")
        return self

    def shapes(self) -> str:
        return f"device_types={self.device_types.shape}, seq_mask={self.seq_mask.shape}, targets={self.targets.shape}, x={self.x.shape}"


class BatchLoader:
    def batches(self) -> Generator[Batch, None, None]:
        pass


class HdfBatchLoader(BatchLoader):
    """
    Loads batches of data previously prepared as HDF files.
    """

    batch_size: int
    data_hdfs: List[os.PathLike]

    def __init__(self, data_hdfs: List[os.PathLike], batch_size: int):
        assert len(data_hdfs) > 0
        assert batch_size > 0

        self.batch_size = batch_size
        self.data_hdfs = data_hdfs

    def batches(self, batch_size_override: Optional[int] = None) -> Generator[Batch, None, None]:
        batch_size = batch_size_override or self.batch_size

        for file in self.data_hdfs:
            with h5py.File(file, "r") as hdf:
                batches = hdf["batches"]
                keys = ["features", "seq_mask", "targets", "device_types"]

                num_batches = int(math.ceil(batches[keys[0]].shape[0] / batch_size))

                for i in range(num_batches):
                    start = int(i * batch_size)
                    end = int(start + batch_size)
                    features, seq_mask, targets, device_types = [batches[key][start:end] for key in keys]
                    yield Batch(
                        device_types=jax.device_put(device_types),
                        seq_mask=jax.device_put(seq_mask),
                        targets=jax.device_put(targets),
                        x=jax.device_put(features),
                    )


class EqxTrainingJob(Job):
    def __init__(
        self,
        data_hdf: List[Path],
        num_steps: int = 900e3,
        peak_lr: float = 1e-3,
    ):
        assert len(data_hdf) > 0

        self.data_hdf = data_hdf
        self.num_steps = int(num_steps)
        self.peak_lr = peak_lr

        self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)
        self.out_loss_file = self.output_path("loss.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 1, "mem": 8, "gpu": 1})

    def run(self):
        assert jax.device_count("gpu") > 0, "no JAX GPU devices found"

        logging.info(f"trainer starting up, loading model")
        logging.info(f"default device {jnp.ones(3).device_buffer.device()}")

        train_data_loader = HdfBatchLoader(data_hdfs=self.data_hdf, batch_size=30)

        key = jrandom.PRNGKey(42)
        k0, k1 = jrandom.split(key)
        classifier = model()

        logging.info(f"training for {self.num_steps} steps")
        self.train(
            classifier,
            train_data_loader,
            steps=self.num_steps,
            key=k1,
            out_checkpoints_folder=self.out_checkpoints_dir,
            out_loss_file=self.out_loss_file,
            peak_lr=self.peak_lr,
            print_every_step=self.num_steps // 100,
        )

    def train(
        self,
        classifier: eqx.Module,
        data_loader: BatchLoader,
        # dev_data_loader: BatchLoader,
        steps: int,
        key: jrandom.KeyArray,
        out_checkpoints_folder: Union[Path, str],
        out_loss_file: Union[Path, str],
        peak_lr: float,
        print_every_step: int = 5,
        optim: Optional[optax.GradientTransformation] = None,
        opt_state: Optional[optax.OptState] = None,
    ) -> eqx.Module:
        if optim is None:
            optim = optax.adamw(peak_lr)
        if opt_state is None:
            opt_state = optim.init(eqx.filter(classifier, eqx.is_array))
        print_every_step = max(print_every_step, 5)

        @eqx.filter_value_and_grad
        def compute_loss(classifier, inputs: Batch, key: jrandom.KeyArray) -> jnp.array:
            b = inputs.x.shape[0]
            batch_keys = jrandom.split(key, b)
            logits = eqx.filter_vmap(classifier)(inputs.x, inputs.device_types, inputs.seq_mask, batch_keys)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=inputs.targets)
            return jnp.mean(loss)

        @eqx.filter_jit
        def make_step(
            classifier: eqx.Module,
            opt: optax.GradientTransformation,
            opt_state: PyTree,
            inputs: Batch,
            key: jrandom.KeyArray,
        ) -> Tuple[eqx.Module, PyTree, jnp.array]:
            loss_value, grad = compute_loss(classifier, inputs, key)
            updates, opt_state = opt.update(grad, opt_state, classifier)
            classifier = eqx.apply_updates(classifier, updates)

            return classifier, opt_state, loss_value

        data_load_t = Timer("t_data_load")
        eval_t = Timer("t_eval")
        step_t = Timer("t_step")
        train_t = Timer("t_train")
        write_ckpt_t = Timer("t_wckpt")

        def data():
            while True:
                yield from data_loader.batches()
                logging.info("epoch marker")

        keys = jrandom.split(key, steps)
        train_data = data()

        train_losses = {}

        for step, key in zip(range(1, steps + 1), keys):
            k0, k1 = jrandom.split(key)

            with step_t.enter():
                with data_load_t.enter():
                    inputs = next(train_data)
                with train_t.enter():
                    try:
                        classifier, opt_state, train_loss = make_step(classifier, optim, opt_state, inputs, k0)
                    except Exception as e:
                        logging.error(f"training failed on step {step} at batch {inputs.shapes()}: {e}")

                        keys = jrandom.split(k0, inputs.x.shape[0])
                        jaxpr = jax.make_jaxpr(eqx.filter_vmap(classifier))(
                            inputs.x, inputs.device_types, inputs.seq_mask, keys
                        )
                        with open("jaxpr.log", "wt") as pr_file:
                            pr_file.write(str(jaxpr))

                        raise e

            train_losses[step] = train_loss.item()

            if step in [0, 1] or step % print_every_step == 0 or step >= steps - 5:
                percent_train = round(100 * (train_t.value() / step_t.value()), 1)

                with eval_t.enter():
                    # test_loss, acc, pcr, rcl = evaluate(classifier, dev_data_loader, k1)
                    test_loss, acc, pcr, rcl = 0, 0, 0, 0
                    pass

                last_steps = set((max(v, 1) for v in range(step - print_every_step, step)))
                last_losses = [train_losses[s] for s in last_steps]
                loss_avg = round(sum(last_losses) / len(last_losses), 5)

                parts = [
                    f"step={step}",
                    f"train_loss={loss_avg}",
                    f"test_loss={round(test_loss, 5)}",
                    f"test_accuracy={round(acc, 5)}",
                    f"test_precision={round(pcr, 5)}",
                    f"test_recall={round(rcl, 5)}",
                    str(step_t),
                    str(data_load_t),
                    str(eval_t),
                    str(train_t),
                    f"t_train={percent_train}%",
                ]
                logging.info(", ".join(parts))

                with write_ckpt_t.enter():
                    out_path = os.path.join(out_checkpoints_folder, f"model.{step}.ckpt.gz")
                    with gzip.open(out_path, "wb") as out_file:
                        eqx.tree_serialise_leaves(out_file, classifier)

                    with open(out_loss_file, "wt") as file:
                        json.dump(train_losses, file, indent=2)

        logging.info(f"done, {train_t}, {write_ckpt_t}")

        return classifier


def chunk_array(
    data: Float[Array, "X ..."],
    size: int,
    offset: int,
    fill_value: ArrayLike = 0.0,
) -> Float[Array, "B C ..."]:
    """
    Applies a sliding window on `data` returning chunks with size `chunk_size`
    and `chunk_offset` elements between them.

    The last chunks are padded to its size with `fill_value`.
    """

    x = data.shape[0]
    num_chunks = max(1 + math.ceil((x - size) / offset), 1)

    i = jnp.arange(size)  # C
    i = jnp.expand_dims(i, axis=0)  # 1 C
    i = jnp.repeat(i, num_chunks, axis=0)  # B C

    off = jnp.arange(num_chunks) * offset  # B
    off = jnp.expand_dims(off, axis=1)  # B 1

    ch_i = i + off  # B C
    out_chunks = data.at[ch_i, ...].get(mode="fill", fill_value=fill_value)  # B C Y

    return out_chunks


class FeaturesToHdfJob(Job):
    out_hdfs: List[Path]

    def __init__(
        self,
        gt_feature_bundle: Path,
        alignment_bundle: Path,
        allophones: Path,
        state_tying: Path,
        seq_len: int,
        seq_overlap: int,
        out_num_hdfs: int = 50,
    ):
        assert seq_len > seq_overlap > 0
        assert out_num_hdfs > 0

        self.alignment_bundle = alignment_bundle
        self.allophones = allophones
        self.gt_feature_bundle = gt_feature_bundle
        self.num_hdfs = out_num_hdfs
        self.seq_len = seq_len
        self.seq_overlap = seq_overlap
        self.state_tying = state_tying

        self.out_hdfs = [self.output_path(f"data.hdf.{i}") for i in range(out_num_hdfs)]

    def tasks(self) -> Iterator[Task]:
        yield Task("run", args=list(range(self.num_hdfs)), rqmt={"cpu": 1, "mem": 2})

    def run(self, i: int):
        cpu_device = jax.devices("cpu")[0]
        with jax.default_device(cpu_device):
            with tempfile.TemporaryDirectory() as tmpdir:
                out_file = os.path.join(tmpdir, "data.hdf")
                self.run_(i, out_file)
                shutil.move(out_file, self.out_hdfs[i])

    def run_(self, i: int, out_file: str):
        def extend_dset(dset: h5py.Dataset, x: jnp.array):
            prev_x = dset.shape[0]
            dset.resize(prev_x + x.shape[0], axis=0)
            dset[prev_x:] = x

        rng = random.Random()
        rng.seed(42)

        with open(self.state_tying, "rt") as st:
            state_tying = {k: int(v) for line in st for k, v in [line.strip().split()[:2]]}

        a_bundle = FileArchiveBundle(self.sh(f"cf {self.alignment_bundle}"))
        a_bundle.setAllophones(self.allophones.get_path())

        f_bundle = FileArchiveBundle(self.sh(f"cf {self.gt_feature_bundle}"))
        segments = [f for f in f_bundle.file_list() if ".attribs" not in f]
        rng.shuffle(segments)
        segments = list(chunks(segments, len(self.out_hdfs)))[i]

        with h5py.File(out_file, "w") as hdf:
            out_f = hdf.create_dataset(
                "features",
                maxshape=(0, self.seq_len, 50),
                shape=(0, self.seq_len, 50),
            )
            out_sm = hdf.create_dataset(
                "seq_mask",
                maxshape=(0, self.seq_len),
                shape=(0, self.seq_len),
                dtype="i1",
            )
            out_t = hdf.create_dataset(
                "targets",
                maxshape=(0, self.seq_len),
                shape=(0, self.seq_len),
                dtype="i4",
            )

            for segment in segments:
                alignment = a_bundle.read(segment, "align")
                times, features = f_bundle.read(segment, "feat")

                alignment_states = [f"{a_bundle.files[segment].allophones[t[1]]}.{t[2]:d}" for t in alignment]
                targets = [state_tying[allophone] for allophone in alignment_states]
                seq_len = len(targets)

                # pad it to alignment length to avoid len mismatches
                features = features.resize((len(targets), features.shape[1]))

                feature_chunks = chunk_array(jnp.array(features), self.seq_len, self.seq_overlap)
                seq_mask_chunks = chunk_array(jnp.ones(seq_len), self.seq_len, self.seq_overlap)
                target_chunks = chunk_array(jnp.array(targets), self.seq_len, self.seq_overlap)

                extend_dset(out_f, feature_chunks)
                extend_dset(out_sm, seq_mask_chunks)
                extend_dset(out_t, target_chunks)


def run():
    hdf_job = FeaturesToHdfJob(
        alignment_bundle=Path("ALIGNMENT"),
        allophones=Path("ALLOPHONES"),
        gt_feature_bundle=Path("TRAIN_FEATURES_GT50"),
        state_tying=Path("TYING"),
        seq_len=400,
        seq_overlap=200,
    )
    train_job = EqxTrainingJob(hdf_job.out_hdfs)


if __name__ == "__main__":
    run()
