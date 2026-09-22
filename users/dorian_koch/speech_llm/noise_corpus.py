"""Noise corpora for audio augmentation.

``DnsNoisePool`` stages the DNS-Challenge 5 noise set (Dubey et al., ICASSP 2023), which is the set the
Moshi paper (§4.4) draws its user-stream noise from. The files and URL are those of Microsoft's own
download script (`download-dns-challenge-5-noise-ir.sh` in microsoft/DNS-Challenge): the fullband
noise from AudioSet and Freesound, ~39 GB compressed. The impulse responses in the same script are not
staged -- the paper's reverb is RevEcho, which needs none.
"""

from __future__ import annotations

import json
import os

from sisyphus import Job, Task, tk

from i6_experiments.users.dorian_koch.speech_llm.common import run_worker_script

DNS5_BASE_URL = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"
DNS5_NOISE_FILES = (
    *(f"noise_fullband/datasets_fullband.noise_fullband.audioset_{i:03d}.tar.bz2" for i in range(7)),
    *(f"noise_fullband/datasets_fullband.noise_fullband.freesound_{i:03d}.tar.bz2" for i in range(2)),
)


class DnsNoisePool(Job):
    """DNS-Challenge noise as memory-mappable int16 clips at ``sample_rate`` (one arrow file per tarball).

    Read with ``moshi_family.train_data_common.NoisePool``. CPU only; the tarballs are streamed and
    decoded in memory, so the only files written are the outputs (see ``dns_noise_main.py``).
    """

    def __init__(
        self,
        *,
        mimi_venv_python,
        sample_rate: int = 24000,
        files: tuple[str, ...] = DNS5_NOISE_FILES,
        base_url: str = DNS5_BASE_URL,
        rqmt: dict | None = None,
    ):
        self.mimi_venv_python = mimi_venv_python
        self.sample_rate = int(sample_rate)
        self.files = tuple(files)
        self.base_url = base_url
        self.out_dir = self.output_path("pool", directory=True)
        # One process per tarball in flight: bz2 decompression is single-threaded and the bottleneck.
        self.rqmt = rqmt or {"cpu": 9, "mem": 32, "time": 12}

    @classmethod
    def hash(cls, parsed_args):
        d = dict(parsed_args)
        d.pop("rqmt", None)
        return super().hash(d)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def _n_staged(self) -> int:
        out = self.out_dir.get()
        return sum(p.endswith(".arrow") for p in os.listdir(out)) if os.path.isdir(out) else 0

    def completed_fraction(self):
        return self._n_staged() / len(self.files)

    def info(self):
        try:
            with open(os.path.join(self.out_dir.get(), "meta.json")) as f:
                meta = json.load(f)
            return f"{meta['n_clips']} clips, {meta['hours']} h"
        except OSError:
            return f"{self._n_staged()}/{len(self.files)} tarballs staged"

    def run(self):
        from i6_experiments.users.dorian_koch.speech_llm.podcast_ingest import _moshi_family_lib_parent

        run_worker_script(
            self.mimi_venv_python.get(),
            os.path.join(_moshi_family_lib_parent(), "moshi_family", "dns_noise_main.py"),
            [
                "--out_dir",
                self.out_dir.get(),
                "--base_url",
                self.base_url,
                "--files_json",
                json.dumps(list(self.files)),
                "--sample_rate",
                self.sample_rate,
                "--workers",
                self.rqmt["cpu"],
            ],
            log_label="DNS noise pool",
            with_hf_home=False,
        )


def dns_noise_pool(mimi_venv_python, *, sample_rate: int = 24000) -> tk.Path:
    """The staged DNS-Challenge 5 noise directory, registered as ``noise/dns5_<sample_rate>``."""
    job = DnsNoisePool(mimi_venv_python=mimi_venv_python, sample_rate=sample_rate)
    tk.register_output(f"noise/dns5_{sample_rate}", job.out_dir)
    return job.out_dir
