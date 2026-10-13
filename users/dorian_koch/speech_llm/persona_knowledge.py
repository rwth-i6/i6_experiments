"""Synthetic knowledge dialogues as PersonaPlex training rows (ablation ab_epad1_know30, 2026-09-25).

User (2026-09-25): "mix in 30% in-batch of the synthetic knowledge dialogues. Think of something smart to make
them work with personaplex." PersonaPlex trains on rows that carry a HYBRID system prompt (voice + role text);
the TTS'd TriviaQA-train dialogues carry neither. ``KnowledgeDialogueCodes`` gives them both, the way the
PersonaPlex paper built its own QA data (arXiv 2602.06053, section 3.2):

  * role text: the paper's fixed QA role, verbatim -- "We use a fixed role for this dataset: 'You are a wise and
    friendly teacher. Answer questions or provide advice in a clear and engaging way.'" It is also the engine's
    DEFAULT_SYSTEM_PROMPT, i.e. the prompt the knowledge benchmark runs every PersonaPlex model with, so the
    knowledge rows teach "teacher prompt -> answer" while the JRE rows keep their podcast personas;
  * voice: the paper clones "a randomly chosen voice sample for each role" with Chatterbox and uses that sample as
    the agent's voice prompt. Our corpus clones ONE assistant voice (``pipelines.DEFAULT_SPEAKER_ALIAS``), so every
    knowledge row gets that reference clip as its voice prompt, encoded like the JRE voices. It is not cut from any
    dialogue, so it cannot leak a row's content;
  * audio + text: the JRE rows' conventions (loudness, offline Mimi encode) plus the A21 clamp that a codes
    corpus can only get at build time. See the worker ``moshi_family.knowledge_codes`` for each step.
"""

from __future__ import annotations

import json
import os
import subprocess

from sisyphus import Job, Task, tk

from .common import job_progress_fraction, worker_env

#: PersonaPlex's QA role, verbatim from the paper (section 3.2.1) and ``personaplex.engine.DEFAULT_SYSTEM_PROMPT``.
PERSONAPLEX_QA_ROLE = (
    "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
)


class KnowledgeDialogueCodes(Job):
    """TTS'd knowledge dialogues (``MoshiAnnotate`` waveform corpora) -> one PersonaPlex codes corpus.

    ``corpora``: ``[(name, annotated dataset dir)]``, concatenated in this order (every row of each; for the v6
    corpus the per-template weights are the slices' row shares, so all rows = the merged corpus). ``voice_dir`` +
    ``voice_alias``: the TTS speaker dir and the assistant's alias (``<dir>/<alias>.wav``, the clip Chatterbox
    cloned). ``system_prompt``: the role text of every row. ``loudness_norm``: the ingest's dict
    (``synthetic_train_data.PODCAST_INGEST["loudness_norm"]``). Output ``out_dir`` (dataset) + ``out_summary``.

    v7 (user, 2026-09-28: random assistant voices + the JRE user augmentation), all three opt-in and hash-excluded
    when None (the v6 job keeps its hash):
      * ``assistant_audio``: ``[(name, KnowledgeVoiceConvert.out_dir)]`` with the SAME names as ``corpora``: each
        row's assistant channel is read from there (row-aligned, ids checked) instead of the annotated corpus;
      * ``voice_pool``: ``KnowledgeVoicePool.out_dir`` (then ``voice_dir`` / ``voice_alias`` must be None): each
        row's voice prompt = its pool voice's codes (the ``voice`` the conversion recorded), plus a ``voice_id``
        column (``moshi_family.knowledge_concat`` groups by it);
      * ``user_augment`` (+ ``noise_pool``): the ingest's ``{"variants", "seed", "augment"}`` dict: stored augmented
        variants of the USER channel (``codes_user_aug``, ``n_user_aug``, ``user_aug_json``) made by the ingest's
        own ``podcast_mimi_encode.augment_user_variants`` from the TTS waveform, encoded like the clean codes.

    ``user_words`` (user, 2026-10-12: "make sure that for future synthetic and knowledge generations we always also
    generate the word timings"; ON for every new declaration): the rows also carry the TTS questioner's words --
    ``alignments_other`` + ``other_ok``, the JRE windows' layout, which the user-text head reads -- taken from the
    annotated corpora's own ``alignments_other`` (``MoshiAnnotate(user_words=True)``: Whisper on the user channel),
    clamped and cut as the assistant's. Every input corpus must carry them (the worker refuses otherwise). Hashed only
    when True: ``user_words=False`` = the v6 / v7 jobs (declared before; those call sites pin it).
    """

    def __init__(
        self,
        *,
        corpora: list,
        voice_dir: tk.Path | None,
        voice_alias: str | None,
        venv_python_path: tk.Path,
        loudness_norm: dict,
        system_prompt: str = PERSONAPLEX_QA_ROLE,
        voice_sec: float = 4.0,
        max_sec: float = 60.0,
        assistant_audio: list | None = None,
        voice_pool: tk.Path | None = None,
        user_augment: dict | None = None,
        noise_pool: tk.Path | None = None,
        user_words: bool = True,
    ):
        names = [n for n, _ in corpora]
        assert len(set(names)) == len(names), f"corpus names must be unique: {names}"
        self.corpora = [(str(n), p) for n, p in corpora]
        assert (voice_pool is None) == (voice_dir is not None and voice_alias is not None), (
            "exactly one voice source: voice_dir + voice_alias (one voice) or voice_pool (per-row voices)"
        )
        self.voice_dir = voice_dir
        self.voice_alias = None if voice_alias is None else str(voice_alias)
        self.venv_python_path = venv_python_path
        self.loudness_norm = dict(loudness_norm)
        self.system_prompt = str(system_prompt)
        self.voice_sec = float(voice_sec)
        self.max_sec = float(max_sec)
        if assistant_audio is not None:
            assert [str(n) for n, _ in assistant_audio] == names, "assistant_audio names must match corpora, in order"
            assistant_audio = [(str(n), p) for n, p in assistant_audio]
        assert voice_pool is None or assistant_audio is not None, "voice_pool reads the voices assistant_audio used"
        self.assistant_audio = assistant_audio
        self.voice_pool = voice_pool
        self.user_augment = None if user_augment is None else dict(user_augment)
        if self.user_augment is not None:
            assert "noise_pool" not in self.user_augment, "pass the noise pool as noise_pool= (a tk.Path)"
            draws_noise = float(self.user_augment.get("augment", {}).get("noise_prob", 0.0)) > 0.0
            assert draws_noise == (noise_pool is not None), "noise_pool goes with a nonzero noise_prob"
        else:
            assert noise_pool is None, "noise_pool without user_augment"
        self.noise_pool = noise_pool
        self.user_words = bool(user_words)
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_summary = self.output_path("summary.json")
        # One GPU for the offline Mimi encode (~327 h of stereo audio for the v6 corpus); decode, loudness and the
        # clamp run in CPU threads beside it. Small GPU memory: Mimi alone. With user_augment: 1 + variants encodes per
        # row and the augmentation in the CPU threads (the noise pool is staged on node-local disk first).
        self.rqmt = {"gpu": 1, "cpu": 8, "mem": 48, "time": 8 if user_augment is None else 16, "gpu_mem_gb": 20}

    @classmethod
    def hash(cls, parsed_args):
        d = dict(parsed_args)
        for k in ("assistant_audio", "voice_pool", "user_augment", "noise_pool"):
            if d.get(k) is None:
                d.pop(k, None)  # v7's opt-ins: absent at the default, so the v6 corpus keeps its hash
        if not d.get("user_words"):
            d.pop("user_words", None)  # False = the v6 / v7 jobs (before 2026-10-12): their hashes are unchanged
        return super().hash(d)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        return job_progress_fraction(self)

    def info(self):
        f = job_progress_fraction(self)
        return None if f is None else f"{100 * f:.1f}% of rows encoded"

    def run(self):
        env = worker_env(offline=False)
        if getattr(self, "voice_pool", None) is not None:
            voice_args = ["--voice_pool", self.voice_pool.get_path()]
        else:
            voice_wav = os.path.join(self.voice_dir.get_path(), self.voice_alias + ".wav")
            if not os.path.isfile(voice_wav):
                raise RuntimeError(f"voice clip not found: {voice_wav}")
            voice_args = ["--voice_wav", voice_wav]
        extra = []
        for n, p in getattr(self, "assistant_audio", None) or []:
            extra += ["--assistant_audio", f"{n}={p.get_path()}"]
        if getattr(self, "user_augment", None) is not None:
            aug = dict(self.user_augment)
            if self.noise_pool is not None:
                aug["noise_pool"] = self.noise_pool.get_path()
            extra += ["--user_augment", json.dumps(aug, sort_keys=True)]
        if getattr(self, "user_words", False):
            extra += ["--user_words"]
        cmd = [
            self.venv_python_path.get(),
            "-m",
            "moshi_family.knowledge_codes",
            "--out",
            self.out_dir.get_path(),
            *voice_args,
            *extra,
            "--voice_sec",
            str(self.voice_sec),
            "--system_prompt",
            self.system_prompt,
            "--loudness_norm",
            json.dumps(self.loudness_norm, sort_keys=True),
            "--max_sec",
            str(self.max_sec),
            "--device",
            "cuda",
            "--progress",
            "progress.json",
        ] + [f"{n}={p.get_path()}" for n, p in self.corpora]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
        # the worker writes summary.json beside the dataset dir, i.e. into output/
        if not os.path.isfile(self.out_summary.get_path()):
            raise RuntimeError(f"worker wrote no {self.out_summary.get_path()}")


class KnowledgeConcatMeta(Job):
    """Per-row turn metadata of a ``KnowledgeDialogueCodes`` corpus for concatenating its dialogues back to back
    (ablation ab_epad1_know30_concat, 2026-09-25; worker ``moshi_family.knowledge_concat``).

    The annotated corpora keep only the assistant's words; the TTS outputs keep each turn's speaker / start / end.
    Joined by id (``<name>/<TTS id>``), per codes row: the user's first-turn start, the last (assistant) turn's
    start, the answer end (min of the last word's end and the last turn's end) and ``ok`` (first turn = user, last =
    assistant, last turn ends at the row's last frame +- 2 frames, the last answer has words). Plus the silence
    filler (offline Mimi codes of digital zeros, mid-signal). CPU only, ~minutes (no audio is decoded).

    ``codes``: ``KnowledgeDialogueCodes.out_dir``; ``tts``: ``[(name, ChatterboxInference output)]`` with the SAME
    names as the codes job's corpora. Output ``out_meta`` (meta.npz, row-aligned with ``codes``), ``out_summary``.
    """

    def __init__(self, *, codes: tk.Path, tts: list, venv_python_path: tk.Path):
        names = [n for n, _ in tts]
        assert len(set(names)) == len(names), f"tts names must be unique: {names}"
        self.codes = codes
        self.tts = [(str(n), p) for n, p in tts]
        self.venv_python_path = venv_python_path
        self.out_meta = self.output_path("meta.npz")
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"cpu": 2, "mem": 16, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        env = worker_env(offline=False)
        cmd = [
            self.venv_python_path.get(),
            "-m",
            "moshi_family.knowledge_concat",
            "--codes",
            self.codes.get_path(),
            "--out",
            self.out_meta.get_path(),
            "--summary",
            self.out_summary.get_path(),
        ] + [f"{n}={p.get_path()}" for n, p in self.tts]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)


class KnowledgeUserWordsCTC(Job):
    """The USER side's word timings of TTS'd knowledge dialogues by CTC forced alignment of the KNOWN script
    (backfill; user, 2026-10-13: "just backfill ... Lets do user word timings on the knowledge rows too"; worker
    ``moshi_family.ctc_user_words``: torchaudio's MMS_FA aligner as published, per user turn of the TTS script on the
    annotated user channel inside the turn's TTS span). Chosen over the annotate stage's Whisper user pass
    (``MoshiAnnotateUserWords``) on cost: that pass measured ~1x the assistant pass, ~10 GPU-h for the 60,000 v6 rows,
    above the ~6 GPU-h bound; the two agree on onsets (220 seeded v6 rows: |CTC - Whisper| median 0.48 Mimi frames, 82%
    within 1 frame, 97% within 2; rl_alignment.md "User word timings").

    ``corpora``: ``[(name, annotated dir, ChatterboxInference dir)]``. Output per corpus ``out_sidecars[name]`` (the
    ``{id, alignments_other, other_ok}`` sidecar ``KnowledgeAttachUserWords`` reads, row-aligned with the annotated
    corpus) + ``out_summary``. The MMS_FA weights come from torch hub (``~/.cache/torch``)."""

    def __init__(self, *, corpora: list, venv_python_path: tk.Path):
        names = [c[0] for c in corpora]
        assert len(set(names)) == len(names), f"corpus names must be unique: {names}"
        assert all(len(c) == 3 for c in corpora), "corpora: [(name, annotated dir, TTS dir)]"
        self.corpora = [(str(n), a, t) for n, a, t in corpora]
        self.venv_python_path = venv_python_path
        self.out_root = self.output_path("sidecars", directory=True)
        self.out_sidecars = {n: self.out_root.join_right(n) for n in names}
        self.out_summary = self.out_root.join_right("summary.json")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 6, "gpu_mem_gb": 20}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        return job_progress_fraction(self)

    def info(self):
        f = job_progress_fraction(self)
        return None if f is None else f"{100 * f:.1f}% of rows aligned"

    def run(self):
        env = worker_env(offline=False)
        cmd = [
            self.venv_python_path.get(),
            "-m",
            "moshi_family.ctc_user_words",
            "--out_root",
            self.out_root.get_path(),
            "--device",
            "cuda",
            "--progress",
            "progress.json",
        ] + [f"{n}={a.get_path()}={t.get_path()}" for n, a, t in self.corpora]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
        if not os.path.isfile(self.out_summary.get_path()):
            raise RuntimeError(f"worker wrote no {self.out_summary.get_path()}")


class KnowledgeAttachUserWords(Job):
    """An existing knowledge codes corpus + the USER side's word timings (backfill; user, 2026-10-13: "just backfill
    ... Lets do user word timings on the knowledge rows too"; worker ``moshi_family.knowledge_user_words``).

    ``codes``: a ``KnowledgeDialogueCodes(user_words=False)`` dataset (v6); ``corpora``: ``[(name, annotated dir,
    MoshiAnnotateUserWords sidecar)]`` with the codes build's names, in its order; ``loudness_norm`` / ``max_sec``:
    the codes build's. Output ``out_dir`` = ``codes`` with every column bit-identical (nothing re-encoded) +
    ``alignments_other`` / ``other_ok`` exactly as ``KnowledgeDialogueCodes(user_words=True)`` writes them (the
    builder's own ``prepare_row``: the user words clamped to the user channel's speech onset, cut at ``max_sec``),
    + ``out_summary``. Fails unless the recomputed assistant words and durations equal the stored build's on every
    row. Row ids unchanged, so the codes corpus's ``KnowledgeConcatMeta`` (keyed by id, built from the assistant side)
    serves it as is. CPU only (decode + loudness of every row)."""

    def __init__(
        self,
        *,
        codes: tk.Path,
        corpora: list,
        loudness_norm: dict,
        max_sec: float,
        venv_python_path: tk.Path,
    ):
        names = [c[0] for c in corpora]
        assert len(set(names)) == len(names), f"corpus names must be unique: {names}"
        assert all(len(c) == 3 for c in corpora), "corpora: [(name, annotated dir, user-words sidecar)]"
        self.codes = codes
        self.corpora = [(str(n), a, s) for n, a, s in corpora]
        self.loudness_norm = dict(loudness_norm)
        self.max_sec = float(max_sec)
        self.venv_python_path = venv_python_path
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"cpu": 8, "mem": 48, "time": 6}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        return job_progress_fraction(self)

    def info(self):
        f = job_progress_fraction(self)
        return None if f is None else f"{100 * f:.1f}% of rows"

    def run(self):
        env = worker_env(offline=False)
        cmd = [
            self.venv_python_path.get(),
            "-m",
            "moshi_family.knowledge_user_words",
            "--codes",
            self.codes.get_path(),
            "--out",
            self.out_dir.get_path(),
            "--loudness_norm",
            json.dumps(self.loudness_norm, sort_keys=True),
            "--max_sec",
            str(self.max_sec),
            "--workers",
            str(self.rqmt["cpu"]),
            "--progress",
            "progress.json",
        ] + [f"{n}={a.get_path()}={s.get_path()}" for n, a, s in self.corpora]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
        if not os.path.isfile(self.out_summary.get_path()):
            raise RuntimeError(f"worker wrote no {self.out_summary.get_path()}")


class KnowledgeKnownScore(Job):
    """Which TriviaQA-train questions does base PersonaPlex already know? (ablation ab2_know_known, 2026-09-29; worker
    ``moshi_family.personaplex.knowledge_known_score``, whose docstring has the method.)

    User: "restrict training to only those examples that are already known by the base model" (Gekhman et al. 2024),
    "Make sure to normalize across length tho. We want to score the speech model tho, not some random text model".
    One teacher-forced forward of the RELEASED PersonaPlex per knowledge dialogue, read as the training loader reads a
    knowledge row; score = mean log-prob per token of the answer where the ASSISTANT first says it (TriviaQA's main
    answer; unscorable when never said or when the user says it first). Per question: the mean over its scorable
    dialogues. Known = the top ``quantile`` of the scored questions.

    ``codes``: ``KnowledgeDialogueCodes.out_dir``; ``dialogues``: the cleaned dialogue dataset the TTS read (question_id,
    answer, dialogue, template_name); ``tts``: ``[(name, ChatterboxInference output)]`` with the codes corpora's names
    (codes id = ``<name>/<TTS id>``; joined to the dialogue by slice + turn text). ``settings``: the text-stream / loss
    keys of a run config (``heldout_loss.loss_settings``). ``max_rows``: 0 = every row; N = N rows evenly spaced over the corpus (the smoke).
    Outputs: ``out_dir`` (dialogues.jsonl, questions.jsonl, known_questions.json, summary.json), ``out_rows`` (the
    ascending codes-row keep-list of every dialogue of a known question: ``knowledge_train_rows`` of an arm),
    ``out_summary``."""

    def __init__(
        self,
        *,
        codes: tk.Path,
        dialogues: tk.Path,
        tts: list,
        venv_python_path: tk.Path,
        settings: dict,
        quantile: float = 0.25,
        max_rows: int = 0,
        batch: int = 16,
        control_n: int = 400,
    ):
        names = [n for n, _ in tts]
        assert len(set(names)) == len(names), f"tts names must be unique: {names}"
        assert 0.0 < float(quantile) < 1.0, quantile
        self.codes = codes
        self.dialogues = dialogues
        self.tts = [(str(n), p) for n, p in tts]
        self.venv_python_path = venv_python_path
        self.settings = dict(settings)
        self.quantile = float(quantile)
        self.max_rows = int(max_rows)
        self.batch = int(batch)
        self.control_n = int(control_n)
        self.out_dir = self.output_path("score", directory=True)
        self.out_rows = self.output_path("known_rows.npy")
        # the same cut within each answer-length stratum (the per-token mean still favours longer answers)
        self.out_rows_len_strat = self.output_path("known_rows_len_strat.npy")
        self.out_summary = self.output_path("summary.json")
        # one 7B forward per 16 dialogues (~21 s each) in bf16; test node 2026-09-29: ~14 rows/s -> ~75 min for 60k
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 64, "time": 3, "gpu_mem_gb": 80}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        try:
            d = json.load(open(os.path.join(self.out_dir.get_path(), "progress.json")))
            return min(d["done"] / d["total"], 1.0)
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            return None

    def info(self):
        f = self.completed_fraction()
        return None if f is None else f"{100 * f:.1f}% of dialogues scored"

    def run(self):
        import shutil

        env = worker_env(offline=True)
        out = self.out_dir.get_path()
        cmd = [self.venv_python_path.get(), "-u", "-m", "moshi_family.personaplex.knowledge_known_score"]
        cmd += ["--codes", self.codes.get_path(), "--dialogues", self.dialogues.get_path(), "--out", out]
        cmd += ["--settings", json.dumps(self.settings, sort_keys=True), "--quantile", str(self.quantile)]
        cmd += ["--max_rows", str(self.max_rows), "--batch", str(self.batch), "--control_n", str(self.control_n)]
        for n, p in self.tts:
            cmd += ["--tts", f"{n}={p.get_path()}"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
        shutil.copyfile(os.path.join(out, "known_rows.npy"), self.out_rows.get_path())
        shutil.copyfile(os.path.join(out, "known_rows_len_strat.npy"), self.out_rows_len_strat.get_path())
        shutil.copyfile(os.path.join(out, "summary.json"), self.out_summary.get_path())


class KnowledgeVoicePool(Job):
    """The assistant voices of the knowledge corpus v7 (user, 2026-09-28: "Maybe we should also get different voices
    in there, to align with the jre training. Just use random voices ig."; worker ``moshi_family.knowledge_voices``,
    CPU, minutes).

    ``n`` voice clips drawn uniformly (``seed``) from the (episode, speaker) clips that ``train_data``'s rows (p4's
    AttachPersonaPrompts table) use as voice prompts, minus the episodes of ``heldout`` (the shared held-out set) and
    ``exclude_episodes`` (the evaluation episodes), with a source clip >= ``prompt_sec`` + 6 s. Per voice: its
    ``voice_codes`` rows' codes as the voice prompt (identical to JRE's) and the same reference clip's
    ``[prompt_sec, prompt_sec + 10 s)`` as the cloning reference (same speaker, not the prompt audio).
    ``episode_shard_dirs``: the ingest shards whose episode rows hold the reference clips (``speaker_ref_audio``).
    Output ``out_dir`` (one row per voice) + ``out_summary``.
    """

    def __init__(
        self,
        *,
        voice_codes: tk.Path,
        train_data: tk.Path,
        heldout: tk.Path,
        exclude_episodes: list,
        episode_shard_dirs: list,
        venv_python_path: tk.Path,
        n: int = 1000,
        seed: int = 0,
        prompt_sec: float = 4.0,
    ):
        self.voice_codes = voice_codes
        self.train_data = train_data
        self.heldout = heldout
        self.exclude_episodes = sorted(str(e) for e in exclude_episodes)
        self.episode_shard_dirs = list(episode_shard_dirs)
        self.venv_python_path = venv_python_path
        self.n = int(n)
        self.seed = int(seed)
        self.prompt_sec = float(prompt_sec)
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"cpu": 2, "mem": 16, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        env = worker_env(offline=False)
        excl = os.path.abspath("exclude_episodes.json")
        with open(excl, "w") as f:
            json.dump(self.exclude_episodes, f)
        cmd = [
            self.venv_python_path.get(),
            "-m",
            "moshi_family.knowledge_voices",
            "--out",
            self.out_dir.get_path(),
            "--summary",
            self.out_summary.get_path(),
            "--voice_codes",
            self.voice_codes.get_path(),
            "--train_data",
            self.train_data.get_path(),
            "--heldout",
            self.heldout.get_path(),
            "--exclude_episodes_json",
            excl,
            "--n",
            str(self.n),
            "--seed",
            str(self.seed),
            "--prompt_sec",
            str(self.prompt_sec),
        ] + [d.get_path() for d in self.episode_shard_dirs]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)


class KnowledgeVoiceConvert(Job):
    """One knowledge slice's ASSISTANT channel re-voiced with the TTS's own decoder (knowledge corpus v7, 2026-09-28;
    worker ``knowledge_voice_convert.py`` in the chatterbox venv, one GPU).

    Per row a pool voice (``pool``: ``KnowledgeVoicePool.out_dir``; balanced, seeded by ``seed`` + ``name``); per
    assistant turn (``tts``: the slice's ``ChatterboxInference`` output, for the turn spans) Chatterbox's voice
    conversion with the Turbo S3Gen decoder that made v6 -- S3 tokens of the turn's audio decoded with the new speaker
    condition -- written back at the turn's own samples, so words, timing and the user channel are v6's. Why not
    re-TTS from text: that changes timing and prosody as well (two changes) and costs ~1 GPU-h per 1,000 dialogues
    plus a Whisper pass and ~3 inodes per dialogue. ``annotated``: the slice's ``MoshiAnnotate`` output (the v6
    assistant audio). ``orig_voice_wav``: the v6 reference clip, for the summary's speaker check only.
    ``max_rows``: a pilot on the first rows (0 = all). Output ``out_dir`` (row-aligned with ``annotated``) +
    ``out_summary``.
    """

    def __init__(
        self,
        *,
        name: str,
        annotated: tk.Path,
        tts: tk.Path,
        pool: tk.Path,
        orig_voice_wav: tk.Path,
        venv_python_path: tk.Path,
        ffmpeg_path: tk.Path | None = None,
        seed: int = 0,
        max_rows: int = 0,
    ):
        self.name = str(name)
        self.annotated = annotated
        self.tts = tts
        self.pool = pool
        self.orig_voice_wav = orig_voice_wav
        self.venv_python_path = venv_python_path
        self.ffmpeg_path = ffmpeg_path
        self.seed = int(seed)
        self.max_rows = int(max_rows)
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 4, "gpu_mem_gb": 20}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        return job_progress_fraction(self)

    def info(self):
        f = job_progress_fraction(self)
        return None if f is None else f"{100 * f:.1f}% of rows converted"

    def run(self):
        from pathlib import Path

        from .common import add_cuda_npp_to_env, add_venv_python_lib_to_env, run_worker_script
        from .tts import InstallFFmpeg

        def env_hook(env):
            # as ChatterboxInference: the chatterbox stack imports torchaudio/torchcodec, which need our FFmpeg,
            # CUDA NPP and libpython on the library path on nodes without them
            if self.ffmpeg_path is not None:
                InstallFFmpeg.add_to_env(self.ffmpeg_path, env)
            add_cuda_npp_to_env(self.venv_python_path.get(), env)
            add_venv_python_lib_to_env(self.venv_python_path.get(), env)

        args = [
            "--annotated",
            self.annotated.get_path(),
            "--tts",
            self.tts.get_path(),
            "--pool",
            self.pool.get_path(),
            "--name",
            self.name,
            "--out",
            self.out_dir.get_path(),
            "--summary",
            self.out_summary.get_path(),
            "--orig_voice_wav",
            self.orig_voice_wav.get_path(),
            "--seed",
            self.seed,
            "--max_rows",
            self.max_rows,
            "--progress",
            os.path.abspath("progress.json"),
        ]
        run_worker_script(
            self.venv_python_path.get(),
            Path(__file__).resolve().parent / "knowledge_voice_convert.py",
            args,
            log_label="knowledge voice conversion",
            env_hook=env_hook,
        )
        if not os.path.isfile(self.out_summary.get_path()):
            raise RuntimeError(f"worker wrote no {self.out_summary.get_path()}")
