"""
Continue a finished RETURNN training **with its optimizer state**, using RETURNN's own resume path.

🔴 **Why this exists, and why `import_model_train_epoch1` is not it.**

`import_model_train_epoch1` is the "start a NEW training from someone else's weights" path. It is
weights-only by design (`torch/engine.py:1207-1211` reads only the ``"model"`` key) and it forces the
epoch counter to 1 (`engine/base.py:191-192` yields ``(0, file)``, `:234-235` turns that into
``start_epoch = 1``). Since `torch/engine.py:258` only loads the optimizer ``if self._start_epoch > 1``,
an imported run **always** begins with fresh optimizer moments.

RETURNN's answer to "keep the optimizer" is not an import option -- it is **resumption**, and the
mechanism is defined by the optimizer file itself: `engine/base.py:101-107` refuses any checkpoint in
the model dir that has no ``.opt.pt`` sibling, precisely so that a resumable model is one whose
optimizer survived. So this job adds no feature to RETURNN; it *arranges the preconditions* RETURNN
already documents, by seeding the new job's model dir with a donor checkpoint pair.

**Three things this buys at once**, which is why it is worth a job rather than a flag:

1. **Optimizer moments carry over.** The +TTS finetune degraded below its own starting point for
   ~5 epochs before recovering (dev CE 0.1819 -> 0.1864 at epoch 5, and the same shape on *devtrain*,
   so it is not overfitting). Whether that dip is caused by cold moments or by the LR ramp is not
   settled -- see (3) -- but a resumed run removes one of the two candidates.
2. **The LM text partition advances.** RETURNN picks the text partition as
   ``(epoch - 1) % partition_epoch`` (`datasets/lm.py:706`). A run that restarts at epoch 1 with
   ``partition_epoch=75`` re-reads partitions 0-9 -- which is exactly what the finetune did, having
   inherited them from the winner's own epochs 1-10. Resuming at epoch N+1 advances to fresh text.
3. **The LR schedule is ours to choose** over the continued range, so a flat low LR is expressible.
   That is the experiment separating "the dip was avoidable damage" from "the dip was the price of
   learning the new data": if GlowTTS-audio WER still improves without a dip, the ramp was waste.

⚠ **Why the donor travels in the CONFIG and not as constructor arguments.** The obvious design is
``ReturnnTrainingResumedJob(..., resume_from_checkpoint=...)``, and it is a trap twice over.
`ReturnnTrainingJob.hash` builds its dict from only ``returnn_config`` / ``returnn_python_exe`` /
``returnn_root`` (+ horovod / multi_node_slots) (`i6_core/returnn/training.py:548-561`), so a new
constructor argument **never reaches the hash** -- two donors would collide on one job and the second
would silently serve the first's result. Overriding ``hash`` to fix that then fails on contact:
Sisyphus binds ``parsed_args`` against *this* class's ``__init__`` (`job.py`, ``get_args``), so a
``**kwargs`` catch-all hides the parent's 14 named parameters and
``ReturnnTrainingJob.create_returnn_config(**inner)`` raises ``missing 6 required positional
arguments``. Restating the parent's whole signature to dodge that is exactly the kind of copy that
rots when upstream adds a parameter.

Putting the donor in ``returnn_config`` sidesteps all of it: the parent hashes the config it already
hashes, the donor identity is therefore *in* the hash for free, the written ``returnn.config``
documents what this run continues, and ``rqmt`` stays unhashed as it should.
"""

from __future__ import annotations

import os
import shutil

from sisyphus import tk

from i6_core.returnn.training import ReturnnTrainingJob


# Config key carrying the donor. Underscore-prefixed because RETURNN itself never reads it; it exists
# so the value lands in the hashed config (and in the written returnn.config, where a reader can see
# what this run resumes).
RESUME_CONFIG_KEY = "_resume_donor"


class ReturnnTrainingResumedJob(ReturnnTrainingJob):
    """
    :class:`ReturnnTrainingJob` that RESUMES a donor training instead of importing its weights.

    Takes **no** extra constructor arguments (see the module docstring for why). The donor is read
    from ``returnn_config.config[RESUME_CONFIG_KEY]``, a dict of::

        {"checkpoint": tk.Path, "optimizer_state": tk.Path, "learning_rates": tk.Path, "epoch": int}

    Use :class:`PatchTrainingJobToResumed` rather than constructing this directly.
    """

    def _resume_donor(self) -> dict:
        donor = self.returnn_config.get(RESUME_CONFIG_KEY, None)
        assert donor, (
            f"ReturnnTrainingResumedJob: config has no {RESUME_CONFIG_KEY!r}; this job class only"
            " makes sense with a donor to resume from"
        )
        return donor

    def create_files(self):
        """Write the config as usual, then seed the model dir + learning-rate file for the resume."""
        super().create_files()

        donor = self._resume_donor()
        epoch = int(donor["epoch"])
        cfg = self.returnn_config

        # Guards for the two ways this silently degrades into "an ordinary run", both of which
        # produce a healthy-looking training that simply is not what was asked for.
        assert not cfg.get("import_model_train_epoch1", None), (
            "`import_model_train_epoch1` is set, which takes RETURNN's weights-only import branch"
            " (engine/base.py:191-192) and forces start_epoch=1, so the optimizer would NOT be"
            " loaded (torch/engine.py:258) -- i.e. exactly the non-resumed arm, under a new name."
        )
        num_epochs = cfg.get("num_epochs", None)
        assert num_epochs is not None and int(num_epochs) > epoch, (
            f"num_epochs={num_epochs!r} must exceed the resume epoch {epoch}: RETURNN bounds its"
            " model scan by the final epoch (engine/base.py:69-72, :91), so a lower value never"
            " finds the seeded checkpoint and training starts from scratch at epoch 1."
        )

        models_dir = self.out_model_dir.get_path()
        stem = f"epoch.{epoch:03d}"
        for key, suffix in (("checkpoint", ".pt"), ("optimizer_state", ".opt.pt")):
            dst = os.path.join(models_dir, stem + suffix)
            if os.path.lexists(dst):
                continue
            src_path = os.path.abspath(tk.uncached_path(donor[key]))
            # ⚠ `os.symlink` creates a DANGLING link without complaining, and a dangling checkpoint
            # reads as "no resumable model" -> silent restart at epoch 1. Assert the target resolves,
            # the same lesson as the ResolveOverlayCheckpoint dangling-link bug.
            assert os.path.exists(src_path), f"resume source does not exist: {src_path}"
            os.symlink(src_path, dst)
            assert os.path.exists(dst), f"seeded link dangles: {dst} -> {src_path}"

        # `learning_rate_file` is RELATIVE, so it resolves against the task cwd -- this job's `work/`
        # dir for every task (verified: create_files' own `rnn.sh` lands there beside the run logs).
        # Needed because `_check_missing_eval` (torch/engine.py:1733-1752) refuses to start when a
        # learning_rate_file is configured and the epochs before the resume point have no scores.
        lrf = cfg.get("learning_rate_file", "learning_rates")
        if not os.path.exists(lrf):
            # Copied, never symlinked: RETURNN APPENDS to this file as training proceeds, and a
            # symlink would write the new epochs back into the donor's finished output.
            shutil.copyfile(tk.uncached_path(donor["learning_rates"]), lrf)


    def _get_run_cmd(self):
        """Drop the ``start_epoch`` guard once this job has written its OWN checkpoints.

        🔴 Hit 2026-09-21: both resumed arms died on their first resubmission with ``KeyError: 38``.
        ``start_epoch = donor + 1`` sits in the (hashed) config, so on EVERY run RETURNN indexes
        ``existing_models[start_epoch - 1]`` (`engine/base.py:211`) -- but ``cleanup_old_models``
        (keep_last_n) had already deleted the seeded ``epoch.038`` link. Re-seeding the link would be
        worse: RETURNN would then restart at the donor and overwrite epochs 39+.
        The guard only exists for the FIRST run (a seeding that silently failed must not restart at
        epoch 1). Once an epoch past the donor exists, ``++start_epoch auto`` hands resumption back to
        RETURNN's normal scan (newest checkpoint with an ``.opt.pt``). A command-line override, so
        neither the hashed config nor the written returnn.config changes.
        """
        cmd = super()._get_run_cmd()
        donor_epoch = int(self._resume_donor()["epoch"])
        models_dir = self.out_model_dir.get_path()
        own = [
            fn
            for fn in (os.listdir(models_dir) if os.path.isdir(models_dir) else [])
            if fn.startswith("epoch.") and fn.endswith(".opt.pt") and int(fn.split(".")[1]) > donor_epoch
        ]
        if own:
            print(f"ReturnnTrainingResumedJob: own checkpoints {sorted(own)} exist -> ++start_epoch auto")
            # ⚠ Quoted: RETURNN eval()s a `++` value against the typed original (an int here), so a bare
            # `auto` is a NameError (hit 2026-09-21). Verified with EngineBase.get_train_start_epoch on
            # both real model dirs: resumes at 49 (ablation, done) and 46 (+TTS).
            cmd = cmd + ["++start_epoch", "'auto'"]
        return cmd


class PatchTrainingJobToResumed:
    """
    Context manager: while active, ``train_v4`` builds a :class:`ReturnnTrainingResumedJob`.

    **Why a patch rather than a parameter.** `train_v4.py:235` does
    ``returnn_train_job = ReturnnTrainingJob(returnn_train_config, **kwargs)`` against the name it
    imported as a module global at `:18`, and exposes no hook to substitute the class. Rebinding that
    module attribute is the supported window and is the same mechanism (and the same reason it works)
    as ``PatchGlowTtsToGerman`` / ``_PatchAsrBranchWithTts`` elsewhere in this setup.

    Everything else -- the whole `_train_tts_encoder` call, the dataset, the model def, the recogs,
    ``ModelWithCheckpoints.from_training_job`` -- stays byte-identical to the arm being continued.
    Only the job class changes, and the subclass IS a ``ReturnnTrainingJob``, so every downstream
    consumer keeps working unchanged.

    It also injects ``start_epoch = epoch + 1``. That is a real RETURNN option
    (`engine/base.py:126`, `:140`) and it makes the resume **fail loudly** rather than quietly:
    with it set, `engine/base.py:203-211` indexes ``existing_models[start_epoch - 1]`` directly, so a
    seeding that did not happen raises instead of restarting at epoch 1.

    ⚠ Asserts **exactly one** substitution on exit. A silent zero would build the ordinary job, which
    trains perfectly well and is precisely the non-resumed arm we already have.
    """

    def __init__(self, *, checkpoint: tk.Path, optimizer_state: tk.Path, learning_rates: tk.Path, epoch: int):
        self.donor = {
            "checkpoint": checkpoint,
            "optimizer_state": optimizer_state,
            "learning_rates": learning_rates,
            "epoch": int(epoch),
        }
        self.count = 0
        self._patch = None

    def __enter__(self):
        import unittest.mock

        from i6_experiments.users.zeyer import train_v4 as _t4

        def _wrapped(returnn_config, **kwargs):
            self.count += 1
            # Mutate a copy: the same config object must not leak the donor into a sibling arm.
            import copy as _copy

            cfg = _copy.deepcopy(returnn_config)
            cfg.config[RESUME_CONFIG_KEY] = self.donor
            cfg.config["start_epoch"] = self.donor["epoch"] + 1
            return ReturnnTrainingResumedJob(cfg, **kwargs)

        assert _t4.ReturnnTrainingJob is ReturnnTrainingJob, "already patched (nested?)"
        self._patch = unittest.mock.patch.object(_t4, "ReturnnTrainingJob", _wrapped)
        self._patch.start()

        # 🔴 The per-epoch recog would ask for epochs the resumed job never writes. ``fixed_epochs``
        # comes from ``default_returnn_keep_epochs(num_epochs)`` (5, 10, 20, ...), but this job starts
        # at donor+1, so ``epoch.005.pt`` etc. never exist and that recog fails its input check
        # forever ("Job isn't runnable", hit 2026-09-21 on BatchedReturnnForwardJob.LSsqPhS3J6fK).
        # Drop the impossible epochs from the ModelWithCheckpoints only -- the training job and its
        # hash are untouched, and the last fixed epoch (what every headline recog uses) is unchanged.
        import dataclasses as _dc

        from i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints import ModelWithCheckpoints

        orig_from_job = ModelWithCheckpoints.from_training_job
        donor_epoch = self.donor["epoch"]

        def _from_job(definition, training_job, **kw):
            res = orig_from_job(definition=definition, training_job=training_job, **kw)
            if isinstance(training_job, ReturnnTrainingResumedJob):
                kept = {e for e in res.fixed_epochs if e > donor_epoch}
                assert kept, f"no fixed epoch after the donor epoch {donor_epoch}: {sorted(res.fixed_epochs)}"
                res = _dc.replace(res, fixed_epochs=kept)
            return res

        self._patch_mwc = unittest.mock.patch.object(ModelWithCheckpoints, "from_training_job", staticmethod(_from_job))
        self._patch_mwc.start()
        return self

    def __exit__(self, *exc):
        self._patch_mwc.stop()
        self._patch.stop()
        if exc[0] is None:
            assert self.count == 1, (
                f"expected to build exactly 1 resumed training job, built {self.count} --"
                " the arm would be an ordinary cold-optimizer run under a resumed name"
            )
        return False


def winner_resume_inputs(winner_model, *, epoch: int) -> dict:
    """
    The donor inputs for :class:`PatchTrainingJobToResumed`, taken from a ``ModelWithCheckpoints``.

    Derived from the donor's own training job rather than hardcoded, so a re-import cannot silently
    point at a different run -- the same reasoning as ``winner_plus_tts.winner_checkpoint``.

    ⚠ The ``.opt.pt`` sibling and ``learning_rates`` are plain paths under the donor's output dir.
    Safe **only because the donor is finished** (asserted): Sisyphus builds no dependency edge on a
    creator-less path, so a still-running donor would be read while incomplete.
    """
    ckpt = winner_model.get_epoch(epoch).checkpoint
    job = ckpt.path.creator
    assert job is not None, "donor checkpoint has no creator job; cannot locate its optimizer state"
    models_dir = job.out_model_dir.get_path()
    _opt_path = os.path.join(models_dir, f"epoch.{epoch:03d}.opt.pt")
    # `hash_overwrite` = the path's own string: hash-neutral (`job_path.py:129-136` builds
    # `(None, path)` either way) but it FREEZES the identity, so relocating the setup does not
    # re-hash this job. Without it sisyphus warns, correctly, that the hash is a location.
    opt = tk.Path(_opt_path, hash_overwrite=_opt_path)
    assert os.path.exists(opt.get_path()), (
        f"donor has no optimizer state at {opt.get_path()} -- RETURNN only treats a checkpoint as"
        " resumable when the .opt.pt sibling exists (engine/base.py:101-107)."
        " Note RETURNN keeps only the last two .opt.pt files (torch/engine.py:1518-1524),"
        " so an early donor epoch will not have one."
    )
    return {
        "checkpoint": ckpt.path,
        "optimizer_state": opt,
        "learning_rates": job.out_learning_rates,
        "epoch": epoch,
    }
