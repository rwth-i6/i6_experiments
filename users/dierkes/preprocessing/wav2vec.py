import soundfile
import glob
import os
import random
import numpy as np
import typing
import torch
import sys
import better_exchook

from sisyphus import Job, Task, tk

from returnn.tf.util.basic import debug_register_better_repr, setup_tf_thread_pools, print_available_devices
from returnn.log import log

class FairseqAudioManifestCreationJob(Job):
    """
    Creates required manifest files for wav2vec pretraining with fairseq. For the
    script see https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py
    """
    def __init__(
        self, audio_dir_path, file_extension="wav", valid_percent=0.01, seed=42, path_must_contain=None
    ):
        """
        :param tk.Path audio_dir_path: path to raw audio files to be included
        :param str file_extension: file extension to look for in audio_dir_path
        :param float valid_percent: percentage of files to be in validation set
        :param int seed: random seed for splitting into train and valid set
        :param str|None path_must_contain: if set, path must contain this substring
            for a file to be included in the manifest
        """
        self.audio_dir_path = audio_dir_path
        self.file_extension = file_extension
        self.valid_percent = valid_percent
        assert 0 <= self.valid_percent <= 1.0
        self.seed = seed
        self.path_must_contain = path_must_contain

        self.out_manifest_path = self.output_path("manifest/", directory=True)
        self.rqmt = {"time": 2, "mem": 8, "cpu": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        dir_path = os.path.realpath(self.audio_dir_path.get_path())
        search_path = os.path.join(dir_path, "**/*." + self.file_extension)
        rand = random.Random(self.seed)

        valid_f = (
            open(os.path.join(self.out_manifest_path, "valid.tsv"), "w")
            if self.valid_percent > 0
            else None
        )

        with open(os.path.join(self.out_manifest_path, "train.tsv"), "w") as train_f:
            print(dir_path, file=train_f)

            if valid_f is not None:
                print(dir_path, file=valid_f)

            for fname in glob.iglob(search_path, recursive=True):
                file_path = os.path.realpath(fname)

                if self.path_must_contain and self.path_must_contain not in file_path:
                    continue

                frames = soundfile.info(fname).frames
                dest = train_f if rand.random() > self.valid_percent else valid_f
                print(
                    "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), 
                        file=dest
                )
        if valid_f is not None:
            valid_f.close()


class FairseqWav2VecModelConvertAndDumpJob(Job):
    """
    Conversion Job for fairseq model checkpoints to RETURNN model dictionary and checkpoint
    """
    def __init__(
            self,
            wav2vec_config,
            output_name,
            audio_input,
            device="cpu",
            validate_atol=None,
            fairseq_root=None,
            pytorch_to_returnn_root=None,
    ):
        """
        :param tk.Path wav2vec_config: path to wav2vec config to be converted
        :param str output_name: name of resulting RETURNN model dictionary and checkpoint
        :param tk.Path audio_input: path to exemplary audio file for forwarding through the model
        :param str device: cpu or gpu
        :param float|None validate_atol: allowed absolute difference between fairseq model outputs
            and converted RETURNN outputs.
        :param tk.Path|None fairseq_root: path to fairseq version to use for conversion
        :param tk.Path|None pytorch_to_returnn_root: path to pytorch_to_returnn version to use for conversion
        """
        self.wav2vec_config = wav2vec_config
        self.output_name = output_name
        self.audio_input = audio_input
        self.device = device
        assert self.device in ["cpu", "gpu"]
        self.validate_atol = validate_atol
        self.fairseq_root = fairseq_root
        self.pytorch_to_returnn_root = pytorch_to_returnn_root

        if not self.fairseq_root:
            print("WARNING: no explicit fairseq root directory given")
        if not self.pytorch_to_returnn_root:
            print("WARNING: no explicit pytorch_to_returnn root directory given")

        self.returnn_model_out = self.output_path(self.output_name)
        self.rqmt = {"time": 2, "mem": 32, "cpu": 1, "gpu": 1 if self.device=="gpu" else 0,}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        better_exchook.install()
        debug_register_better_repr()
        log.initialize(verbosity=[5])
        log.Verbosity = 10
        setup_tf_thread_pools()
        print_available_devices()

        if self.fairseq_root:
            sys.path.append(self.fairseq_root.get_path())
        if self.pytorch_to_returnn_root:
            sys.path.append(self.pytorch_to_returnn_root.get_path())

        def model_func(wrapped_import, inputs: torch.Tensor):
            if typing.TYPE_CHECKING or not wrapped_import:
                import torch
                import fairseq
                from fairseq.models import wav2vec as wav2vec_models
            else:
                torch = wrapped_import("torch")
                torch_nn = wrapped_import("torch.nn.modules")
                fairseq = wrapped_import("fairseq")
                wav2vec_models = wrapped_import("fairseq.models.wav2vec")
                # the following imports are necessary to make sure that these parts are registered also for the wrapped runs
                audio_pretraining = wrapped_import("fairseq.tasks.audio_pretraining")
                adam = wrapped_import("fairseq.optim.adam")
                polynomial_decay = wrapped_import("fairseq.optim.lr_scheduler.polynomial_decay_schedule")
                wav2vec_criterion = wrapped_import("fairseq.criterions.wav2vec_criterion")

            # Initialize PyTorch example
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.wav2vec_config.get_path()])
            w2v2_model = model[0]
            print("WARNING: Simplifications: feature_grad_mult=1.0, encoder layer dropout=0.0")
            w2v2_model.feature_grad_mult = 1.0
            w2v2_model.encoder.layerdrop = 0.0
            print("WARNING: remove quantizer, this involves using non-optimized parameters")
            w2v2_model.quantizer = None
            w2v2_model.project_q.in_features = w2v2_model.embed
            w2v2_model.project_q._parameters["weight"] = torch.from_numpy(np.random.randn(256, 512).astype("float32"))

            print("WARNING: simplify model by setting mask=False")
            return w2v2_model(inputs, mask=False, features_only=True)["x"]

        print("Load audio")
        print("  ", self.audio_input.get_path())
        audio, fr = soundfile.read(self.audio_input.get_path())
        audio = np.reshape(audio, [4, len(audio) // 4]).astype('float32')  # dim (B, T) -> (4, *)
        audio_pt = torch.from_numpy(audio)
        print(f"  input shape: {audio.shape}")

        import pytorch_to_returnn.log
        pytorch_to_returnn.log.Verbosity = 6
        from pytorch_to_returnn.converter import verify_torch_and_convert_to_returnn
        converter = verify_torch_and_convert_to_returnn(
            model_func, inputs=audio,
            inputs_data_kwargs={"shape": (None,), "batch_dim_axis": 0, "time_dim_axis": 1, "feature_dim_axis": None},
            returnn_dummy_input_shape=audio.shape,
            export_tf_checkpoint_save_path=self.returnn_model_out.get_path(),
            validate_allclose_kwargs={"atol": self.validate_atol} if self.validate_atol else None,
            )
        with open(self.returnn_model_out.get_path() + ".network.dict", "wt", encoding="utf-8") as f:
            f.write(converter.get_returnn_config_serialized())
        os.rename(self.returnn_model_out.get_path() + ".network.dict", self.returnn_model_out.get_path() + ".network.dict.py" )

        features = model_func(None, inputs=audio_pt)
        print(f"  output shape PyTorch: {features.shape}")
