import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.amp import autocast

from returnn.config import Config
from returnn.datasets.basic import Dataset
from returnn.log import log
from returnn.torch.engine import Engine
from returnn.torch.updater import Updater
from returnn.torch.context import get_run_ctx, init_load_run_ctx, init_train_step_run_ctx, init_forward_step_run_ctx, RunCtx, Loss


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl[:,:,:gl.size(2)] - gl))

    return loss*2


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

class CustomUpdater(Updater):
    """
    Wraps a torch.optim.Optimizer, and extends it by some further functionality.
    """

    def __init__(self, *, config: Dict[str, Any], network, device, initial_learning_rate=1.0):
        """
        :param config defining the training conditions.
        :param torch.nn.Module network: PyTorch Module defining the network.
        :param float initial_learning_rate:
        """
        self.config = config
        self.learning_rate = initial_learning_rate
        self.network = network
        self._device = device
        self.optimizer = None  # type: typing.Optional[torch.optim.Optimizer]

        self._grad_scaler = None  # type: amp.GradScaler

        self._grad_clip = self.config.get("gradient_clip", None)
        self._grad_clip_norm = self.config.get("gradient_clip_norm", None)

        self._accum_grad_multiple_step = config.get("accum_grad_multiple_step", 1)

    def create_optimizer(self):
        """
        Creates an optimizer and stores it in self.optimizer.
        """
        optimizer_opts = self.config.get("optimizer", None)
        if optimizer_opts is None:
            raise ValueError("config field 'optimizer' needs to be set explicitely for the Torch backend")
        self.optimizer = self._create_optimizer(optimizer_opts)
        
    def _get_optimizer_param_groups(self, optim_class, optimizer_opts):
        """
        TODO: as the original code to exclude some layers from the weight_decay was incorrect,
              the current behavior is to not exclude any parameter at all

        The following is the old docstring. It defines the desired behaviour, but not the actual one,
        which is doing nothing.
        ----------------------------------------------------------------------------------------------------
        The weight_decay parameter from AdamW affects the weights of layers such as LayerNorm and Embedding.
        This function creates a blacklist of network modules and splits the optimizer groups in two:
        those who will receive weight decay, and those who won't receive it.
        The weight_decay parameter of the rest of the optimizers is L2 regularization.

        For further reading, see https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025 and
        https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994.

        This code is based on https://github.com/karpathy/minGPT (MIT license):
        https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136.

        :param type[torch.optim.Optimizer] optim_class: Optimizer class.
        :param dict[str] optimizer_opts: Optimizer configuration specified by the user.
        :return: List of configurations for the different sets of parameters.
        :rtype: List[Dict[str]]
        """
        network_params = self.network.parameters()
        return [{"params": network_params}]



class CustomEngine(Engine):
    """
        Written for MiniRETURNN version 0.3+
    """

    def __init__(self, config: Config):
        """
        :param config:
        """
        super().__init__(config=config)
        self._updater: Dict[Any, Updater] = {}

    def init_train(
        self,
        train_data: Optional[Dataset] = None,
        dev_data: Optional[Dataset] = None,
    ):
        """
        :param train_data: Used when initializing from existing Datasets
        :param dev_data: Used when initializing from existing Datasets
        """
        super(Engine, self).init_train(train_data=train_data, dev_data=dev_data)

        self._train_dataloader = self._create_data_loader(self.train_dataset)
        for dataset_name, dataset in self.eval_datasets.items():
            self._eval_dataloaders[dataset_name] = self._create_data_loader(dataset)

        self._start_epoch, filename = self.get_epoch_model(self.config)

        if self._start_epoch is not None:
            self._start_epoch += 1
        else:
            self._start_epoch = 1

        self._final_epoch = self.config_get_final_epoch(self.config)

        self._load_model(epoch=self._start_epoch, filename=filename)

        self._save_model_epoch_interval = self.config.int("save_interval", 1)


        updater_settings = self.config.typed_value("updater")
        assert isinstance(updater_settings, dict)
        assert "disc" in updater_settings and "gen" in updater_settings, "CustomEngine requires disc and gen updater"

        for key, updater_config in updater_settings.items():
            updater = CustomUpdater(
                config=updater_config,
                network=getattr(self._model, updater_config["submodel"]),
                device=self._device,
                initial_learning_rate=self.learning_rate
            )
            updater.create_optimizer()
            self._updater[key] = updater


        if self._start_epoch > 1:
            self._load_optimizer(self._start_epoch)

        self._train_step_func = self.config.typed_value("train_step")
        assert self._train_step_func, "train_step not defined"

        amp_options = self.config.typed_value("torch_amp_options")
        if amp_options is not None:
            assert isinstance(amp_options, dict)
            amp_dtype_str = amp_options.get("dtype")
            assert amp_dtype_str in ["float16", "bfloat16"]
            self._amp_dtype = getattr(torch, amp_dtype_str)
            for updater in self._updater.values():
                updater.create_grad_scaler()
                
    def init_train_epoch(self):
        """
        init train (sub)epoch. LR etc
        """
        self.learning_rate = self.learning_rate_control.get_learning_rate_for_epoch(self.epoch)

        # Update learning rate
        for updater in self._updater.values():
            updater.set_learning_rate(self.learning_rate)


    def _load_optimizer(self, epoch):
        """
        Loads a torch.optim.Optimizer from disk and uses it as the optimizer.
        This function is a wrapper to Updater.load_optimizer().

        :param int epoch: Epoch from which to load the optimizer state.
        """
        for key, updater in self._updater.items():
            filename = self.get_epoch_model_filename(epoch=epoch - 1) + f".opt_{key}.pt"

            if os.path.isfile(filename):
                updater.load_optimizer(filename)
            elif self.config.bool("allow_missing_optimizer_checkpoint", False):
                print(
                    "Warning: No optimizer state for the given checkpoint could be loaded. Continuing training with a fresh optimizer...",
                    file=log.v4,
                )
            else:
                raise Exception(f"Optimizer file {filename} not found and allow_missing_optimizer_checkpoint is False")


    def _save_optimizer(self):
        """
        Saves the optimizer state to a file.
        This function is a wrapper to Updater.save_optimizer().
        """
        for key, updater in self._updater.items():
            filename = self.get_epoch_model_filename() + f".opt_{key}.pt"
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            updater.save_optimizer(filename)

            # keep only the last two optimizer states (two in case one file gets corrupted)
            clean_epoch = self.epoch - 2
            if clean_epoch > 0:
                filename = self.get_epoch_model_filename(epoch=clean_epoch) + f".opt_{key}.pt"
                if os.path.isfile(filename):
                    os.unlink(filename)
                    
    @staticmethod
    def save_wav(wav, path, sr, peak_normalization=True):
        from scipy.io import wavfile
        if peak_normalization:
            wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        else:
            wav *= 32767
        wavfile.write(path, sr, wav.astype(np.int16))

    def run_train_step(
        self, data: dict[str, torch.Tensor], run_ctx: RunCtx, step_idx: int
    ) -> Tuple[Tensor, Dict[str, Loss]]:
        """
        :param data: model inputs for the step
        :param run_ctx: the current run ctx object
        :param step_idx: the current step index passed to the updater
        :return: total loss (weighted sum) calculated for the step, and individual losses as a name -> value mapping
        """
        assert isinstance(data, dict) and data
        data = self.move_data_to_device(data)

        tags = data["seq_tag"]
        samples = data["raw_audio"]  # [B, T', 1]
        samples_len = data["raw_audio:size1"]  # [B]

        y = samples.transpose(1, 2)
        mel, _ = self._model.feature_extraction(torch.squeeze(samples), samples_len)  # [B, T, F]

        # Noise
        z = torch.randn(samples.size(0), self._model.config.generator_config.cond_in_channels, mel.size(1)).to(samples.device)


        y_g_hat = self._model.generator(z, mel.transpose(1, 2))  # Generator needs mel as B,F,T

        #y_g_hat_mel = self._model.feature_extraction(y_g_hat, samples_len)
        #y_g_hat_mel = torch.swapaxes(y_g_hat_mel, 1, 2)

        if run_ctx.epoch >= self._model.config.start_discriminator_epoch:
            # MPD
            loss_disc_all = self._model.discriminator(y, y_g_hat.detach())
            if step_idx != -1:
                self._updater["disc"].step(loss_disc_all, step_idx)
                # free some memory
                self._updater["disc"].get_optimizer().zero_grad()

            run_ctx.mark_as_loss(name="disc_all", loss=loss_disc_all)

        sc_loss, mag_loss = self._model.stft_loss(y_g_hat[:, :, :y.size(2)].squeeze(1), y.squeeze(1))

        loss_mel = self._model.config.mel_loss_scale * (sc_loss + mag_loss)  # STFT Loss

        if run_ctx.epoch >= self._model.config.start_discriminator_epoch:
            # cut y_g_hat
            y_g_hat = y_g_hat[:, :, :y.size(2)]
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self._model.discriminator.mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self._model.discriminator.msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen = loss_gen_f + loss_gen_s
            loss_fm = loss_fm_f + loss_fm_s
            loss_gen_all = loss_gen + loss_fm + loss_mel
            run_ctx.mark_as_loss(name="gen", loss=loss_gen)
            run_ctx.mark_as_loss(name="fm", loss=loss_fm)
        else:
            loss_gen_all = loss_mel
        run_ctx.mark_as_loss(name="mel", loss=loss_mel)

        if step_idx != -1:
            self._updater["gen"].step(loss_gen_all, step_idx)
            # free some memory
            self._updater["gen"].get_optimizer().zero_grad()

        run_ctx.mark_as_loss(name="gen_all", loss=loss_gen_all)
        losses_dict = run_ctx.losses
        total_loss = run_ctx.total_loss()

        if step_idx == -1 and run_ctx.epoch % 5 == 0:
            epoch = run_ctx.epoch
            audio_numpy = y_g_hat.detach().cpu().numpy()
            basename = self.get_epoch_model_filename()
            directory = os.path.join(os.path.dirname(basename), "audio_ep_%i" % epoch)
            if not os.path.exists(directory):
                os.mkdir(directory)
            for tag, audio in zip(tags, audio_numpy):
                filename = os.path.join(directory, tag.replace("/", "_") + ".wav")
                if not os.path.exists(filename):
                    self.save_wav(audio[0], filename, sr=16000)

        return total_loss, losses_dict
    
    def run_eval_step(self, data: dict[str, torch.Tensor], run_ctx: RunCtx) -> Tuple[Tensor, Dict[str, Loss]]:
        return self.run_train_step(data=data, run_ctx=run_ctx, step_idx=-1)
