import torch


class InputNormalization(torch.nn.Module):
    """Performs mean and variance normalization of the input tensor.

    Arguments
    ---------
    mean_norm : True
         If True, the mean will be normalized.
    std_norm : True
         If True, the standard deviation will be normalized.
    norm_type : str
         It defines how the statistics are computed ('sentence' computes them
         at sentence level, 'batch' at batch level, 'speaker' at speaker
         level, while global computes a single normalization vector for all
         the sentences in the dataset). Speaker and global statistics are
         computed with a moving average approach.
    avg_factor : float
         It can be used to manually set the weighting factor between
         current statistics and accumulated ones.
    requires_grad : bool
        Whether this module should be updated using the gradient during training.
    update_until_epoch : int
        The epoch after which updates to the norm stats should stop.

    Example
    -------
    >>> import torch
    >>> norm = InputNormalization()
    >>> inputs = torch.randn([10, 101, 20])
    >>> inp_len = torch.ones([10])
    >>> features = norm(inputs, inp_len)
    """

    from typing import Dict

    spk_dict_mean: Dict[int, torch.Tensor]
    spk_dict_std: Dict[int, torch.Tensor]
    spk_dict_count: Dict[int, int]

    def __init__(
        self,
        feat_dim=80,
        mean_norm=True,
        std_norm=True,
        norm_type="global",
        avg_factor=None,
        requires_grad=False,
        update_until_epoch=3,
    ):
        super().__init__()
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.norm_type = norm_type
        self.avg_factor = avg_factor
        self.requires_grad = requires_grad
        self.register_buffer("glob_mean", torch.tensor([0.0]*feat_dim))
        self.register_buffer("glob_std", torch.tensor([0.0]*feat_dim))
        self.spk_dict_mean = {}
        self.spk_dict_std = {}
        self.spk_dict_count = {}
        self.weight = 1.0
        self.count = 0
        self.eps = 1e-10
        self.update_until_epoch = update_until_epoch

    def forward(self, x, lengths, spk_ids=torch.tensor([]), epoch=0):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : torch.Tensor
            A batch of tensors.
        lengths : torch.Tensor
            A batch of tensors containing the relative length of each
            sentence (e.g, [0.7, 0.9, 1.0]). It is used to avoid
            computing stats on zero-padded steps.
        spk_ids : torch.Tensor containing the ids of each speaker (e.g, [0 10 6]).
            It is used to perform per-speaker normalization when
            norm_type='speaker'.
        epoch : int
            The epoch count.

        Returns
        -------
        x : torch.Tensor
            The normalized tensor.
        """
        N_batches = x.shape[0]

        current_means = []
        current_stds = []

        if self.norm_type == "sentence" or self.norm_type == "speaker":
            # we will do in-place slice assignments over `out`
            out = torch.empty_like(x)
        # otherwise don't assign it yet

        for snt_id in range(N_batches):
            # Avoiding padded time steps
            actual_size = lengths[snt_id]

            # computing statistics
            current_mean, current_std = self._compute_current_stats(
                x[snt_id, 0:actual_size, ...]
            )

            current_means.append(current_mean)
            current_stds.append(current_std)

            if self.norm_type == "sentence":
                out[snt_id] = (x[snt_id] - current_mean.data) / current_std.data

        if self.norm_type == "batch" or self.norm_type == "global":
            current_mean = torch.mean(torch.stack(current_means), dim=0)
            current_std = torch.mean(torch.stack(current_stds), dim=0)

            # For multi GPU, we need to sync the statistics across processes.

            if self.norm_type == "batch":
                out = (x - current_mean.data) / (current_std.data)

            if self.norm_type == "global":
                if self.training:
                    if self.count == 0:
                        self.glob_mean = current_mean
                        self.glob_std = current_std

                    elif epoch is None or epoch < self.update_until_epoch:
                        if self.avg_factor is None:
                            self.weight = 1 / (self.count + 1)
                        else:
                            self.weight = self.avg_factor

                        self.glob_mean = (1 - self.weight) * self.glob_mean.to(
                            current_mean
                        ) + self.weight * current_mean

                        self.glob_std = (1 - self.weight) * self.glob_std.to(
                            current_std
                        ) + self.weight * current_std

                    self.glob_mean.detach()
                    self.glob_std.detach()

                    self.count = self.count + 1

                out = (x - self.glob_mean.data.to(x)) / (
                    self.glob_std.data.to(x)
                )

        return out

    def _compute_current_stats(self, x):
        """Computes mean and std

        Arguments
        ---------
        x : torch.Tensor
            A batch of tensors.

        Returns
        -------
        current_mean : torch.Tensor
            The average of x along dimension 0
        current_std : torch.Tensor
            The standard deviation of x along dimension 0
        """
        # Compute current mean
        if self.mean_norm:
            current_mean = torch.mean(x, dim=0).detach().data
        else:
            current_mean = torch.tensor([0.0], device=x.device)

        # Compute current std
        if self.std_norm:
            current_std = torch.std(x, dim=0).detach().data
        else:
            current_std = torch.tensor([1.0], device=x.device)

        # Improving numerical stability of std
        current_std = torch.max(
            current_std, self.eps * torch.ones_like(current_std)
        )

        return current_mean, current_std

    def _statistics_dict(self):
        """Fills the dictionary containing the normalization statistics."""
        state = {}
        state["count"] = self.count
        state["glob_mean"] = self.glob_mean
        state["glob_std"] = self.glob_std
        state["spk_dict_mean"] = self.spk_dict_mean
        state["spk_dict_std"] = self.spk_dict_std
        state["spk_dict_count"] = self.spk_dict_count

        return state

    def _load_statistics_dict(self, state):
        """Loads the dictionary containing the statistics.

        Arguments
        ---------
        state : dict
            A dictionary containing the normalization statistics.

        Returns
        -------
        state : dict
        """
        self.count = state["count"]
        if isinstance(state["glob_mean"], int):
            self.glob_mean = state["glob_mean"]
            self.glob_std = state["glob_std"]
        else:
            self.glob_mean = state["glob_mean"]  # .to(self.device_inp)
            self.glob_std = state["glob_std"]  # .to(self.device_inp)

        # Loading the spk_dict_mean in the right device
        self.spk_dict_mean = {}
        for spk in state["spk_dict_mean"]:
            self.spk_dict_mean[spk] = state["spk_dict_mean"][spk]  # .to(
            #    self.device_inp
            # )

        # Loading the spk_dict_std in the right device
        self.spk_dict_std = {}
        for spk in state["spk_dict_std"]:
            self.spk_dict_std[spk] = state["spk_dict_std"][spk]  # .to(
            #    self.device_inp
            # )

        self.spk_dict_count = state["spk_dict_count"]

        return state

    def to(self, device):
        """Puts the needed tensors in the right device."""
        self = super(InputNormalization, self).to(device)
        self.glob_mean = self.glob_mean.to(device)
        self.glob_std = self.glob_std.to(device)
        for spk in self.spk_dict_mean:
            self.spk_dict_mean[spk] = self.spk_dict_mean[spk].to(device)
            self.spk_dict_std[spk] = self.spk_dict_std[spk].to(device)
        return self

    def _save(self, path):
        """Save statistic dictionary.

        Arguments
        ---------
        path : str
            A path where to save the dictionary.
        """
        stats = self._statistics_dict()
        torch.save(stats, path)

    def _load(self, path, end_of_epoch=False):
        """Load statistic dictionary.

        Arguments
        ---------
        path : str
            The path of the statistic dictionary
        end_of_epoch : bool
            Whether this is the end of an epoch.
            Here for compatibility, but not used.
        """
        del end_of_epoch  # Unused here.
        device = "cpu"
        stats = torch.load(path, map_location=device)
        self._load_statistics_dict(stats)