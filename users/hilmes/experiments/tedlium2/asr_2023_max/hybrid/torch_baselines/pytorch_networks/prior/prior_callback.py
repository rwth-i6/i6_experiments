import numpy as np
import torch
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import TensorDict


class ComputePriorCallback(ForwardCallbackIface):
    def init(self, *, model: torch.nn.Module):
        self.n = 1
        self.avg_probs = None

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        log_prob_tensor = outputs["log_probs"].raw_tensor
        assert log_prob_tensor is not None
        if isinstance(log_prob_tensor, torch.Tensor):
            prob_tensor_iter = iter(torch.exp(log_prob_tensor))
        else:
            prob_tensor_iter = iter(np.exp(log_prob_tensor))

        if self.avg_probs is None:
            self.avg_probs = next(prob_tensor_iter)

        for prob_tensor in prob_tensor_iter:
            self.n += 1
            self.avg_probs += (prob_tensor - self.avg_probs) / self.n

    def finish(self):
        if isinstance(self.avg_probs, torch.Tensor):
            prob_array = self.avg_probs.cpu().numpy()
        else:
            prob_array = self.avg_probs
        log_prob_array = np.log(prob_array)
        log_prob_strings = ["%.20e" % s for s in log_prob_array]

        print("Sum of probs, should be close to 1.0:", np.sum(prob_array))
        # Write txt file
        with open("prior.txt", "wt") as f:
            f.write(" ".join(log_prob_strings))

        # Write xml file
        with open("prior.xml", "wt") as f:
            f.write(f'<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="{len(log_prob_array)}">\n')
            f.write(" ".join(log_prob_strings))
            f.write("\n</vector-f32>")

        # Plot png file
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xdata = range(len(prob_array))
        plt.semilogy(xdata, prob_array)
        plt.xlabel("emission idx")
        plt.ylabel("prior")
        plt.grid(True)
        plt.savefig("prior.png")
