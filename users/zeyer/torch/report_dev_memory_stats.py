import torch


def report_dev_memory_stats(device: torch.device):
    from returnn.util import basic as util

    if device.type == "cuda":
        stats = [
            f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(device))}",
            f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(device))}",
            f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(device))}",
            f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(device))}",
        ]
        print(f"Memory usage ({device!s}):", " ".join(stats))
