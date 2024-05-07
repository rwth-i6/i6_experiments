import torch
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor.tensor_dict import TensorDict


class SearchCallback(ForwardCallbackIface):
    def init(self, *, model: torch.nn.Module):
        self.recognition_file = open("search_out.py", "w")
        self.recognition_file.write("{\n")

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        token_seq = outputs["tokens"].raw_tensor
        token_seq_length = outputs["tokens"].dims[0].dyn_size_ext.raw_tensor
        assert token_seq is not None
        assert token_seq_length is not None
        token_str = " ".join([token.item() for token in token_seq[:token_seq_length]])
        self.recognition_file.write(f"{repr(seq_tag)}: {repr(token_str)},\n")

    def finish(self):
        self.recognition_file.write("}\n")
        self.recognition_file.close()
