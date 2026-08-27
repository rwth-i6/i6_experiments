"""
Shared helper for the no_outq recognition variants in this folder.

In the *_mem_inited modules the *input* quantizers are already nn.Identity (prep_quant() bakes
their scales into the memristor DAC during the conversion job), while the activation *output*
quantizers (*_out_quant) survive. Removing those as well improves the hardware WER
(v11 w4/a8 512dim, dev-other, lm 0.8 / prior 0.5: 6.48-6.65 vs 6.80-6.91 with out-quants).

This replaces them generically instead of copying the ~29 KB module files per model version --
the naming convention is identical from memristor_v7 up to v17:
    net file:     lin_1 / lin_2 / dconv_1 / pconv_1 / pconv_2 / lin_out _out_quant
    modules file: in_proj / out_proj / learn_emb _out_quant
q_quantizer / k_quantizer (dot-product quantizers) and the weight quantizers are deliberately
left alone.
"""

import torch


def strip_output_quantizers(model: torch.nn.Module) -> int:
    """
    Replace every *_out_quant submodule of `model` by nn.Identity, in place.

    Two passes on purpose: `lin_out_out_quant` is additionally held by reference inside
    `self.final_linear = ModuleList([Sequential(lin_out_in_quant, lin_out, lin_out_out_quant)])`
    (only built when quantize_output is set, which no current config does). Replacing by
    attribute name alone would leave that alias in place, so the second pass matches on object
    identity and therefore also rewrites the Sequential entry.

    :return: number of replaced modules
    """
    targets = {
        id(child)
        for module in model.modules()
        for name, child in module.named_children()
        if name.endswith("_out_quant")
    }
    num_replaced = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if id(child) in targets:
                setattr(module, name, torch.nn.Identity())
                num_replaced += 1
    assert num_replaced > 0, "no *_out_quant modules found -- did the naming convention change?"
    return num_replaced
