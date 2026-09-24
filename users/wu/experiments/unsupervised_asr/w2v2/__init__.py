"""New in the port (package marker, no source counterpart): the frozen wav2vec2-large-lv60 front end of phase 4a.

The layer-15 feature dump as a RETURNN forward (:mod:`.features` builds the job, :mod:`.forward` is
the RETURNN-side model, forward step and callback) and the 500-unit k-means stream quantized from it
(:mod:`.units`).
"""
