from i6_core import returnn
from ..base import AlignmentFunctor
from ... import dataclasses


class ReturnnAlignmentFunctor(AlignmentFunctor[returnn.ReturnnTrainingJob, returnn.ReturnnConfig]):
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.ReturnnTrainingJob],
        prior_config: returnn.ReturnnConfig,
        align_config: returnn.ReturnnConfig,
        align_corpus: dataclasses.NamedRasrDataInput,
        **kwargs,
    ) -> dataclasses.AlignmentData:
        raise NotImplementedError
