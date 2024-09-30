from i6_experiments.users.berger.recipe import returnn
from ..base import AlignmentFunctor
from ... import dataclasses


class OptunaReturnnAlignmentFunctor(AlignmentFunctor[returnn.OptunaReturnnTrainingJob, returnn.OptunaReturnnConfig]):
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[returnn.OptunaReturnnTrainingJob],
        prior_config: returnn.OptunaReturnnConfig,
        align_config: returnn.OptunaReturnnConfig,
        align_corpus: dataclasses.NamedRasrDataInput,
        **kwargs,
    ) -> dataclasses.AlignmentData:
        raise NotImplementedError
