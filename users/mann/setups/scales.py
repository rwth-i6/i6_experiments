
class RecognitionScaleApplicator:

    def __init__(self, system):
        self.system = system
    
    def _get_amc_ref(self):
        csp = self.system.csp
        corpus = "dev"
        # copy acoustic model config to change value
        # exclusive to specified corpus
        amc = csp[corpus].acoustic_model_config
        if amc is csp["base"].acoustic_model_config:
            csp[corpus].acoustic_model_config = copy.deepcopy(amc)
        return csp[corpus].acoustic_model_config

    amc = property(_get_amc_ref)
    
    def apply(self, scorer_args, am=None, tdp=None, prior=None, **_ignored):
        if am is not None:
            scorer_args["scale"] = am
        if prior is not None:
            scorer_args["priori_scale"] = prior
        if tdp is not None:
            self.amc.tdp.scale = tdp
