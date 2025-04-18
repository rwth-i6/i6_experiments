from ...streamable_module import StreamableModule
from ..components.base_parts import BaseFeatureExtractor

class FeatureExtractionBuilder:
    
    @staticmethod
    def build(config) -> BaseFeatureExtractor:
        pass
        # fe_class = Registry.get_fe_class(config.type)  # e.g., "feature_extraction_streamable"
        # return fe_class(**config.params)  # Inject hyperparams