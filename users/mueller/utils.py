import string
import copy

from i6_core.returnn import ReturnnConfig as RF
from i6_core.util import instanciate_delayed
from i6_experiments.common.setups.serialization import PartialImport as PI
from i6_experiments.users.mann.nn.util import DelayedCodeWrapper
from sisyphus.hash import sis_hash_helper


class PartialImportCustom(PI):
    def get(self) -> str:
        arguments = {**self.hashed_arguments}
        arguments.update(self.unhashed_arguments)
        print(arguments)
        return string.Template(self.TEMPLATE).substitute(
            {
                "KWARGS": str(instanciate_delayed(arguments)),
                "IMPORT_PATH": self.module,
                "IMPORT_NAME": self.object_name,
                "OBJECT_NAME": self.import_as if self.import_as is not None else self.object_name,
            }
        )
        
class ReturnnConfigCustom(RF):
    def __init__(
        self,
        config,
        post_config=None,
        staged_network_dict=None,
        *,
        python_prolog=None,
        python_prolog_hash=None,
        python_epilog="",
        python_epilog_hash=None,
        hash_full_python_code=False,
        sort_config=True,
        pprint_kwargs=None,
        black_formatting=True,
    ):
        if python_prolog_hash is None and python_prolog is not None:
            python_prolog_hash = []
            
        super().__init__(
            config=config,
            post_config=post_config,
            staged_network_dict=staged_network_dict,
            python_prolog=python_prolog,
            python_prolog_hash=python_prolog_hash,
            python_epilog=python_epilog,
            python_epilog_hash=python_epilog_hash,
            hash_full_python_code=hash_full_python_code,
            sort_config=sort_config,
            pprint_kwargs=pprint_kwargs,
            black_formatting=black_formatting,
        )
        
        if self.python_prolog_hash == []:
            self.python_prolog_hash = None
    
    def _sis_hash(self):
        conf = copy.deepcopy(self.config)
        if "preload_from_files" in conf:
            for v in conf["preload_from_files"].values():
                if "filename" in v and isinstance(v["filename"], DelayedCodeWrapper):
                    v["filename"] = v["filename"].args[0]
        h = {
            "returnn_config": conf,
            "python_epilog_hash": self.python_epilog_hash,
            "python_prolog_hash": self.python_prolog_hash,
        }
        if self.staged_network_dict:
            h["returnn_networks"] = self.staged_network_dict

        return sis_hash_helper(h)