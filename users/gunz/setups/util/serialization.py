from i6_experiments.common.setups.serialization import SerializerObject

from sisyphus.hash import sis_hash_helper


class ModuleImport(SerializerObject):
    def __init__(self, module_name: str, import_as: str):
        super().__init__()

        self.module_name = module_name
        self.import_as = import_as

    def get(self) -> str:
        return f"import {self.module_name} as {self.import_as}\n"

    def _sis_hash(self):
        return sis_hash_helper({"module_name": self.module_name, "import_as": self.import_as})
