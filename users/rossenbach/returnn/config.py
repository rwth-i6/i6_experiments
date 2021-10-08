import json
import os
import pprint
import string

from sisyphus.hash import sis_hash_helper

from i6_core.returnn.config import ReturnnConfig, instanciate_delayed


class ExtendedReturnnConfig(ReturnnConfig):

    def __init__(
            self,
            config,
            staged_network_dict=None,
            post_config=None,
            *,
            python_prolog=None,
            python_prolog_hash=None,
            python_epilog="",
            python_epilog_hash=None,
            hash_full_python_code=False,
            pprint_kwargs=None,
    ):
        super().__init__(
            config=config,
            post_config=post_config,
            python_prolog=python_prolog,
            python_prolog_hash=python_prolog_hash,
            python_epilog=python_epilog,
            python_epilog_hash=python_epilog_hash,
            hash_full_python_code=hash_full_python_code,
            pprint_kwargs=pprint_kwargs
        )

        self.staged_network_dict = staged_network_dict


    def write_network(self, config_path):
        """
        :param str config_path:
        :param dict network:
        :param int epoch:
        :return:
        """
        config_dir = os.path.dirname(config_path)
        network_dir = os.path.join(config_dir, "networks")
        if not os.path.exists(network_dir):
            os.mkdir(network_dir)

        init_file = os.path.join(network_dir, "__init__.py")
        init_import_code = "\n"
        init_dict_code = "\n\nnetworks_dict = {\n"

        for epoch in self.staged_network_dict.keys():
            network_path = os.path.join(network_dir, "network_%i.py" % epoch)
            pp = pprint.PrettyPrinter(indent=2, width=150, **self.pprint_kwargs)
            content = "\nnetwork = %s" % pp.pformat(self.staged_network_dict[epoch])
            with open(network_path, "wt", encoding="utf-8") as f:
                f.write(content)
            init_import_code += "from .network_%i import network as network_%i\n" % (epoch, epoch)
            init_dict_code += "  %i: network_%i,\n" % (epoch, epoch)

        init_dict_code+="}\n"

        with open(init_file, "wt", encoding="utf-8") as f:
            f.write(init_import_code + init_dict_code)

    def write(self, path):
        if self.staged_network_dict:
            self.write_network(path)
        with open(path, "wt", encoding="utf-8") as f:
            f.write(self.serialize())

    def serialize(self):
        self.check_consistency()
        config = self.config
        config.update(self.post_config)

        config = instanciate_delayed(config)

        config_lines = []
        unreadable_data = {}

        pp = pprint.PrettyPrinter(indent=2, width=150, **self.pprint_kwargs)
        for k, v in sorted(config.items()):
            if pprint.isreadable(v):
                config_lines.append("%s = %s" % (k, pp.pformat(v)))
            else:
                unreadable_data[k] = v

        if len(unreadable_data) > 0:
            config_lines.append("import json")
            json_data = json.dumps(unreadable_data).replace('"', '\\"')
            config_lines.append('config = json.loads("%s")' % json_data)
        else:
            config_lines.append("config = {}")

        python_prolog_code = self._parse_python(self.python_prolog)
        python_epilog_code = self._parse_python(self.python_epilog)

        if self.staged_network_dict:
            get_network_string = "\ndef get_network(epoch, **kwargs):\n"
            get_network_string += "  from networks import networks_dict\n"
            get_network_string += "  while(True):\n"
            get_network_string += "    if epoch in networks_dict:\n"
            get_network_string += "      return networks_dict[epoch]\n"
            get_network_string += "    else:\n"
            get_network_string += "      epoch -= 1\n"
            get_network_string += "      assert epoch > 0, \"Error, no networks found\"\n"

            config_lines.append(get_network_string)

            python_prolog_code = (
                    "import os\nimport sys\nsys.path.insert(0, os.path.dirname(__file__))\n\n" + python_prolog_code)

        python_code = string.Template(self.PYTHON_CODE).substitute(
            {
                "PROLOG": python_prolog_code,
                "REGULAR_CONFIG": "\n".join(config_lines),
                "EPILOG": python_epilog_code,
            }
        )
        return python_code

    def _sis_hash(self):
        h = {
            "returnn_config": self.config,
            "python_epilog_hash": self.python_epilog_hash,
            "python_prolog_hash": self.python_prolog_hash,
            "returnn_networks": self.staged_network_dict,
        }
        return sis_hash_helper(h)
