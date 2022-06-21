
class ConstructNNetJob(Job):
    """
    Construct a network using `returnn_common.nn`

    Requires that the worker env (`SIS_COMMAND` in settings.py) is able to run Returnn with Tensorflow.
    """

    def __init__(self,
                 returnn_root: tk.Path,
                 returnn_common_root: tk.Path,
                 network_file: tk.Path,
                 parameter_dict: Dict[str, Any]):
        """

        :param returnn_root:
        :param returnn_common_root:
        :param network_file:
        :param parameter_dict:
        """
        self.returnn_root = returnn_root
        self.returnn_common_root = returnn_common_root
        self.network_file = network_file
        self.parameter_dict = parameter_dict

        self.out_network_code = self.output_path("network_code.txt")
        self.out_base_code = self.output_path("base_code.txt")

    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        sys.path.insert(0, self.returnn_root.get())
        sys.path.insert(0, os.path.dirname(self.returnn_common_root.get()))

        print(sys.path)
        print(sys.executable)

        from returnn_common import nn

        def create_data(constructor_data: DataInitArgs):
            """
            convert NetConstructorData into an actual

            :param constructor_data:
            :return:
            """
            dims = [
                nn.Dim(
                    kind=nn.Dim.Types.Feature if c_dim.is_feature else nn.Dim.Types.Spatial,
                    description=c_dim.name,
                    dimension=instanciate_delayed(c_dim.dim)
                )
                for c_dim in constructor_data.dim_tags
            ]
            if constructor_data.has_batch_dim:
                dims = [nn.batch_dim] + dims

            if constructor_data.sparse_dim is not None:
                sparse_dim = nn.Dim(
                    kind=nn.Dim.Types.Feature,
                    description=constructor_data.sparse_dim.name,
                    dimension=instanciate_delayed(constructor_data.sparse_dim.dim)
                )
            else:
                sparse_dim = None

            data = nn.Data(
                name = constructor_data.name,
                available_for_inference=constructor_data.available_for_inference,
                dim_tags=dims,
                sparse_dim=sparse_dim,
                sparse=sparse_dim is not None,
            )
            return data

        local_parameter_dict = {k: create_data(var) if isinstance(var, DataInitArgs) else instanciate_delayed(var) for k, var in self.parameter_dict.items()}

        network_file_path = self.network_file.get()
        shutil.copy(network_file_path, "network_constructor.py")

        sys.path.insert(0, os.getcwd())
        from network_constructor import get_single_network

        net, name_ctx_network = get_single_network(**local_parameter_dict)
        serializer = nn.ReturnnConfigSerializer(name_ctx_network)
        base_string = serializer.get_base_extern_data_py_code_str()
        network_string = serializer.get_ext_net_dict_py_code_str(net, ref_extern_data_dims_via_global_config=True)

        with open(self.out_network_code.get(), "wt") as out_f:
            out_f.write(network_string)

        with open(self.out_base_code.get(), "wt") as out_f:
            out_f.write(base_string)