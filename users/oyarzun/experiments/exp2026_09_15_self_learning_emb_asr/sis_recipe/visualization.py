

from sisyphus import Job, Task, gs, setup_path, tk

from i6_core.returnn.config import ReturnnConfig
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable

from i6_experiments.common.setups.returnn.datasets import Dataset


class VisualizeEmbeddingsJob(Job): 

    def __init__(
            self, 
            prefix_name: str,
            returnn_config: ReturnnConfig,
            checkpoint: tk.Path,
            recognition_dataset: Dataset,
            dataset_name: str,
            returnn_exe: tk.Path,
            returnn_root: tk.Path,
            rqmt: Optional[Dict[str, int]] = None,
            lowercase_ref: bool = False,
            apply_text_norm: bool = False,
            vocab_opts: Optional[Dict] = None,
            recog_post_proc_funcs: Optional[List[Callable[[tk.Path], tk.Path]]] = None,
            score_function: Optional[Callable] = None
            ): 
        output_file_names = []
        self.dataset_name = dataset_name
        self.recognition_dataset= recognition_dataset
        pass


    def preprocess_data(self): 
        
        dataset_dict = self.recognition_dataset.as_returnn_opts() # TODO  understand what this does


        pass



    def tasks(self): 

        yield Task("preprocess_data")
        yield Task("generate_plots")


    


    def generate_plots(self): 
        pass