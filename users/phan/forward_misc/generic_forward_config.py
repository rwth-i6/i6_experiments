"""
Generic forward config
"""

import copy
import torch
import torchaudio
from typing import Optional

from i6_experiments.users.schmitt import hdf
from returnn_common.datasets_old_2022_10.interface import DatasetConfig

from returnn.tensor import Dim, single_step_dim, TensorDict
from returnn.forward_iface import ForwardCallbackIface
from returnn_common import nn
from i6_core.returnn import ReturnnConfig
from typing import TYPE_CHECKING, Optional, Union, Any, Dict, Sequence, Collection, Iterator, Callable

from i6_core.util import instanciate_delayed
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_experiments.common.setups.returnn_common import serialization
from i6_experiments.users.gaudino.model_interfaces.model_interfaces import ModelDef, RecogDef, ModelWithCheckpoint, ModelWithCheckpoints
from i6_experiments.users.zeyer import tools_paths


def generic_forward_job(
	*,
	dataset: DatasetConfig,
	model: ModelWithCheckpoint,
	forward_def,
	forward_callback: ForwardCallbackIface,
	output_files,
	forward_extra_config: dict = {},
	forward_post_config: Optional[Dict[str, Any]] = None,
	forward_mem_rqmt: Union[int, float] = 6,
	forward_time_rqmt: Union[int, float] = 4,
	device: Optional[str] = "gpu",
	dataset_key: str = "train",
	job_vis_name: str = "",
	extra_hash = None,
):
	"""
	This "generic" forward is intended to work
	with Albert's model interface
	"""
	forward_job = ReturnnForwardJobV2(
		model_checkpoint=model.checkpoint,
		returnn_config=forward_config(
			dataset,
			model.definition,
			forward_def,
			forward_callback,
			forward_extra_config=forward_extra_config,
			post_config=forward_post_config,
			device=device,
			dataset_key=dataset_key,
			extra_hash=extra_hash,
		),
		output_files=output_files,
		returnn_python_exe=tools_paths.get_returnn_python_exe(),
		returnn_root=tools_paths.get_returnn_root(),
		mem_rqmt=forward_mem_rqmt,
		time_rqmt=forward_time_rqmt,
		device=device,
	)
	forward_job.set_vis_name(job_vis_name)
	
	return forward_job # Use forward_job.out_files to get all out files





def forward_config(
	dataset: Union[DatasetConfig, dict],
	model_def,
	forward_def,
	forward_callback,
	*,
	forward_extra_config: dict = {},
	post_config: Optional[Dict[str, Any]] = None,
	device: Optional[str] = "gpu",
	dataset_key: str = "train",
	extra_hash = None,
) -> ReturnnConfig:
	if isinstance(dataset, DatasetConfig):
		if dataset_key == "devtrain":
			forward_data = dataset.get_eval_datasets()["devtrain"]
		else:
			forward_data = dataset.get_main_dataset()
	else:
		forward_data = dataset
	returnn_align_config_dict = dict(
		backend=model_def.backend,
		behavior_version=model_def.behavior_version,
		# dataset
		default_input=dataset.get_default_input(),
		target=dataset.get_default_target(),
		forward_data=forward_data,
	)
	returnn_align_config_dict.update(forward_extra_config)
	extern_data_raw = dataset.get_extern_data()

	extern_data_raw = instanciate_delayed(extern_data_raw)

	# if model_args.get("preload_from_files", None):
	# 	model_args = copy.deepcopy(model_args)
	# 	preload_from_files = model_args.pop("preload_from_files")
	# 	returnn_align_config_dict["preload_from_files"] = preload_from_files
	# returnn_align_config_dict.update({
	# 	"search_args": search_args,
	# 	"model_args": model_args,
	# })
	serial_collection = [
		serialization.NonhashedCode(
			nn.ReturnnConfigSerializer.get_base_extern_data_py_code_str_direct(extern_data_raw)
		),
		serialization.Import(model_def, import_as="_model_def", ignore_import_as_for_hash=True),
		serialization.Import(_returnn_v2_get_model, import_as="get_model"),
		serialization.Import(forward_def, import_as="_forward_def", ignore_import_as_for_hash=True),
		serialization.Import(_returnn_v2_forward_step, import_as="forward_step"),
		serialization.Import(forward_callback, import_as="forward_callback"),
		serialization.ExplicitHash(
			{
				# Increase the version whenever some incompatible change is made in this recog() function,
				# which influences the outcome, but would otherwise not influence the hash.
				"version": "01/11/2024",
				"device": device,
			}
		),
		serialization.PythonEnlargeStackWorkaroundNonhashedCode,
		serialization.PythonCacheManagerFunctionNonhashedCode,
		serialization.PythonModelineNonhashedCode,
	]
	if extra_hash is not None:
		serial_collection.append(
			serialization.ExplicitHash(extra_hash)
		)
	returnn_align_config = ReturnnConfig(
		config=returnn_align_config_dict,
		python_epilog=[
			serialization.Collection(
				serial_collection
			)
		],
		post_config=dict(  # not hashed
			log_batch_size=True,
			# debug_add_check_numerics_ops = True
			# debug_add_check_numerics_on_output = True
			# flat_net_construction=True,
		),
		sort_config=False,
	)
	# batch_size = 20000 * (search_args.get("bsf", 0) if search_args.get("bsf", 0) > 0 else model_def.batch_size_factor)
	# max_seqs = model_def.max_seqs if search_args.get("max_seq", 0) == 0 else search_args.get("max_seq", 200)
	# max_seqs = search_args.get("max_seq", 200)
	# batch_size_dependent = search_args.get("batch_size_dependent", False)
	# (returnn_align_config.config if batch_size_dependent else returnn_align_config.post_config).update(
	# 	dict(
	# 		batching="sorted",
	# 		batch_size=batch_size,
	# 		max_seqs=max_seqs,
	# 	)
	# )

	if post_config:
		returnn_align_config.post_config.update(post_config)
	SharedPostConfig = {
		# In case pretraining overwrites some of these, they need a default.
		"accum_grad_multiple_step": None,
		"use_last_best_model": None,
	}
	for k, v in SharedPostConfig.items():
		if k in returnn_align_config.config or k in returnn_align_config.post_config:
			continue
		returnn_align_config.post_config[k] = v

	return returnn_align_config





def _returnn_v2_get_model(*, epoch: int, **_kwargs_unused):
	from returnn.tensor import Tensor
	from returnn.config import get_global_config

	config = get_global_config()
	default_input_key = config.typed_value("default_input")
	default_target_key = config.typed_value("target")
	extern_data_dict = config.typed_value("extern_data")
	data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
	targets = Tensor(name=default_target_key, **extern_data_dict[default_target_key])
	assert targets.sparse_dim and targets.sparse_dim.vocab, f"no vocab for {targets}"

	model_def = config.typed_value("_model_def")
	model = model_def(
		epoch=epoch,
		in_dim=data.feature_dim,
		target_dim=targets.sparse_dim,
		)
	return model


def _returnn_v2_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
	import returnn.frontend as rf
	from returnn.tensor import Tensor, Dim, batch_dim
	from returnn.config import get_global_config

	if rf.is_executing_eagerly():
		batch_size = int(batch_dim.get_dim_value())
		for batch_idx in range(batch_size):
			seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
			print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")

	config = get_global_config()
	default_input_key = config.typed_value("default_input")
	data = extern_data[default_input_key]
	data_spatial_dim = data.get_time_dim_tag()
	forward_def = config.typed_value("_forward_def")

	default_target_key = config.typed_value("target")
	targets = extern_data[default_target_key]
	targets_spatial_dim = targets.get_time_dim_tag()

	forward_def(
		model=model,
		data=data,
		data_spatial_dim=data_spatial_dim,
		non_blank_targets=targets,
		non_blank_targets_spatial_dim=targets_spatial_dim,
	)

	# if len(forward_out) == 2:
	# 	# realign results including viterbi_align,
	# 	# out_spatial_dim,
	# 	viterbi_align, out_spatial_dim = forward_out
	# else:
	# 	raise ValueError(f"unexpected num outputs {len(forward_out)} from align_def")
	# assert isinstance(viterbi_align, Tensor)
	# assert isinstance(out_spatial_dim, Dim)
	# rf.get_run_ctx().mark_as_output(viterbi_align, "viterbi_align", dims=[batch_dim, out_spatial_dim])



def _returnn_v2_get_forward_callback():
	from typing import TextIO
	import numpy as np
	from returnn.tensor import Tensor, TensorDict
	from returnn.forward_iface import ForwardCallbackIface
	from returnn.config import get_global_config
	from returnn.datasets.hdf import SimpleHDFWriter

	class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
		def __init__(self):
			self.alignment_file: Optional[SimpleHDFWriter] = None

		def init(self, *, model):
			target_dim = model.target_dim_w_blank.dyn_size_ext.raw_tensor.item() if model.target_dim_w_blank.dyn_size_ext is not None else model.target_dim_w_blank.size

			self.alignment_file = SimpleHDFWriter(
				filename=_v2_forward_out_filename, dim=target_dim, ndim=1
			)

		def process_seq(self, *, seq_tag: str, outputs: TensorDict):
			viterbi_align: Tensor = outputs["viterbi_align"]  # [T]
			assert len(viterbi_align.dims) == 1, f"expected hyps to be 1D, but got {viterbi_align.dims}"
			assert viterbi_align.dims[0].dyn_size_ext, f"viterbi_align {viterbi_align} does not define seq lengths"

			seq_len = viterbi_align.dims[0].dyn_size_ext.raw_tensor.item()
			viterbi_align_raw = viterbi_align.raw_tensor[:seq_len]

			hdf.dump_hdf_numpy(
				hdf_dataset=self.alignment_file,
				data=viterbi_align_raw[None],  # [1, T]
				seq_lens=np.array([seq_len]),  # [1]
				seq_tags=[seq_tag],
			)

		def finish(self):
			self.alignment_file.close()

	return _ReturnnRecogV2ForwardCallbackIface()

def _returnn_v2_forward_step_text_only(*, model, extern_data: TensorDict, **_kwargs_unused):
	import returnn.frontend as rf
	from returnn.tensor import Tensor, Dim, batch_dim
	from returnn.config import get_global_config

	if rf.is_executing_eagerly():
		batch_size = int(batch_dim.get_dim_value())
		for batch_idx in range(batch_size):
			seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
			print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")

	config = get_global_config()
	forward_def = config.typed_value("_forward_def")

	default_target_key = config.typed_value("target")
	targets = extern_data[default_target_key]
	targets_spatial_dim = targets.get_time_dim_tag()

	forward_def(
		model=model,
		non_blank_targets=targets,
		non_blank_targets_spatial_dim=targets_spatial_dim,
	)
