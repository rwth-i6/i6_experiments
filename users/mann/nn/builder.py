from i6_experiments.users.mann.nn.config import viterbi_ffnn
from i6_experiments.users.mann.nn.config import viterbi_lstm
from i6_experiments.users.mann.nn.config.tdnn import make_baseline as make_tdnn_baseline
from i6_experiments.users.mann.nn.config.constants import BASE_BW_LRS

class ConfigBuilder:
	def __init__(self, num_input):
		self.num_input = num_input
		self.config_args = {}
		self.network_args = {}
		self.ce_args = {}
		self.scales = {}
		self.encoder = None
		self.loss = "bw"
		self.prior = "povey"
		self.transforms = []
		self.updates = {}
		self.deletions = []
    
    # def register(self, system, name):
    #     self.system = system
    #     self.system.builders[name] = self.copy()

	def set_ffnn(self):
		self.encoder = viterbi_ffnn
		return self
	
	def set_lstm(self):
		self.encoder = viterbi_lstm
		return self
	
	def set_tdnn(self):
		self.encoder = make_tdnn_baseline
		return self
	
	def set_config_args(self, config_args):
		self.config_args = config_args
		return self
	
	def set_network_args(self, network_args):
		self.network_args = network_args
		return self
	
	def set_ce_args(self, **ce_args):
		self.ce_args = ce_args
		return self
	
	def set_pretrain(self, **kwargs):
		raise NotImplementedError()
		return self
	
	def set_transcription_prior(self):
		self.prior = "transcription"
		return self
	
	def set_loss(self, loss="bw"):
		assert loss in ["bw", "viterbi"]
		self.loss = loss
		return self
	
	def set_povey_prior(self):
		self.prior = "povey"
		return self
	
	def set_no_prior(self):
		self.prior = None
		return self
	
	def set_scales(self, am=None, prior=None, tdp=None):
		self.scales.update({
			m + "_scale": v
			for m, v in locals().items()
			if m != "self" and v is not None
		})
		return self
		
	def set_tina_scales(self):
		self.scales = {
			"am_scale": 0.3,
			"prior_scale": 0.1,
			"tdp_scale": 0.1
		}
		return self
	
	def copy(self):
		new_instance = type(self)(self.system)
		new_instance.config_args = self.config_args.copy()
		new_instance.network_args = self.network_args.copy()
		new_instance.scales = self.scales.copy()
		new_instance.ce_args = self.ce_args.copy()
		new_instance.transforms = self.transforms.copy()
		new_instance.updates = self.updates.copy()
		new_instance.deletions = self.deletions.copy()
		new_instance.encoder = self.encoder
		new_instance.prior = self.prior
		new_instance.loss = self.loss
		return new_instance
	
	def set_oclr(self, dur=None, **kwargs):
		from i6_experiments.users.mann.nn.learning_rates import get_learning_rates
		dur = dur or int(0.8 * max(self.system.default_epochs))
		self.update({
			"learning_rates": get_learning_rates(
				increase=dur, decay=dur, **kwargs
			),
			"newbob_multi_num_epochs" : self.system.default_nn_training_args["partition_epochs"].get("train", 1),
			"newbob_multi_update_interval" : 1,
		})
		return self
	
	def set_specaugment(self):
		from i6_experiments.users.mann.nn import specaugment
		self.transforms.append(specaugment.set_config)
		return self
	
	def update(self, *args, **kwargs):
		self.updates.update(*args, **kwargs)
		return self
	
	def delete(self, *args):
		self.deletions += args
		return self
	
	def build(self):
		from i6_experiments.users.mann.nn import BASE_BW_LRS
		from i6_experiments.users.mann.nn import prior, pretrain, bw, get_learning_rates
		kwargs = BASE_BW_LRS.copy()
		kwargs.update(self.config_args)

		if self.encoder is viterbi_lstm:
			viterbi_config_dict = self.encoder(
				self.num_input,
				network_kwargs=self.network_args,
				ce_args=self.ce_args,
				**kwargs)
		else:
			viterbi_config_dict = self.encoder(
				self.num_input,
				network_kwargs=self.network_args,
				**kwargs)

		assert "chunking" in viterbi_config_dict.config

		assert self.prior in ["povey", "transcription", None], "Unknown prior: {}".format(self.prior)

		if self.loss == "bw":
			config = bw.ScaleConfig.copy_add_bw(
				viterbi_config_dict, self.system.csp["train"],
				num_classes=self.system.num_classes(),
				prior=self.prior,
				**self.scales,
			)
		elif self.loss == "viterbi":
			config = bw.ScaleConfig.from_config(viterbi_config_dict)
		else:
			raise ValueError("Unknown loss: {}".format(self.loss))

		if self.prior == "transcription":
			assert self.loss == "bw"
			self.system.prior_system.add_to_config(config)

		config.config.update(copy.deepcopy(self.updates))

		for key in self.deletions:
			del config.config[key]

		for transform in self.transforms:
			transform(config)
		
		return config
	
	def build_compile_config(self):
		viterbi_config = self.encoder(
			self.system.num_input,
			network_kwargs=self.network_args,
		)

		net = viterbi_config.config["network"]
		for key in ["loss", "loss_opts", "targets"]:
			net["output"].pop(key, None)
		net["output"]["n_out"] = self.system.num_classes()

		pruned_config_dict = {
			k: v for k, v in viterbi_config.config.items()
			if k in ["network", "num_outputs", "extern_data"]
		}

		return crnn.ReturnnConfig(pruned_config_dict)
