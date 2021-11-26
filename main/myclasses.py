import torch
import importlib
import sys

sys.path.append("./main/models")
from . import wrapper
from .utils import LOSS

##############################################################################################################

class MODEL():
	def __init__(self):
		
		self.default_params = wrapper.PARAM_WRAPPER().get_default_params("mymodel")
		self.params = self.default_params.copy()
		self.names = None
		self.embdim = None
		self.loss_f = None
		self.data = {"z_mean":None, "z_cluster":None, "z_embedded":None, "plot_z":None, "plot_tag":None}
		
		return

	def update_names(self):
		# CHECK
		for key in ['dim', 'mod', 'loss']: # param_path not needed to load the model
			if self.params[key] is None: return "ERROR (MODEL.update_names): param '{}' is None".format(key)
		# DIM NAME
		self.embdim = str(self.dim[0])
		for i in self.dim[1:]:
			self.embdim += "-" + str(i)

		# MODEL NAME
		self.name = self.mod + "_" + self.embdim + "_" + self.loss
		for prefix in ['beta', 'betastart', 'bias', 'embname', 'blocksN', 'hN', 'reduction_emb', 'lr']:
			if self.params[prefix] == self.default_params[prefix]: continue

			if type(self.params[prefix]) == float:
				self.name += "_{0}-{1:g}".format(prefix, self.params[prefix]).replace(".", "-")
			else: 
				self.name += "_{0}-{1}".format(prefix, self.params[prefix])

		# DATA NAMES
		for key in self.data.keys():
			self.data[key] = key + "_" + self.name

		return None

	def set_params(self, **kwargs):

		param_names = self.default_params.keys()
		for i,j in kwargs.items():
			if i in param_names: self.params[i] = j

		self.update_params()

		return

	def update_params(self):
		for k, v in self.params.items():
			setattr(self, k, v)
		return

	def load_model(self):
		# CHECK
		for key in ['mod', 'device']: # param_path not needed to load the model
			if self.params[key] is None: return [None, "ERROR (MODEL.load_model): param '{}' is None".format(key)]

		# IMPORT MODEL
		#model_lib = importlib.import_module(self.model_path.replace("/", ".") + "." + self.mod)
		model_lib = importlib.import_module(self.mod)
		model_lib = model_lib.model(**self.params).to(self.device)
		model_lib.eval()
		
		return [model_lib, None]

	def load_model_params(self, model=None):
		# CHECK
		for key in ['mod', 'param_path']:
			if self.params[key] is None: return [None, "ERROR (MODEL.load_model_params): param '{}' is None".format(key)]
		if self.name is None: 
			error = self.update_names()
			if error is not None: return error

		if model is None:
			# IMPORT MODEL AND LOAD PARAMS
			#model = importlib.import_module(self.model_path + "." + self.mod)
			model = importlib.import_module(self.mod)
			model = model.model(**self.params).to(self.device)
			model.load_state_dict(torch.load(self.param_path + "/" + self.name))
			model.eval()
		else:
			model.load_state_dict(torch.load(self.param_path + "/" + self.name))
			model.eval()
		
		return [model, None]

	def load_loss(self):
		# CHECK
		if self.loss is None: return "ERROR (MODEL.load_loss): No loss function specified"

		myloss = LOSS(self)
		if self.loss == 'BCE':
			self.loss_f = myloss.BCE_loss

		elif self.loss == 'BCE2':
			self.loss_f = myloss.BCE_loss2

		elif self.loss == 'CE':
			self.loss_f = myloss.CE_loss

		elif self.loss == 'BCE+KLD':
			myloss.set_weight_epoch(lambda x:[1.,self.beta])
			self.loss_f = myloss.BCEKLD_loss

		elif self.loss == 'BCE+KLD2':
			myloss.set_weight_epoch(lambda x:[1.,self.beta])
			self.loss_f = myloss.BCEKLD_loss2

		elif self.loss == 'CE+KLD':
			myloss.set_weight_epoch(lambda x:[1.,self.beta])
			self.loss_f = myloss.CEKLD_loss

		elif self.loss == 'FOCAL':
			self.loss_f = myloss.FOCAL_loss

		elif self.loss == 'FOCAL2':
			self.loss_f = myloss.FOCAL_loss2

		elif self.loss == 'FOCAL+KLD':
			myloss.set_weight_epoch(lambda x:[1.,self.beta])
			self.loss_f = myloss.FOCALKLD_loss

		elif self.loss == 'FOCAL+KLD2':
			myloss.set_weight_epoch(lambda x:[1.,self.beta])
			self.loss_f = myloss.FOCALKLD_loss2

		elif self.loss == 'FLOW':
			self.loss_f = myloss.FLOW_loss

		elif self.loss == "dummy":
			pass

		else:
			return "ERROR (MODEL.load_loss): No loss function named " + str(self.loss)

		return None

	def update_loss(self):
		error = self.load_loss()
		return error

	def songlist2input(self, SONG_INDEXES):
		SONG_INDEXES = torch.LongTensor(SONG_INDEXES)
		if len(SONG_INDEXES) > self.idim:
			SONG_INDEXES = SONG_INDEXES[torch.randint(0,len(SONG_INDEXES),(self.idim,))]
		x = -torch.ones(self.idim, dtype=torch.long)
		x[range(len(SONG_INDEXES))] = torch.LongTensor(SONG_INDEXES)
		x = x.reshape(1,self.idim).to(self.device)
		return x


##############################################################################################################

class METADATA():
	def __init__(self):
		
		self.default_params = wrapper.PARAM_WRAPPER().get_default_params("mymetadata")
		self.params = self.default_params.copy()

		self.userset = None
		self.metadata = None
		self.postset = None
		self.meta2song = None

		return

	def set_params(self, **kwargs):

		param_names = self.default_params.keys()
		for i,j in kwargs.items():
			if i in param_names: self.params[i] = j

		self.update_params()

		return

	def update_params(self):
		for k, v in self.params.items():
			setattr(self, k, v)
		return

	def update_names(self):
		# CHECK
		for key, val in self.params.items(): # param_path not needed to load the model
			if val is None: return "ERROR (METADATA.update_names): param '{}' is None".format(key)

		self.userset = self.userset_path + "/" + self.userset_name
		self.metadata = self.metadata_path + "/" + self.metadata_name
		self.postset = self.postset_path + "/" + self.postset_name + "_{}_{}".format(self.bias_top, self.bias_normal)
		self.meta2song = self.metadata_path + "/" + self.metadata_name + "2song"

		return None

##############################################################################################################

class DATASET():
	def __init__(self):
		
		self.default_params = wrapper.PARAM_WRAPPER().get_default_params("mydataset")
		self.params = self.default_params.copy()

		return

	def set_params(self, **kwargs):

		param_names = self.default_params.keys()
		for i,j in kwargs.items():
			if i in param_names: self.params[i] = j

		self.update_params()

		return

	def update_params(self):
		for k, v in self.params.items():
			setattr(self, k, v)
		return