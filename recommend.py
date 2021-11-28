import torch
import importlib
import sys
import argparse
import numpy as np

import main.wrapper as wrapper
import main.z_data as z_data_lib
"""
METADATA TYPE
=============
This script is for songs that their metadata is a list of items (p.e. tags). 


RECOMMENDATION TYPES
=====================
* standard = regular recommendation, without tunning
* postfiltering = increases probability (from uniform probability) of the selected tags with alpha factor
* tunning = moves user vector in the latent space with linear combination using alpha as factor

"""

#########################################################################################################################

class RECOM_PARAMS():
	def __init__(self):
		
		self.default_params = wrapper.PARAM_WRAPPER().get_default_params("myrecom")
		self.params = self.default_params.copy()

		return

	def set_params(self, **kwargs):

		param_names = self.default_params.keys()
		for i,j in kwargs.items():
			if i in param_names: self.params[i] = j

		for k, v in self.params.items():
			setattr(self, k, v)

		return

#########################################################################################################################

class RECOMMENDER():
	def __init__(self):
		
		self.default_params = {'mymodel':None, 'mymetadata':None, 'myrecom':None, 'meta_inp_out':None, 'z_data_name':None}
		self.params = self.default_params.copy()
		self.metadata = None
		self.z_data = None

		return

	def set_params(self, **kwargs):

		for key in self.default_params.keys():
			if key in kwargs.keys(): self.params[key] = kwargs[key]

		for k, v in self.params.items():
			setattr(self, k, v)

		return

	def load_data(self):
		"""
		Load: metadata, z_mean, meta2song (for fast postfiltering)
		"""
		# CHECK
		for key, val in self.params.items():
			if val is None: return "ERROR (RECOMMENDER.load_data): param '{}' is None".format(key)

		# METADATA
		self.metadata, self.meta = torch.load(self.mymetadata.metadata)
		if self.mymetadata.metadata_type == "list": self.Nclasses = len(self.meta)

		# Z_DATA
		self.z_data, self.meta2idx = torch.load(self.mymetadata.z_data_path + "/" + self.z_data_name)

		# meta2song
		self.meta2song = torch.load(self.mymetadata.meta2song)

		return None

	def load_model(self):
		"""
		Load: model, param
		"""
		if self.params["mymodel"] is None: return "ERROR (RECOMMENDER.load_model): mymodel not loaded"

		self.model, error = self.mymodel.load_model_params()

		return error

	def load(self):

		error = self.load_data()
		if error is not None: return error
		error = self.load_model()
		if error is not None: return error

		return None

	def recommend(self, SONG_INDEXES, betas={}, alpha=1, PROB=False, factor=None):
		"""
		tunning = {class_i:alpha_i} where type(class_i) = str if meta_type != "float", and type(class_i) = float otherwise. 
		"""
		# CHECKING
		if SONG_INDEXES == []: return ["ERROR (RECOMMENDER.recommend): Empty input list", []]
		if (self.model is None) or (self.mymodel is None): return ["ERROR (RECOMMENDER.recommend): No model nor mymodel loaded", []]
		classes = list(betas.keys())
		for c in classes:
			if self.mymetadata.metadata_type == "list" and c not in self.meta: return ["ERROR (RECOMMENDER.recommend): KeyError in betas tunning", []]
			if self.mymetadata.metadata_type == "float" and (c < self.meta[0] or c > self.meta[1]): return ["ERROR (RECOMMENDER.recommend): RangeError in betas tunning", []]

		# CREATE USER VECTOR
		x = self.mymodel.songlist2input(SONG_INDEXES)

		# SELECT RECOMMENDATION TYPE
		r_type = self.myrecom.recomtype

		if r_type == "standard":
			# CALCULATE
			recom = self.r_standard(x, PROB=PROB)
		elif r_type == "postfiltering":
			# CHECKING
			if self.metadata is None: return ["ERROR (RECOMMENDER.recommend): No metadata loaded", []]
			# CALCULATE
			recom = self.r_postfilter(x, betas, alpha, PROB=PROB)
		elif r_type == "tunning":
			# CHECKING
			if self.metadata is None: return ["ERROR (RECOMMENDER.recommend): No metadata loaded", []]
			if self.z_data is None: return ["ERROR (RECOMMENDER.recommend): No z_data loaded", []]
			# CALCULATE
			self.tunned = None
			recom = self.r_tunning(x, betas, alpha, PROB=PROB)
		elif r_type == "tun+post":
			# CHECKING
			if self.metadata is None: return ["ERROR (RECOMMENDER.recommend): No metadata loaded", []]
			if self.z_data is None: return ["ERROR (RECOMMENDER.recommend): No z_data loaded", []]
			# CALCULATE
			self.tunned = None
			recom  =self.r_tun_post(x, betas, alpha, PROB=PROB, factor=factor)
		else:
			return ["ERROR (RECOMMENDER.recommend): KeyError in recommendation: " + str(r_type), []]

		return [None, recom]

	def r_standard(self, x, PROB=False):
		with torch.no_grad():
			recom, args = self.model.forward(x)

		if PROB: return recom #return probability

		# CALCULATE TOP N
		_,recom = torch.topk(recom, self.myrecom.topN)
		recom = recom.flatten().tolist()

		return recom

	def r_postfilter(self, x, betas, alpha, PROB=False):
		"""
		If the song has metadata:
		recom = recom * mean(1 + beta_i)^alpha
		"""
		if betas == {}: return self.r_standard(x)
		if (np.array(list(betas.values())) < 1E-5).all() and (np.array(list(betas.values())) > -1E-5).all(): return self.r_standard(x)

		with torch.no_grad():
			recom, args = self.model.forward(x)
			# TRANSFORM TO PROBABILTY
			if self.mymodel.loss in ["BCE", "BCE+KLD", "FOCAL", "FOCAL+KLD", "FOCAL2", "FOCAL+KLD2", "BCE2", "BCE+KLD2"]:
				recom = torch.sigmoid(recom)
			elif self.mymodel.loss in ["CE", "CE+KLD"]:
				softmax = torch.nn.Softmax(1)
				recom = softmax(recom - torch.max(recom)) #stability of softmax

		# TUNNING
		with torch.no_grad():
			tunning = torch.zeros(self.mymodel.Nsongs).to(self.mymodel.device)
			Ntunning = torch.zeros(self.mymodel.Nsongs).to(self.mymodel.device)
			for tag in list(betas.keys()):
					tunning[self.meta2song[tag]] += (1 + betas[tag])**alpha
					Ntunning[self.meta2song[tag]] += 1
			tunning[tunning == 0] = 1
			Ntunning[Ntunning == 0] = 1
			recom = recom * tunning / Ntunning

		if PROB: return recom #return probability
		# CALCULATE TOP N
		_,recom = torch.topk(recom, self.myrecom.topN)
		recom = recom.flatten().tolist()

		return recom

	def r_tunning(self, x, betas, alpha, PROB=False):
		if betas == {}: return self.r_standard(x)
		if (np.array(list(betas.values())) < 1E-5).all() and (np.array(list(betas.values())) > -1E-5).all(): return self.r_standard(x)
		
		# Z_MEAN_USER
		with torch.no_grad():
			z_mean_user = torch.zeros(self.mymodel.dim[-1]).to(self.mymodel.device)
			total = 0
			for song in [i for i in x.flatten().data.tolist() if i != -1]:
				if self.metadata[song] != -1:
					for tag in self.metadata[song]:
						z_mean_user += self.z_data[self.meta2idx[tag]]
						total += 1
			if total != 0: z_mean_user /= total
			else: z_mean_user = self.model.latent(x).flatten()
		#with torch.no_grad():
		#	z_mean_user = self.model.latent(x).flatten()

		# Z_MEAN_TUNNING
		classes = list(betas.keys())
		with torch.no_grad():
			z_mean_tunning = torch.zeros(self.mymodel.dim[-1]).to(self.mymodel.device)
			for c in classes:
				z_mean_tunning += betas[c]*self.z_data[self.meta2idx[c]]
			z_mean_tunning /= sum(betas.values())

		# APPLY TUNNING
		with torch.no_grad():
			z = self.model.latent(x).flatten()
			self.tunned = z + alpha*(z_mean_tunning - z_mean_user) #self.tunned to have acces to z_tunned for NDCG DIST
			recom = self.model.decoder(self.tunned)

		if PROB: return recom #return probability
		# CALCULATE TOP N
		_,recom = torch.topk(recom, self.myrecom.topN)
		recom = recom.flatten().tolist()

		return recom

	def r_tun_post(self, x, betas, alpha, PROB=False, factor=None):
		"""
		factor is for the relation between alpha in postfitering and tunning
		alpha_postfilter = alpha_tunning * factor
		"""
		# CHECK
		if factor is None: factor = 1.

		# TUNNING
		if betas == {}: return self.r_standard(x)
		if (np.array(list(betas.values())) < 1E-5).all() and (np.array(list(betas.values())) > -1E-5).all(): return self.r_standard(x)
		
		# Z_MEAN_USER
		with torch.no_grad():
			z_mean_user = torch.zeros(self.mymodel.dim[-1]).to(self.mymodel.device)
			total = 0
			for song in [i for i in x.flatten().data.tolist() if i != -1]:
				if self.metadata[song] != -1:
					for tag in self.metadata[song]:
						z_mean_user += self.z_data[self.meta2idx[tag]]
						total += 1
			if total != 0: z_mean_user /= total
			else: z_mean_user = self.model.latent(x).flatten()
		#with torch.no_grad():
		#	z_mean_user = self.model.latent(x).flatten()

		# Z_MEAN_TUNNING
		classes = list(betas.keys())
		with torch.no_grad():
			z_mean_tunning = torch.zeros(self.mymodel.dim[-1]).to(self.mymodel.device)
			for c in classes:
				z_mean_tunning += betas[c]*self.z_data[self.meta2idx[c]]
			z_mean_tunning /= sum(betas.values())

		# APPLY TUNNING
		with torch.no_grad():
			z = self.model.latent(x).flatten()
			self.tunned = z + alpha*(z_mean_tunning - z_mean_user) #self.tunned to have acces to z_tunned for NDCG DIST
			recom = self.model.decoder(self.tunned).view(1, self.mymodel.Nsongs) # it is probability

		# POSTFILTERING
		alpha *= factor 
		with torch.no_grad():
			# TRANSFORM TO PROBABILTY
			if self.mymodel.loss in ["BCE", "BCE+KLD", "FOCAL", "FOCAL+KLD", "FOCAL2", "FOCAL+KLD2", "BCE2", "BCE+KLD2"]:
				recom = torch.sigmoid(recom)
			elif self.mymodel.loss in ["CE", "CE+KLD"]:
				softmax = torch.nn.Softmax(1)
				recom = softmax(recom - torch.max(recom)) #stability of softmax

		with torch.no_grad():
			tunning = torch.zeros(self.mymodel.Nsongs).to(self.mymodel.device)
			Ntunning = torch.zeros(self.mymodel.Nsongs).to(self.mymodel.device)
			for tag in list(betas.keys()):
					tunning[self.meta2song[tag]] += (1 + betas[tag])**alpha
					Ntunning[self.meta2song[tag]] += 1
			tunning[tunning == 0] = 1
			Ntunning[Ntunning == 0] = 1
			recom = recom * tunning / Ntunning

		if PROB: return recom #return probability
		# CALCULATE TOP N
		_,recom = torch.topk(recom, self.myrecom.topN)
		recom = recom.flatten().tolist()

		return recom

	def user_latent(self, SONG_INDEXES):
		# CREATE USER VECTOR
		x = self.mymodel.songlist2input(SONG_INDEXES)

		# CALCULATE z
		with torch.no_grad():
			z = self.model.latent(x)
			
		return z

	def user_distance(self, SONG_INDEXES):

		# CREATE USER VECTOR
		x = self.mymodel.songlist2input(SONG_INDEXES)

		# CALCULATE z
		with torch.no_grad():
			z = self.model.latent(x)

		# CALCULATE distance
		dist = []
		for mean in z_mean:
			dist += [torch.dist(z, self.z_data)]
		dist = torch.Tensor(dist)

		# FILTER
		dist_tag = []
		for i in self.meta2idx:
			dist_tag += [[dist[self.meta2idx[i]].data.tolist(), i]]
		sorted_dist = sorted(dist_tag, key=lambda kv: kv[0]) #small to large distances

		return sorted_dist

#########################################################################################################################

def dist(x, y, S):
	"""
	Mahalanobis distance:
	d(x,y) = \\sqrt( (x-y)^T*S^-1*(x-y) )
	where S^-1 = diag(1/s_i^2 for s_i in std)
	Input must be torch.Tensor((1,N)) for x,y,S
	"""
	d = ((x-y).pow(2)/S).sum(1).sqrt()
	return d

def classes_dist(z, z_mean, z_std):
	"""
	Returns tensor with Mahalanobis distance between x and z_mean[tag]
	"""

	d = []
	for mean, std in zip(z_mean, z_std):
		d += [dist(z,mean,std.pow(2))]
	d = torch.Tensor(d)

	return d

#########################################################################################################################
