import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

from . import dataset
from . import wrapper


class Z_MEAN():
	def __init__(self):
		
		self.default_params = wrapper.PARAM_WRAPPER().get_default_params("z_data")
		self.params = self.default_params.copy()

		self.z_mean_dir = None
		self.mean_class = None
		self.meta2idx = None

		return

	def set_params(self, **kwargs):

		param_names = self.default_params.keys()
		for i,j in kwargs.items():
			if i in param_names: self.params[i] = j

		self.update_params()

		return None

	def update_params(self):
		for k, v in self.params.items():
			setattr(self, k, v)
		return

	def update_names(self):
		for key in ["mymetadata", "mymodel"]:
			if self.params[key] is None: return "ERROR (Z_MEAN.update_names): param '{}' is None".format(key)

		self.z_mean_dir = self.mymetadata.z_data_path + "/" + self.mymodel.data["z_mean"]

		if self.z_data_name is None:
			self.z_data_name = self.mymodel.data["z_mean"]
			self.params['z_data_name'] = self.mymodel.data["z_mean"]
		else:
			self.z_data_name = self.mymetadata.z_data_path + "/" + self.z_data_name

		return None

	def load_data(self, meta_inp_out=""):
		# CHECK
		if self.z_mean_dir is None: return "ERROR (Z_MEAN.load_data): no names found (update_names)"
		if meta_inp_out == "": return "ERROR (Z_MEAN.load_data): specify meta_inp_out"

		try: 
			self.mean_class, self.meta2idx = torch.load(self.z_mean_dir + "_" + meta_inp_out)
		except:
			return "ERROR (Z_MEAN.load_data): could not load data"

		return None

	def calculate(self, model, meta_inp_out="", force_calculation=False, mean="users"):
		if mean == "users":
			error = self.calculate_1(model, meta_inp_out=meta_inp_out, force_calculation=force_calculation)
			return error
		elif mean == "meta2song":
			error = self.calcualte_2(model, force_calculation=force_calculation)
			return error
		else: 
			return "ERROR (Z_MEAN.calculate): no option for mean named '{}'".format(mean)

		return

	def calculate_1(self, model, meta_inp_out="", force_calculation=False):
		"""
		meta_inp_out selects if it will use the metadata of the input or output to calculate the z_mean
		"""
		if not force_calculation:
			error = self.load_data(meta_inp_out = meta_inp_out)
			if error is None: return None # data is already calculated

		# CHECK
		for key in self.params.keys():
			if self.params[key] is None: return "ERROR (Z_MEAN.calculate): param '{}' is None".format(key)
		if meta_inp_out not in ['inp', 'out']: return "ERROR (Z_MEAN.calculate): specify meta_inp_out ('inp' or 'out')"
		if self.mymodel.data["z_mean"] is None: return "ERROR (Z_MEAN.calculate): specify name of z_mean in mymodel.data"
		
		# INPUT DATA
		print("\rLoading data..." + " "*20, end="")
		if self.mymodel.mod != "flow":
			model.eval()
		else:
			model_emb, model_flow = model
			model_emb.eval()
			model_flow.eval()
		metadata, meta = torch.load(self.mymetadata.metadata)
		meta2idx = {meta[i]:i for i in range(len(meta))}
		idx2meta = {j:i for i,j in meta2idx.items()}
		Nclass = len(meta)
		Nsongs, idim, dim, device = self.mymodel.Nsongs, self.mymodel.idim, self.mymodel.dim, self.mymodel.device

		train_ds = dataset.UserSet(self.mymetadata.userset, tsplit='train', idim=idim, seed=self.mydataset.seed, Nsongs=Nsongs, pc_split=self.mydataset.pc_split)
		train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

		# CALCULATE MEAN_CLASS
		print("\rCalculating z_mean..." + " "*20, end="")
		mean_class = torch.zeros((Nclass, dim[-1])).to(device)
		counts_class = torch.zeros(Nclass).to(device)
		Ntotal = len(train_dl)
		with torch.no_grad():
			for b, (inp, out) in enumerate(train_dl):
				print("\rCalculating z_mean... {0:6.2f}%".format((b+1)/Ntotal*100.) + " "*20,end="")
				if self.mymodel.mod != "flow":
					z = model.latent(inp.to(device))
				else:
					z = model_emb.latent(inp.to(device))
					z = model_flow.latent(z)
				meta_i = []
				if meta_inp_out == "inp": data = inp[0,:len((inp+1).nonzero())].long().data.tolist()
				if meta_inp_out == "out": data = out.view(-1).nonzero().view(-1).data.tolist()
				for i in data:
					if metadata[i] != -1:
						meta_i += metadata[i]
				if len(meta_i) != 0:
					topmeta = max(set(meta_i), key=meta_i.count)
					mean_class[meta2idx[topmeta]] += z.view(-1)
					counts_class[meta2idx[topmeta]] += 1

		for i in range(Nclass):
			if counts_class[i] == 0: counts_class[i] = 1 #solves nan problem (1/0)

		mean_class = mean_class/counts_class.unsqueeze(1)

		print("\rSaving z_mean..." + " "*20, end="")
		torch.save([mean_class, meta2idx], self.z_mean_dir + "_" + meta_inp_out)

		return None

	def calculate_2(self, model, force_calculation=False):
		# CHECK IF DATA ALREADY EXISTS (if force_calculation==False)
		if not force_calculation:
			error = self.load_data(meta_inp_out = meta_inp_out)
			if error is None: return None # data is already calculated

		# CHECK
		for key in self.params.keys():
			if self.params[key] is None: return "ERROR (Z_MEAN.calculate): param '{}' is None".format(key)
		if self.mymodel.data["z_mean"] is None: return "ERROR (Z_MEAN.calculate): specify name of z_mean in mymodel.data"

		# INPUT DATA
		print("\rLoading data..." + " "*20, end="")
		model.eval()
		#metadata, meta = torch.load(self.mymetadata.metadata)
		#meta2idx = {meta[i]:i for i in range(len(meta))}
		#idx2meta = {j:i for i,j in meta2idx.items()}
		#Nclass = len(meta)
		#Nsongs, idim, dim, device = self.mymodel.Nsongs, self.mymodel.idim, self.mymodel.dim, self.mymodel.device

		"""
		Mirar en el bitbucket com ho vaig fer l'altre vegada (diria que ja està programat) i copiar-ho i adaptar-ho. 
		"""

		return


##############################################################################################################


class Z_CLUSTER():
	def __init__(self):
		
		self.default_params = wrapper.PARAM_WRAPPER().get_default_params("z_data")
		self.params = self.default_params.copy()

		self.z_cluster = None
		self.cluster2tags = None

		self.z_cluster_dir = None

		return

	def set_params(self, **kwargs):

		param_names = self.default_params.keys()
		for i,j in kwargs.items():
			if i in param_names: self.params[i] = j

		self.update_params()

		return None

	def update_params(self):
		for k, v in self.params.items():
			setattr(self, k, v)
		return

	def update_names(self):
		for key in ["mymetadata", "mymodel", "Nclusters"]:
			if self.params[key] is None: return "ERROR (Z_CLUSTER.update_names): param '{}' is None".format(key)

		self.z_cluster_dir = self.mymetadata.z_data_path + "/z_clusters_{}_{}_{}".format(self.mymodel.name, self.Nclusters, self.N_users)
		self.z_cluster_log_dir = self.mymetadata.z_data_path + "/z_clusters_log_{}_{}_{}.txt".format(self.mymodel.name, self.Nclusters, self.N_users)
		self.z_cluster_plot_dir = self.mymetadata.z_data_path + "/z_clusters_plot_{}_{}_{}.pdf".format(self.mymodel.name, self.Nclusters, self.N_users)

		return None

	def load_data(self):
		# CHECK
		for key in ["mymodel", "mymetadata"]:
			if self.params[key] is None: return "ERROR (Z_CLUSTER.load_data): param '{}' is None".format(key)

		try:
			self.z_cluster, self.cluster2tags = torch.load(self.z_cluster_dir)
		except:
			return "ERROR (Z_CLUSTER.load_data): could not load data"

		return None

	def calculate(self, model, meta_inp_out="", force_calculation=False):
		if not force_calculation:
			error = self.load_data()
			if error is None: return None # data is already calculated

		# CHECK
		for key in ["mymodel", "mymetadata", "Nclusters", "N_users"]:
			if self.params[key] is None: return "ERROR (Z_CLUSTER.calculate): param '{}' is None".format(key)
		if meta_inp_out == "": return "ERROR (Z_CLUSTER.calculate): specify meta_inp_out ('inp' or 'out')"
		
		# LOAD DATA
		Nsongs, idim, dim, device = self.mymodel.Nsongs, self.mymodel.idim, self.mymodel.dim, self.mymodel.device
		metadata, meta = torch.load(self.mymetadata.metadata)

		train_ds = dataset.UserSet(self.mymetadata.userset, tsplit='train', idim=idim, seed=self.mydataset.seed, Nsongs=Nsongs, pc_split=self.mydataset.pc_split)
		train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

		if self.N_users == -1: N_users = len(train_dl)
		else: N_users = self.N_users 

		# CALCULATE Z_USER
		print("\rCalculating z_user..." + " "*20, end="")
		X = torch.Tensor().to(self.mymodel.device)
		b = 0
		with torch.no_grad():
			for inp, out in train_dl:
				if b >= N_users: break
				print("\rCalculating z_user... {0:0.2f}% Ntotal={1}".format((b+1)/N_users*100, N_users) + " "*20,end="")
				with torch.no_grad():
					z = model.latent(inp.to(self.mymodel.device))
				X = torch.cat((X, z.view((1,len(z[0])))), 0)
				b += 1


		# CALCULATE CLUSTERS
		print("\rCalculating clusters {}... ".format(meta_inp_out) + " "*20, end="")
		kmeans = KMeans(n_clusters=self.Nclusters, random_state=0).fit(X.cpu().detach().numpy())
		self.z_clusters = kmeans.cluster_centers_ # idx cluster 2 center 

		# SEPARATE USERS
		print("\rSeparating users {}... ".format(meta_inp_out) + " "*20, end="")
		self.cluster2tags = {i:{tag:0 for tag in meta} for i in range(self.Nclusters)} # idx cluster 2 histogram tags 
		b = 0

		for inp, out in train_dl:
			if b >= N_users: break
			if meta_inp_out == "inp": data = inp[0,:len((inp+1).nonzero())].long().data.tolist()
			if meta_inp_out == "out": data = out.view(-1).nonzero().view(-1).data.tolist()
			for i in data: # data is the song_id of the user
				if metadata[i] != -1:
					for tag in metadata[i]:
						self.cluster2tags[kmeans.labels_[b]][tag] += 1
			b += 1

		torch.save([self.z_clusters, self.cluster2tags], self.z_cluster_dir)

		return None

	def write_cluster2tags(self, topNtag = 20):
		if self.cluster2tags is None:
			error = self.load_data()
			if error is not None: return error

		msg = ""
		for c in range(len(self.cluster2tags)):
			msg += "-"*20 + "\nCLUSTER #{}\n".format(c)
			toptags = sorted(self.cluster2tags[c].items(), key=lambda x:-x[1]) # sort in decreasing order
			msg += "\n".join(["{0:25}{1:8}".format(i,j) for i,j in toptags[:topNtag]]) + "\n"
			msg += "\n"

		f = open(self.z_cluster_log_dir, "w")
		f.write(msg)
		f.close()

		return None

	def plot_cluster2tags(self, topNtag = 20):

		# NAMES
		names = []
		for c in self.cluster2tags.keys():
			toptags = sorted(self.cluster2tags[c].items(), key=lambda x:-x[1])[:topNtag]
			names += [j[0] for j in toptags]
		names = list(set(names))

		# VALUES
		values = np.zeros((len(self.cluster2tags), len(names)))
		for c in self.cluster2tags.keys():
			values[c] = np.array([self.cluster2tags[c][n] for n in names])
			norm = np.sum(values[c]) #normalitze each cluster
			values[c] /= norm

		# NORMALIZATION (for the plot)
		for n in range(len(names)):
			values[:,n] /= np.sum(values[:,n])

		# PLOT
		plt.cla()
		fig = plt.figure(10)
		ax = fig.add_subplot(111)
		for c in range(len(self.cluster2tags)):
			ax.plot(range(len(names)), values[c], ".", label=c, bottom=np.sum(values[:c], axis=0))
			#ax.bar(names, values[c], label=c, bottom=np.sum(values[:c], axis=0))

		#ax.set_xticks(names, names, rotation='vertical')
		#ax.legend(loc='best')
		fig.tight_layout()
		fig.savefig(self.z_cluster_plot_dir, bbox_inches='tight', format='pdf')

		return None



##############################################################################################################

class PLOT_Z_DATA():
	def __init__(self):
		
		self.default_params = wrapper.PARAM_WRAPPER().get_default_params("z_data")
		self.params = self.default_params.copy()

		self.plot_z_dir = None
		self.plot_tag_dir = None
		self.z_embedded_dir = None
		self.N_users = None

		return

	def set_params(self, **kwargs):

		param_names = self.default_params.keys()
		for i,j in kwargs.items():
			if i in param_names: self.params[i] = j

		self.update_params()

		return None

	def update_params(self):
		for k, v in self.params.items():
			setattr(self, k, v)

		return

	def update_names(self):
		for key in ["mymetadata", "mymodel"]:
			if self.params[key] is None: return "ERROR (PLOT_Z_DATA.update_names): param '{}' is None".format(key)

		self.plot_z_dir = self.mymetadata.z_data_path + "/" + self.mymodel.data["plot_z"]
		self.plot_tag_dir = self.mymetadata.z_data_path + "/" + self.mymodel.data["plot_tag"]
		self.z_embedded_dir = self.mymetadata.z_data_path + "/" + self.mymodel.data["z_embedded"]

		if self.z_data_name is None:
			self.z_data_name = self.mymodel.data["z_mean"]
			self.params['z_data_name'] = self.mymodel.data["z_mean"]
		else:
			self.z_data_name = self.mymetadata.z_data_path + "/" + self.z_data_name

		return None

	def calculate(self, model, partition="test", meta_inp_out = "inp", N_users=-1):
		"""
		meta_inp_out selects if it will use the metadata of the input or output to calculate the z_mean
		"""
		# CHECK
		for key in self.params.keys():
			if self.params[key] is None: return "ERROR (PLOT_Z_DATA.calculate): param '{}' is None".format(key)
		if meta_inp_out not in ['inp', 'out']: return "ERROR (PLOT_Z_DATA.calculate): specify meta_inp_out ('inp' or 'out')"
		if N_users == -1: return "ERROR (PLOT_Z_DATA.calculate): specify a number of users to be plotted (N_users)"
		if self.mymodel.data["z_mean"] is None: return "ERROR (PLOT_Z_DATA.calculate): specify name of z_mean in mymodel.data"
		if self.mymodel.data["z_embedded"] is None: return "ERROR (PLOT_Z_DATA.calculate): specify name of z_embedded in mymodel.data"
		
		plt.cla()

		# GET Z_MEAN DATA
		print("\rLoadingng data..." + " "*20, end="")
		if self.mymodel.mod != "flow":
			model.eval()
		else:
			model_emb, model_flow = model
			model_emb.eval()
			model_flow.eval()
		metadata, meta = torch.load(self.mymetadata.metadata)
		z_mean, meta2idx = torch.load(self.z_data_name)
		idx2meta = {j:i for i,j in meta2idx.items()}
		Nclass = len(meta)
		Nsongs, idim, dim, device = self.mymodel.Nsongs, self.mymodel.idim, self.mymodel.dim, self.mymodel.device
		N_users = self.N_users

		data_ds = dataset.UserSet(self.mymetadata.userset, tsplit=partition, idim=idim, seed=self.mydataset.seed, Nsongs=Nsongs, pc_split=self.mydataset.pc_split)
		data_dl = DataLoader(data_ds, batch_size=1, shuffle=True, num_workers=4)

		# CALCULATE Z_USER
		print("\rCalculating z_user..." + " "*20, end="")
		X = torch.Tensor().to(device)
		Xi2class = {}
		classes = []
		b = 0
		with torch.no_grad():
			for inp, out in data_dl:
				if b >= N_users: break
				print("\r{0:0.2f}% Ntotal={1}".format((b+1)/(N_users + len(meta2idx))*100, N_users + len(meta2idx)) + " "*20,end="")
				if self.mymodel.mod != "flow":
					z = model.latent(inp.to(device))
				else:
					z = model_emb.latent(inp.to(device))
					z = model_flow.latent(z)
				meta_i = []
				if meta_inp_out == "inp": data = inp[0,:len((inp+1).nonzero())].long().data.tolist()
				if meta_inp_out == "out": data = data = out.view(-1).nonzero().view(-1).data.tolist()
				for i in data:
					if metadata[i] != -1:
						meta_i += metadata[i]
				if len(meta_i) != 0:
					topmeta = max(set(meta_i), key=meta_i.count)
					Xi2class[b] = topmeta
					classes += [topmeta]
					X = torch.cat((X, z.view((1,len(z[0])))), 0)
					b += 1

		for j in range(len(meta2idx)):
			print("\r{0:0.2f}% Ntotal={1}".format((b+1)/(N_users + len(meta2idx))*100, N_users + len(meta2idx)) + " "*20,end="")
			if torch.dist(z_mean, torch.zeros(z_mean.shape).to(device)) != 0: # excludes means that do not exist
				X = torch.cat((X, z_mean[j].view((1,len(z_mean[j])))), 0)
				Xi2class[b] = idx2meta[j] + "_ztype_" + meta_inp_out
				classes += [idx2meta[j]]
				b += 1

		# CREATE X, Y
		print("\rCalculating TSNE... " + " "*20, end="")
		z_embedded = TSNE(n_components=2, n_iter=1000).fit_transform(X.cpu().detach().numpy())
		
		classes = list(set(classes))
		colors = plt.cm.get_cmap('rainbow', len(classes))
		class2colors = {classes[i]:colors(i) for i in range(len(classes))}

		torch.save([z_embedded, Xi2class, classes, class2colors, meta_inp_out, partition], self.z_embedded_dir + "_{}_{}_{}".format(partition, meta_inp_out, N_users))

		return None


	def plot_tags_funct(self, tags, tags_mean=None, legend=True, model=None, N_users=-1, meta_inp_out = "", minN=1, partition="test", data=None):
		# CHECKS
		if type(tags) != list: return "ERROR (PLOT_Z_DATA.plot_tags_funct): tags must be a list"
		if meta_inp_out == "": return "ERROR (PLOT_Z_DATA.plot_tags_funct): specify meta_inp_out"

		if len(tags) == 0: return None

		# CALCULATES OR LOADS DATA
		print("\rLoading data..." + " "*20, end="")
		if data is None: # for plotting separated tags and not having to load data for every plot
			try:
				z_embedded, Xi2class, classes, class2colors, meta_inp_out, partition = torch.load(self.z_embedded_dir + "_{}_{}_{}".format(partition, meta_inp_out, N_users))
			except:
				if model is None: return "ERROR (PLOT_Z_DATA.plot_tags_funct): need a model to calculate z_embedded"
				error = self.calculate(model, N_users = N_users, meta_inp_out = meta_inp_out, partition = partition)
				if error is not None: return error
				z_embedded, Xi2class, classes, class2colors, meta_inp_out, partition = torch.load(self.z_embedded_dir + "_{}_{}_{}".format(partition, meta_inp_out, N_users))
		else:
			z_embedded, Xi2class, classes, class2colors, meta_inp_out, partition = data
		
		tags_name = tags[:] # copies list for plot name

		if tags == ["all"]: tags = classes[:] # pass list by value
		if tags_mean is None: tags_mean = tags
		if tags_mean == ["all"]: tags_mean = classes[:]


		# SELECTS USERS TO PLOT
		idx_tag = [] # tags and mean
		for b, Xi in enumerate(list(Xi2class.keys())):
			if Xi2class[Xi] in tags: idx_tag += [Xi]
			if "_ztype_" in Xi2class[Xi] and Xi2class[Xi].rsplit("_ztype_")[0] in tags_mean: idx_tag += [Xi]

		# CREATE PLOT
		print("\rPlotting tags {}...".format("-".join(tags_name)) + " "*20, end="")
		fig1 = plt.figure(4, figsize=[7*4,5*4])
		ax11 = fig1.add_subplot(111)

		# PLOTTING
		tags_not_labelled = tags[:] # pass list by value
		Nlabel = 0
		Nplotted = 0 # users plotted (excluding z_means)
		total = len(idx_tag)
		for b, i in enumerate(idx_tag):
			print("\rPlotting tags {2} {0}... {1:0.2f}%   ".format("-".join(tags_name), (b+1)/total*100, meta_inp_out) + " "*20, end="")
			x, y = z_embedded[i]
			class_i = Xi2class[i]
			if class_i in tags_not_labelled: #for legend
				ax11.plot(x, y, ".", label=class_i, color=class2colors[class_i], markersize=5)
				Nlabel += 1
				Nplotted += 1
				del tags_not_labelled[tags_not_labelled.index(class_i)]
			elif "_ztype_" in class_i:
				if class_i.rsplit("_ztype_")[1] == "inp":
					ax11.plot(x, y, marker="o", color=class2colors[class_i.rsplit("_ztype_")[0]], markersize=10) # circle
				if class_i.rsplit("_ztype_")[1] == "out":
					ax11.plot(x, y, marker="s", color=class2colors[class_i.rsplit("_ztype_")[0]], markersize=10) # square
			else:
				ax11.plot(x, y, ".", color=class2colors[class_i], markersize=5)
				Nplotted += 1

		if Nlabel == 0 or minN > Nplotted: plt.cla(); fig1.clf(); return None

		# SAVE PLOT
		if legend:
			ax11.legend(loc="best")
		fig1.tight_layout()
		if len(tags_mean) == 0: 
			fig1.savefig(self.plot_z_dir + "_{}_{}_{}_{}.pdf".format(partition, meta_inp_out, N_users, "-".join(tags_name)), format="pdf", bbox_inches='tight')
		else:
			fig1.savefig(self.plot_tag_dir + "_{}_{}_{}_{}.pdf".format(partition, meta_inp_out, N_users, "-".join(tags_name)), format="pdf", bbox_inches='tight')

		plt.cla()
		fig1.clf()
		del ax11, fig1

		return None

	def plot_tags(self, tags, tags_mean=None, legend=True, model=None, N_users=-1, meta_inp_out = "", minN=1, partition="test", separated=False):
		if type(tags) != list: return "ERROR (PLOT_Z_DATA.plot_tags): tags must be a list"
		if meta_inp_out == "": return "ERROR (PLOT_Z_DATA.plot_tags): specify meta_inp_out"
		if len(tags) == 0: return None

		if not separated: 
			error = self.plot_tags_funct(tags, tags_mean=tags_mean, legend=legend, model=model, N_users=N_users, meta_inp_out = meta_inp_out, minN=minN, partition=partition)
			return error

		if separated:
			print("\rLoading data..." + " "*20, end="")
			try:
				z_embedded, Xi2class, classes, class2colors, meta_inp_out, partition = torch.load(self.z_embedded_dir + "_{}_{}_{}".format(partition, meta_inp_out, N_users))
			except:
				if model is None: return "ERROR (PLOT_Z_DATA.plot_tags): need a model to calculate z_embedded"
				error = self.get_z_embedded(model, N_users = N_users, meta_inp_out = meta_inp_out, partition = partition)
				if error is not None: return error

			if tags == ["all"]: tags = classes[:]
			data = [z_embedded, Xi2class, classes, class2colors, meta_inp_out, partition]
			
			for tag in tags:
				error = self.plot_tags_funct([tag], tags_mean=[tag], legend=legend, model=model, N_users=N_users, meta_inp_out = meta_inp_out, minN=20, partition=partition, data=data)
				if error != None: return error

		return None