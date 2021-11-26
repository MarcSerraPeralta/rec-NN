import torch

class model(torch.nn.Module):
	def __init__(self, **kwargs):
		"""
		emdim : int
				Embedding matrix = "10^6" x emdim
				(en realitat els 10^6 són 181497)
		Scheme
		------
		 INP     Embed+ReLU    H1   Linear+ReLU     H2   Linear+ReLU     H3   Linear+ReLU     H4   Linear+ReLU     H5   Linear+ReLU     H6   Linear+ReLU     H7   Linear    OUTOUT
		"10^6" ------------> emdim -------------> emdim- ------------> emdim  ------------> emdim  ------------> emdim  ------------> emdim  ------------> emdim ---------> "10^6"
		"""
		super(model, self).__init__()
		# PARAMS
		params = {}
		for key in ['Nsongs', 'dim', 'reduction_emb', 'Nmeta_classes']:
			params[key] = kwargs[key]
		for k, v in params.items():
			setattr(self, k, v)

		# STRUCTURE
		self.emb = torch.nn.Embedding(self.Nsongs+1, self.dim[0], padding_idx=0) #extra index for padding
		self.w1 = torch.nn.Linear(self.dim[0], self.dim[1])
		self.w2 = torch.nn.Linear(self.dim[1], self.dim[2])
		self.w3 = torch.nn.Linear(self.dim[2], self.dim[3])
		self.w4 = torch.nn.Linear(self.dim[3], self.dim[2])
		self.w5 = torch.nn.Linear(self.dim[2], self.dim[1])
		self.w6 = torch.nn.Linear(self.dim[1], self.dim[0])
		self.inv = torch.nn.Linear(self.dim[0], self.Nsongs)
		self.relu = torch.nn.ReLU()
		self.inv.bias.data.copy_(torch.load("results/metadata/bias_inicialization"))

		# TUNING
		self.z_tag = torch.nn.Parameter(torch.rand(self.Nmeta_classes, self.dim[-1]))

		# ATTENTION
		if self.reduction_emb == "attention":
			self.attention_l = torch.nn.Linear(self.dim[0],1)
			self.attention_a = torch.nn.Tanh()

		return

	def forward(self, x, tag=None): #x = batch = matrix (tensor)
		if tag is None:
			with torch.no_grad(): self.emb.weight[0] = 0 #padding_idx elements always 0
			h, _ = self.encoder(x)
			xhat = self.decoder(h)
		else:
			with torch.no_grad(): self.emb.weight[0] = 0 #padding_idx elements always 0
			h, _ = self.encoder(x)
			h = h + self.z_tag[tag]
			xhat = self.decoder(h)
		return xhat, None

	def encoder(self, x): #x = batch = matrix (tensor)
		if self.reduction_emb == "sum": 
			h = self.relu(self.emb(x+1).sum(1)) #x+1 for padding_idx = 0 (-1+1)
		if self.reduction_emb == "mean": 
			h = self.emb(x+1).sum(1)
			Nitems = (x != 0).sum(1).float().to(h.device)
			h = h / Nitems.view(Nitems.shape[0],1)
			h = self.relu(h)
		if self.reduction_emb == "attention":
			h = self.emb(x+1)
			att = self.attention_a(self.attention_l(h))
			att = torch.softmax(att, dim=1)
			h = (att*h).sum(dim=1)
			h = self.relu(h)
		h = self.relu(self.w1(h))
		h = self.relu(self.w2(h))
		z = self.relu(self.w3(h))
		return z, None

	def decoder(self, x):
		h = self.relu(self.w4(x))
		h = self.relu(self.w5(h))
		h = self.relu(self.w6(h))
		xhat = self.inv(h)
		return xhat

	def latent(self, x):
		return self.encoder(x)[0]
