import torch

class model(torch.nn.Module):
	def __init__(self, **kwargs):
		"""
		emdim : int
				Embedding matrix = "10^6" x emdim
				(en realitat els 10^6 són 181497)
		Scheme
		------
		ENCODER								  Linear
											|-------> mu 		(dim[1])
		 INP     Embed+ReLU    H1           |
		"10^6" ------------> dim[0] --------|
											| Linear
											|-------> logvar	(dim[1])
		
		LATENT SPACE

		z = mu + N(0, exp(logvar)) #revisar si es passa mu, std o mu, var en la funció torch.random

		DECODER

		   z    Linear+ReLU     H2     Linear    OUTOUT
		dim[1] -------------> dim[0] ----------> "10^6"

		
		Té el mateix número que la multiple_layers

		"""
		super(model, self).__init__()
		# PARAMS
		params = {}
		for key in ['Nsongs', 'dim', 'reduction_emb', 'hN']:
			params[key] = kwargs[key]
		for k, v in params.items():
			setattr(self, k, v)

		# STRUCTURE
		self.emb = torch.nn.Embedding(self.Nsongs+1, self.dim[0], padding_idx=0) #extra index for padding
		self.mu = torch.nn.Linear(self.dim[0], self.dim[1])
		self.mu.bias.data.fill_(0)
		self.logvar = torch.nn.Linear(self.dim[0], self.dim[1])
		self.logvar.bias.data.fill_(0)
		self.w1 = torch.nn.Linear(self.dim[1], self.dim[0])
		self.inv = torch.nn.Linear(self.dim[0], self.Nsongs)
		self.relu = torch.nn.ReLU()
		self.inv.bias.data.copy_(torch.load("results/metadata/bias_inicialization"))

		self.MLP = MLP(self.dim[-1], self.hN)

		return

	def encoder(self, x):
		if self.reduction_emb == "sum": 
			h = self.relu(self.emb(x+1).sum(1)) #x+1 for padding_idx = 0 (-1+1)
		if self.reduction_emb == "mean": 
			h = self.emb(x+1).sum(1)
			Nitems = (x != 0).sum(1).float().to(h.device)
			h = h / Nitems.view(Nitems.shape[0],1)
			h = self.relu(h)
		mu = self.mu(h)
		logvar = self.logvar(h)
		return mu, logvar

	def reparametrize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		z = mu + std*torch.randn_like(std) #std inside randn_like only determines the size of the tensor of N(0,I)
		return z

	def decoder(self, z):
		h = self.relu(self.w1(z))
		xhat = self.inv(h)
		return xhat

	def forward(self, x): #x = batch = matrix (tensor)
		with torch.no_grad(): self.emb.weight[0] = 0 #padding_idx elements always 0
		mu, logvar = self.encoder(x)
		z = self.reparametrize(mu, logvar)
		z_t = self.MLP.forward(z)
		xhat = self.decoder(z_t)
		return xhat, [mu, logvar, z, z_t]

	def latent(self, x):
		mu, logvar = self.encoder(x)
		z = self.reparametrize(mu, logvar)
		return z

class MLP(torch.nn.Module):
	def __init__(self, dim, hlayers):
		super(type(self), self).__init__()

		self.dim = dim
		self.hlayers = hlayers
		self.model = torch.nn.ModuleList()
		for i in range(self.hlayers - 1):
			self.model.append(torch.nn.Linear(self.dim, self.dim))
			self.model.append(torch.nn.ReLU())
		self.model.append(torch.nn.Linear(self.dim, self.dim))

		return

	def forward(self, inp):
		for i in range(len(self.model)):
			inp = self.model[i](inp)
		return inp