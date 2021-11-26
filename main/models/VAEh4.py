import torch

class model(torch.nn.Module):
	def __init__(self, **kwargs):
		"""
		emdim : int
				Embedding matrix = "10^6" x emdim
				(en realitat els 10^6 són 181497)
		Scheme
		------
		ENCODER												Linear
														  |-------> mu 		(dim[2])
		 INP     Embed+ReLU    H1   Linear+ReLU     H2    |
		"10^6" ------------> dim[0] ------------> dim[1] -|
														  | Linear
														  |-------> logvar	(dim[2])
		
		LATENT SPACE

		z = mu + N(0, exp(logvar)) #revisar si es passa mu, std o mu, var en la funció torch.random

		DECODER

		   z    Linear+ReLU     H3     Linear+ReLU     H4     Linear    OUTOUT
		dim[2] -------------> dim[1] --------------> dim[0] ----------> "10^6"

		
		No té el mateix número que la multiple_layers

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
		self.mu = torch.nn.Linear(self.dim[1], self.dim[2])
		self.mu.bias.data.fill_(0)
		self.logvar = torch.nn.Linear(self.dim[1], self.dim[2])
		self.logvar.bias.data.fill_(0)
		self.w3 = torch.nn.Linear(self.dim[2], self.dim[1])
		self.w4 = torch.nn.Linear(self.dim[1], self.dim[0])
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

	def encoder(self, x):
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
		mu = self.mu(h)
		logvar = self.logvar(h)
		return mu, logvar

	def reparametrize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		z = mu + std*torch.randn_like(std) #std inside randn_like only determines the size of the tensor of N(0,I)
		return z

	def decoder(self, z):
		h = self.relu(self.w3(z))
		h = self.relu(self.w4(h))
		xhat = self.inv(h)
		return xhat

	def forward(self, x, tag=None): #x = batch = matrix (tensor)
		if tag is None:
			with torch.no_grad(): self.emb.weight[0] = 0 #padding_idx elements always 0
			mu, logvar = self.encoder(x)
			z = self.reparametrize(mu, logvar)
			xhat = self.decoder(z)
		else:
			with torch.no_grad(): self.emb.weight[0] = 0 #padding_idx elements always 0
			mu, logvar = self.encoder(x)
			z = self.reparametrize(mu, logvar)
			z = z + self.z_tag[tag]
			xhat = self.decoder(z)
		return xhat, [mu, logvar]

	def latent(self, x):
		mu, logvar = self.encoder(x)
		z = self.reparametrize(mu, logvar)
		return z