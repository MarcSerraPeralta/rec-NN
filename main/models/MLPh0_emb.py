import torch

class model(torch.nn.Module):
	def __init__(self, **kwargs):
		"""
		emdim : int
				Embedding matrix = "10^6" x emdim
				(en realitat els 10^6 són 181497)
		Scheme
		------
		 INP     Embed+ReLU    H1      Linear    OUTOUT
		"10^6" ------------> emdim ------------> "10^6"
		"""
		super(model, self).__init__()
		# PARAMS
		params = {}
		for key in ['Nsongs', 'dim', 'reduction_emb']:
			params[key] = kwargs[key]
		for k, v in params.items():
			setattr(self, k, v)

		# STRUCTURE
		self.emb = torch.nn.Embedding(self.Nsongs+1, self.dim[0], padding_idx=0) #extra index for padding
		self.inv = torch.nn.Linear(self.dim[0], self.Nsongs, bias=True)
		self.inv.bias.data.copy_(torch.load("results/metadata/bias_inicialization"))

		# ATTENTION
		if self.reduction_emb == "attention":
			self.attention_l = torch.nn.Linear(self.dim[0],1)
			self.attention_a = torch.nn.Tanh()

		return

	def forward(self, x): #x = batch = matrix (tensor)
		with torch.no_grad(): self.emb.weight[0] = 0 #padding_idx elements always 0
		h, _ = self.encoder(x)
		xhat = self.decoder(h)
		return xhat, None

	def encoder(self, x):
		if self.reduction_emb == "sum": 
			h = self.emb(x+1).sum(1) #x+1 for padding_idx = 0 (-1+1)
		if self.reduction_emb == "mean": 
			h = self.emb(x+1).sum(1)
			Nitems = (x != 0).sum(1).float().to(h.device) # normalize
			h = h / Nitems.view(Nitems.shape[0],1)
		if self.reduction_emb == "attention":
			h = self.emb(x+1)
			att = self.attention_a(self.attention_l(h))
			att = torch.softmax(att, dim=1)
			h = (att*h).sum(dim=1)
			h = self.relu(h)
		return h, None

	def decoder(self, x):
		xhat = self.inv(x)
		return xhat

	def latent(self, x):
		return self.encoder(x)[0]