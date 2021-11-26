import torch
import sys
import numpy as np
from scipy import linalg

class model(torch.nn.Module):
	def __init__(self, **kwargs):
		super(type(self), self).__init__()
		# PARAMS
		params = {}
		for key in ['Nsongs', 'dim', 'embname', 'bias', 'blocksN', 'reduction_emb', 'hN']:
			params[key] = kwargs[key]
		for k, v in params.items():
			setattr(self, k, v)
		
		# CHECK for chunks in AffineCoupling
		self.dim = self.dim[0]
		if self.dim%2 != 0: print("ERROR: dim must be even"); sys.exit(0)
		if self.dim > 300: print("ERROR: maximum dim = 300"); sys.exit(0)

		# EMBEDDING
		model = torch.load(self.embname)
		self.register_buffer('forward_emb', model["emb.weight"]) #row of 0 for padding_idx
		self.register_buffer('reverse_emb', model["inv.weight"])
		if self.bias: self.register_buffer('reverse_bias', model["inv.bias"])
		del model
		
		self.emb_f = torch.nn.Embedding(self.Nsongs+1, self.dim, padding_idx=0) #extra index for padding
		self.emb_r = torch.nn.Linear(self.dim, self.Nsongs, bias=self.bias) #bias=0
		self.emb_f.weight.data.copy_(self.forward_emb)
		self.emb_r.weight.data.copy_(self.reverse_emb)
		if self.bias: self.emb_r.bias.data.copy_(self.reverse_bias)

		# BLOCKS
		self.blocks = torch.nn.ModuleList()
		for i in range(self.blocksN):
			 self.blocks.append(Block(dim))
		
		# MLP
		self.MLP = MLP(self.dim, self.hN)

		# LOAD TRAINED FLOW (IF NEEDED)
		#if self.trained_flow is not None:
		#	model = torch.load(self.trained_flow)
		#	keys = list(model.keys())
		#	for i in range(len(keys)):
		#		self.register_buffer(keys[i], model[keys[i]])

		return


	def forward(self, inp, out): #x = batch = matrix (tensor)
		# INP and OUT
		if self.reduction_emb == "sum": 
			emb_inp = self.emb_f(inp+1).sum(1) #x+1 for padding_idx = 0 (-1+1)

			emb_out = self.emb_f(out+1).sum(1)

		if self.reduction_emb == "mean": 
			emb_inp = self.emb_f(inp+1).sum(1)
			Nitems = (inp != 0).sum(1).float().to(emb_inp.device)
			emb_inp = emb_inp / Nitems.view(Nitems.shape[0],1)

			emb_out = self.emb_f(out+1).sum(1)
			Nitems = (out != 0).sum(1).float().to(emb_out.device)
			emb_out = emb_out / Nitems.view(Nitems.shape[0],1)

		logdet_inp = 0
		h_inp = emb_inp
		for block in self.blocks:
			h_inp, ldet = block.forward(h_inp)
			logdet_inp += ldet
		z_inp = h_inp

		logdet_out = 0
		h_out = emb_out
		for block in self.blocks:
			h_out, ldet = block.forward(h_out)
			logdet_out += ldet
		z_out = h_out

		# MLP
		z_inp_MLP = z_inp.data.to(emb_inp.device)
		z_inp_MLP.requires_grad = True
		z_out_MLP = self.MLP.forward(z_inp_MLP)

		h_out_inv = emb_out
		for block in self.blocks[::-1]:
			h_out_inv = block.reverse(h_out_inv)
		z_out_inv = h_out_inv.data.to(emb_inp.device)
		z_out_inv.requires_grad = True

		return z_inp, logdet_inp, z_out, logdet_out, z_out_MLP, z_out_inv

	def decoder(self, h):
		for block in self.blocks[::-1]:
			h = block.reverse(h)
		h = self.emb_r(h)

		return h

	def latent(self, x):
		return

	def calculate_matrix(self):
		for block in self.blocks:
			block.blocks[1].calculate_matrix()
		return


#######################################################################################

class Block(torch.nn.Module):
	def __init__(self, dim):
		super(type(self), self).__init__()

		self.blocks = torch.nn.ModuleList()
		self.blocks.append(ActNorm(dim))
		self.blocks.append(InvConv(dim))
		self.blocks.append(AffineCoupling(dim))

		return

	def forward(self, h):
		logdet = 0
		for block in self.blocks:
			h, ldet = block.forward(h)
			logdet += ldet

		return h, logdet

	def reverse(self, h):
		for block in self.blocks[::-1]:
			h = block.reverse(h)

		return h

	def latent(self, h):
		return self.forward(h)[0]


class MLP(torch.nn.Module):
	def __init__(self, dim, h):
		super(type(self), self).__init__()

		self.dim = dim
		self.h = h
		self.model = torch.nn.ModuleList()
		for i in range(self.h - 1):
			self.model.append(torch.nn.Linear(self.dim, self.dim))
			self.model.append(torch.nn.ReLU())
		self.model.append(torch.nn.Linear(self.dim, self.dim))

		return

	def forward(self, inp):
		for i in range(len(self.model)):
			inp = self.model[i](inp)
		return inp

#######################################################################################s

class ActNorm(torch.nn.Module):
	def __init__(self, dim):
		super(type(self), self).__init__()

		self.dim = dim
		self.s = torch.nn.Parameter(torch.ones(1, dim))
		self.b = torch.nn.Parameter(torch.zeros(1, dim))

		return

	def forward(self, h):
		h = self.s*h + self.b
		logdet = self.dim*self.s.abs().log().sum()

		return h, logdet

	def reverse(self, h):
		h = (h - self.b)/self.s
		return h

	def latent(self, h):
		return self.forward(h)[0]


class InvConv(torch.nn.Module):
	def __init__(self, dim):
		super(type(self), self).__init__()

		self.dim = dim
		
		W = np.random.randn(dim, dim)
		Q, _ = linalg.qr(W) # Q = orthogonal ==> det(W) = 1 ==> logdet(W)=0 (initial)
		P, L, U= linalg.lu(Q.astype(np.float32)) # LU decomposition
		S = np.diag(U)
		U = np.triu(U, 1)
		U_mask = np.triu(np.ones_like(U),1) # make U always triu
		L_mask = U_mask.T # make L always triu, 

		self.register_buffer('P',torch.from_numpy(P))
		self.register_buffer('U_mask',torch.from_numpy(U_mask))
		self.register_buffer('L_mask',torch.from_numpy(L_mask))
		self.register_buffer('L_eye',torch.eye(L_mask.shape[0])) #L will need 1 on the diagonal
		self.register_buffer('S_sign',torch.sign(torch.from_numpy(S)))

		self.L = torch.nn.Parameter(torch.from_numpy(L)) 
		self.S = torch.nn.Parameter(torch.log(1e-7 + torch.abs(torch.from_numpy(S))))
		self.U = torch.nn.Parameter(torch.from_numpy(U))

		self.W = None
		self.invW = None

		return

	def forward(self, h):
		if type(self.W) == type(None): # if W is not imposed, calculate W in each forward
			W = (self.P @ (self.L*self.L_mask + self.L_eye) @ 
				(self.U*self.U_mask + torch.diag(self.S_sign*self.S.exp())) )
		else:
			W = self.W
		h = torch.mm(W, h.t()).t()
		logdet = self.dim*self.S.sum()

		return h, logdet

	def reverse(self, h):
		if type(self.invW) == type(None):
			invW = (self.P @ (self.L*self.L_mask + self.L_eye) @ 
				(self.U*self.U_mask + torch.diag(self.S_sign*self.S.exp())) ).inverse()
		else: 
			invW = self.invW
		h = torch.mm(invW, h.t()).t()
		return h

	def latent(self, h):
		return self.forward(h)[0]

	def calculate_matrix(self):
		self.invW = (self.P @ (self.L*self.L_mask + self.L_eye) @ 
				(self.U*self.U_mask + torch.diag(self.S_sign*self.S.exp())) ).inverse()
		self.W = (self.P @ (self.L*self.L_mask + self.L_eye) @ 
				(self.U*self.U_mask + torch.diag(self.S_sign*self.S.exp())) )
		return


class AffineCoupling(torch.nn.Module):
	def __init__(self, dim):
		super(type(self), self).__init__()

		self.dim = dim
		self.NN = NN(dim)

		return

	def forward(self, h):
		h1, h2 = torch.chunk(h, 2, dim=1)
		logs, t = self.NN(h2)
		s = logs.exp()
		h1 = s*h1 + t
		h = torch.cat((h1, h2), dim=1)
		logdet = s.abs().log().sum()
		return h, logdet

	def reverse(self, h):
		h1, h2 = torch.chunk(h, 2, dim=1)
		logs, t = self.NN(h2)
		s = logs.exp()
		h1 = (h1 - t)/s
		h = torch.cat((h1, h2), dim=1)
		return h

	def latent(self, h):
		return self.forward(h)[0]

class NN(torch.nn.Module):
	def __init__(self, dim):
		super(type(self), self).__init__()

		self.dim = dim//2 #for torch.chunk
		self.model = torch.nn.Sequential(
			torch.nn.Linear(self.dim, self.dim), 
			torch.nn.ReLU(), 
			torch.nn.Linear(self.dim, self.dim), 
			torch.nn.ReLU(), 
			torch.nn.Linear(self.dim, self.dim*2))

		return

	def forward(self, h):
		h_ = self.model(h)
		logs, t = torch.chunk(h_, 2, dim=1)
		logs = torch.sigmoid(logs + 2) + 1e-7
		return logs, t


#######################################################################################