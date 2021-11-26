import torch
import sys
import numpy as np
from scipy import linalg

class model(torch.nn.Module):
	def __init__(self, **kwargs):
		super(type(self), self).__init__()
		# PARAMS
		params = {}
		for key in ['Nsongs', 'dim', 'embname', 'bias', 'blocksN', 'reduction_emb']:
			params[key] = kwargs[key]
		for k, v in params.items():
			setattr(self, k, v)
		
		# CHECK for chunks in AffineCoupling
		self.dim = self.dim[0]
		if self.dim%2 != 0: print("ERROR: dim must be even"); sys.exit(0)
		if self.dim > 300: print("ERROR: maximum dim = 300"); sys.exit(0)

		# BLOCKS
		self.blocks = torch.nn.ModuleList()
		for i in range(self.blocksN):
			 self.blocks.append(Block(self.dim))

		return


	def forward(self, h): #x = batch = matrix (tensor)
		logdet = 0
		for block in self.blocks:
			h, ldet = block.forward(h)
			logdet += ldet

		return h, logdet

	def decoder(self, h):
		for block in self.blocks[::-1]:
			h = block.reverse(h)

		return h

	def latent(self, h):
		return self.forward(h)[0]

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
		self.NN = MLP(dim)

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

class MLP(torch.nn.Module):
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