import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.ioff()

##############################################################################################################
# MAP

def average_precision(ytrue,ypred,k=None,eps=1e-10,reduce_mean=True):
	if k is None:	
		k=ypred.size(1)
	_,spred=torch.topk(ypred,k,dim=1)
	found=torch.gather(ytrue,1,spred)
	pos=torch.arange(1,spred.size(1)+1).unsqueeze(0).to(ypred.device)
	prec=torch.cumsum(found,1)/pos.float()
	mask=(found>0).float()
	ap=torch.sum(prec*mask,1)/(torch.sum(ytrue,1)+eps)
	if reduce_mean:
		return ap.mean()
	return ap

##############################################################################################################
# NDCG

def NDCG_F(ypred, reli, k=None, reduce_mean=True, normalized=True):
	if k is None:	
		k = ypred.size(1)
	_, idx = torch.topk(ypred, k, dim=1)
	ideal, _ = torch.topk(reli, k, dim=1)
	relik = torch.gather(reli, 1, idx)
	pos = torch.arange(1, relik.size(1) + 1).float().unsqueeze(0).to(ypred.device)
	DCG = torch.sum( (torch.pow(2,relik).float() - 1)/torch.log2(pos+1) , 1)
	if not normalized:
		return DCG
	IDCG = torch.sum( (torch.pow(2,ideal).float() - 1)/torch.log2(pos+1) , 1)
	if IDCG != 0:
		NDCG = DCG/IDCG
	else: 
		NDCG = DCG*0
	if reduce_mean:
		return NDCG.mean()
	return NDCG

class NDCG_CLASS():
	def __init__(self, params):

		self.NDCG = None
		self.default = {"mymodel":None, "Rtype":None, "outpath":"results/NDCG", 
				"meta_name":"opt_tags", "meta_path":"results/metadata", 
				"bias_top":1, "bias_normal":1, "alpha":None, "reli":[0.5],
				"minNclass":1, "topN":1000, "Z_TYPE":"out", "tunpost_factor":1}
		self.params = self.default.copy()
		self.set(params)

		return

	def set(self, params):

		for p in params.keys():
			self.params[p] = params[p]

		self.update()

		return

	def update(self):
		#"NDCG_all_(opt_tags)_(t1n1)_(z-out)_(min1)_(top1000)_(mean-y)_alpha-0-2_(reli-0.5)_model-name"

		outname = "NDCG_" + "{}_".format("-".join(map(str, self.params['Rtype'])))
		if len(self.params['alpha']) == 1:
			outname += "{}_".format(self.params['alpha'])
		else:
			outname += "{}-{}_".format(self.params['alpha'][0], self.params['alpha'][-1])

		for p in self.params.keys():
			if p in ["Rtype", "alpha", "mymodel"]: continue
			if (self.params[p] != self.default[p]) and (type(self.params[p])==list): 
				outname += "-".join(map(str, self.params[p])) + "_"
			if (self.params[p] != self.default[p]) and (type(self.params[p])!=list):
				outname += "{}_".format(self.params[p])

		outname += "{}".format(self.params["mymodel"].name)

		self.outname = self.params["outpath"] + "/" + outname

		return

	def get(self):
		
		self.NDCG = torch.load(self.outname)

		return

	def save(self, NDCG):

		torch.save(NDCG, self.outname)

		return

	def plot(self, legend=False):
		"""
		Assuming only one reli is used
		"""
		# CHECK
		if self.NDCG == None: return "ERROR: No NDCG loaded"
		method2linetype = {"tunning":"-", "postfiltering":"--", "tun+post":":"}
		NDCG = self.NDCG

		# TRANSFORM DATA
		indexes = NDCG.keys()
		alpha = []
		reli = 0
		method = []
		for a, t, r in indexes:
			if a not in alpha: alpha += [a]
			if t not in method: method += [t]
		reli = r
		alpha = sorted(alpha)
		data_types = list(NDCG[a,t,r].keys())
		classes = list(NDCG[a,t,r][data_types[1]].keys())
		if len(alpha)==1: print("ERROR: alpha must be a list (not int) for plotting"); sys.exit(0)

		fig1 = plt.figure(1, figsize=[25,10])
		axes = []
		if "NDCG" in data_types:
			axes += [fig1.add_subplot(1, len(data_types), 1)]
			axes[-1].set_xlim(alpha[0], alpha[-1])
			axes[-1].set_ylabel("NDCG_tag")
			for t in method:
				for class_ in classes:
					axes[-1].plot(alpha, [NDCG[a, t, reli]["NDCG"][class_] for a in alpha], method2linetype[t], label=class_)
				if legend and ("COUNTS" not in data_types) and ("DIST" not in data_types): axes[-1].legend(loc='center left', bbox_to_anchor=(1.1, 0.5)); legend=False #plot only one legent
			axes[-1].set_xlabel("alpha")

		if "COUNTS" in data_types:
			idx2reli = {0:0, 1:reli, 2:1}
			if "NDCG" in data_types: start = 2 
			for counts_i in range(3):
				axes += [fig1.add_subplot(3, len(data_types), start + len(data_types)*counts_i)]
				axes[-1].set_xlim(alpha[0], alpha[-1])
				axes[-1].set_ylabel("#reli {}".format(idx2reli[counts_i]))
				for t in method:
					for class_ in classes:
						axes[-1].plot(alpha, [NDCG[a, t, reli]["COUNTS"][class_][counts_i] for a in alpha], method2linetype[t], label=class_)
					if legend and ("DIST" not in data_types): axes[-1].legend(loc='center left', bbox_to_anchor=(1.1, -0.5)); legend=False #plot only one legent
			axes[-1].set_xlabel("alpha")

		if "DIST" in data_types:
			start = 1
			if "NDCG" in data_types: start += 1
			if "COUNTS" in data_types: start += 1
			idx2dist = {0:"dist(z_tunning, z_mean_tag)", 1:"dist(z_tunning, z_out)"}
			for counts_i in range(2):
				axes += [fig1.add_subplot(2, len(data_types), start + len(data_types)*counts_i)]
				axes[-1].set_xlim(alpha[0], alpha[-1])
				axes[-1].set_ylabel("{}".format(idx2dist[counts_i]))
				for t in method:
					for class_ in classes:
						axes[-1].plot(alpha, [NDCG[a, t, reli]["DIST"][class_][counts_i] for a in alpha], method2linetype[t], label=class_)
					if legend: axes[-1].legend(loc='center left', bbox_to_anchor=(1.1, -0.1)); legend=False #plot only one legent
			axes[-1].set_xlabel("alpha")

		#fig1.tight_layout()
		plt.subplots_adjust()
		fig1.savefig(self.outname + ".pdf", format="pdf", bbox_inches='tight')
		fig1.clf()
		del axes, fig1

		return None

	def plot_average(self):
		"""
		Assuming only one reli is used
		"""
		# CHECK
		if self.NDCG == None: return "ERROR: No NDCG loaded"
		method2linetype = {"tunning":"-", "postfiltering":"--", "tun+post":":"}
		NDCG = self.NDCG

		# TRANSFORM DATA
		indexes = NDCG.keys()
		alpha = []
		reli = 0
		method = []
		for a, t, r in indexes:
			if a not in alpha: alpha += [a]
			if t not in method: method += [t]

		reli = r
		alpha = sorted(alpha)
		data_types = list(NDCG[a,t,r].keys())
		classes = list(NDCG[a,t,r][data_types[1]].keys())
		if len(alpha)==1: print("ERROR: alpha must be a list (not int) for plotting"); sys.exit(0)

		# CALCULATE MEAN
		for a, t, r in indexes:
			if "NDCG" in data_types:
				NDCG[a,t,r]["NDCG"] = torch.tensor([i for i in NDCG[a,t,r]["NDCG"].values()]).mean().data.tolist()
			if "COUNTS" in data_types:
				NDCG[a,t,r]["COUNTS"] = torch.tensor([i.data.tolist() for i in NDCG[a,t,r]["COUNTS"].values()]).mean(0).data.tolist()
			if "DIST" in data_types:
				NDCG[a,t,r]["DIST"] = torch.tensor([i.data.tolist() for i in NDCG[a,t,r]["DIST"].values()]).mean(0).data.tolist()

		fig1 = plt.figure(1, figsize=[25,10])
		axes = []
		if "NDCG" in data_types:
			axes += [fig1.add_subplot(1, len(data_types), 1)]
			axes[-1].set_xlim(alpha[0], alpha[-1])
			axes[-1].set_ylabel("NDCG_tag")
			for t in method:
				axes[-1].plot(alpha, [NDCG[a, t, reli]["NDCG"] for a in alpha], method2linetype[t])
			axes[-1].set_xlabel("alpha")

		if "COUNTS" in data_types:
			idx2reli = {0:0, 1:reli, 2:1}
			if "NDCG" in data_types: start = 2 
			for counts_i in range(3):
				axes += [fig1.add_subplot(3, len(data_types), start + len(data_types)*counts_i)]
				axes[-1].set_xlim(alpha[0], alpha[-1])
				axes[-1].set_ylabel("#reli {}".format(idx2reli[counts_i]))
				for t in method:
					axes[-1].plot(alpha, [NDCG[a, t, reli]["COUNTS"][counts_i] for a in alpha], method2linetype[t])
			axes[-1].set_xlabel("alpha")

		if "DIST" in data_types:
			start = 1
			if "NDCG" in data_types: start += 1
			if "COUNTS" in data_types: start += 1
			idx2dist = {0:"dist(z_tunning, z_mean_tag)", 1:"dist(z_tunning, z_out)"}
			for counts_i in range(2):
				axes += [fig1.add_subplot(2, len(data_types), start + len(data_types)*counts_i)]
				axes[-1].set_xlim(alpha[0], alpha[-1])
				axes[-1].set_ylabel("{}".format(idx2dist[counts_i]))
				for t in method:
					axes[-1].plot(alpha, [NDCG[a, t, reli]["DIST"][counts_i] for a in alpha], method2linetype[t])
			axes[-1].set_xlabel("alpha")

		#fig1.tight_layout()
		plt.subplots_adjust()
		fig1.savefig(self.outname + "_average.pdf", format="pdf", bbox_inches='tight')
		fig1.clf()
		del axes, fig1

		return None

def worst_NDCG(topN, ones, relis, rel_i, Nsongs=180198):
	print("topN", topN, "ones", ones, "relis", relis, "reli", rel_i)

	inp_v1 = torch.tensor(range(topN)).view(1,topN)
	reli_v1 = torch.zeros(1, topN)
	reli_v1[0,:ones] = 1
	reli_v1[0,ones:relis+ones] = rel_i

	inp_v2 = torch.tensor(range(Nsongs)).view(1,Nsongs)
	reli_v2 = torch.zeros(1,Nsongs)
	reli_v2[0,:ones] = 1
	reli_v2[0,ones:relis+ones] = rel_i

	inp_v3 = torch.rand(100,Nsongs)

	print("worst NDCG (topN, reli)=", NDCG_F(inp_v1, reli_v1, k=topN).data.tolist())
	print("worst DCG =", NDCG_F(inp_v1, reli_v1, k=topN, normalized=False).data.tolist())
	print("worst NDCG (reli, topN)=", NDCG_F(inp_v2, reli_v2, k=topN).data.tolist())
	print("worst DCG =", NDCG_F(inp_v2, reli_v2, k=topN, normalized=False).data.tolist())
	print("rand NDCG (reli, topN)=", sum([NDCG_F(inp_v3[i].view(1,Nsongs), reli_v2, k=topN).data.tolist() for i in range(100)])/100)
	print("rand DCG =", sum([NDCG_F(inp_v3[i].view(1,Nsongs), reli_v2, k=topN, normalized=False).data.tolist() for i in range(100)])/100)

	return
