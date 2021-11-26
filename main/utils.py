import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import argparse

from . import wrapper

plt.ioff()

##############################################################################################################

class TRAINING_LOSS():
	"""
	Manages data regarding the evoluction of a model's loss in training. 
	"""
	def __init__(self):
		
		self.default_params = wrapper.PARAM_WRAPPER().get_default_params("training_loss")
		self.params = self.default_params.copy()

		self.loss_graph = [[]]
		self.patience_graph = [[]]
		self.plot_loss = None

		return

	def set_params(self, **kwargs):

		param_names = self.default_params.keys()
		for i,j in kwargs.items():
			if i in param_names: self.params[i] = j

		self.update_params()
		self.update_names()

		return

	def update_params(self):
		for k, v in self.params.items():
			setattr(self, k, v)
		return

	def update_names(self):
		# CHECK
		for key, val in self.params.items(): # param_path not needed to load the model
			if val is None: return "ERROR (TRAINING_LOSS.update_names): param '{}' is None".format(key)

		self.plot_loss = self.plot_path + "/loss_" + self.mymodel.name
		self.plot_loss_complete = self.plot_path + "/loss_complete_" + self.mymodel.name

		return None

	def load_data(self):
		# CHECK
		for key in self.params.keys():
			if self.params[key] is None: return "ERROR (TRAINING_LOSS.load_data): param '{}' is None".format(key)

		try:
			self.loss_graph, self.patience_graph = torch.load(self.plot_loss)
		except:
			return "ERROR (TRAINING_LOSS.load_data): error occurred while loading loss data from '{}'".format(self.plot_loss)

		return None

	def add(self, plot, patience):
		self.loss_graph[-1] += [plot]
		self.patience_graph[-1] += [patience]
		return

	def restart(self):
		self.loss_graph += [[]]
		self.patience_graph += [[]]
		return

	def save(self):
		torch.save([self.loss_graph, self.patience_graph], self.plot_loss)
		return

	def plot(self):
		# CHECKING
		plt.cla()
		if len(self.loss_graph) == [[]]: 
			error = self.load_data()
			if error != None: return error
		if len(self.loss_graph) == [[]]: return "ERROR (TRAINING_LOSS.plot): Empty data"

		loss, patience = self.loss_graph, self.patience_graph
		if len(loss[-1])==0: loss=loss[:-1]; patience=patience[:-1]

		# CREATE X, Y
		loss = [np.array(loss_i) for loss_i in loss]
		colors = ["blue", "red"]
		epoch = [np.array(range(len(loss[0]))) + 1]
		for i in np.arange(1, len(loss)): 
			epoch += [np.array(range(len(loss[i]))) + epoch[-1][-1] + 1]

		# CREATE PLOT
		if "MLP" in self.mymodel.mod: 
			fig1 = plt.figure(3, figsize=[7,5])
			ax11 = fig1.add_subplot(111)
			ax11.set_xlim(epoch[0][0], epoch[-1][-1])
			if ("BCE" in self.mymodel.loss) and (np.max([np.max(i) for i in loss]) > 200):
				ax11.set_ylim(0, 200)
			elif ("CE" in self.mymodel.loss) and (np.max([np.max(i) for i in loss]) > 20):
				ax11.set_ylim(0, 20)
		if ("VAE" in self.mymodel.mod) or ("flow" in self.mymodel.mod): 
			fig1 = plt.figure(3, figsize=[7*3,5])
			ax11 = fig1.add_subplot(131)
			ax12 = fig1.add_subplot(132)
			ax13 = fig1.add_subplot(133)
			ax11.set_xlim(epoch[0][0], epoch[-1][-1])
			ax12.set_xlim(epoch[0][0], epoch[-1][-1])
			ax13.set_xlim(epoch[0][0], epoch[-1][-1])
			if ("BCE" in self.mymodel.loss) and (np.max([np.max(i) for i in loss]) > 200):
				ymin = np.min([np.min(loss[0][:,0]), np.min(loss[1][:,0]), np.min(loss[2][:,0])])
				ymax = np.min([np.max(loss[0][:,0]), 100])
				ax11.set_ylim(ymin - 0.05*(ymax - ymin), ymax + 0.05*(ymax - ymin))

				ymin = np.min([np.min(loss[0][:,1]), np.min(loss[1][:,1]), np.min(loss[2][:,1])])
				ymax = np.min([np.max(loss[0][:,1]), self.mymodel.beta*1E4])
				ax12.set_ylim(ymin - 0.05*(ymax - ymin), ymax + 0.05*(ymax - ymin))

				ymin = np.min([np.min(loss[0][:,0]), np.min(loss[1][:,0]), np.min(loss[2][:,0])])
				ymax = np.min([np.max(loss[0][:,0]), 100])
				ax13.set_ylim(ymin - 0.05*(ymax - ymin), ymax + 0.05*(ymax - ymin))
			elif ("CE" in self.mymodel.loss) and (np.max([np.max(i) for i in loss]) > 20):
				ymin = np.min([np.min(loss[0][:,0]), np.min(loss[1][:,0]), np.min(loss[2][:,0])])
				ymax = np.min([np.max(loss[0][:,0]), 15])
				ax11.set_ylim(ymin - 0.05*(ymax - ymin), ymax + 0.05*(ymax - ymin))

				ymin = np.min([np.min(loss[0][:,1]), np.min(loss[1][:,1]), np.min(loss[2][:,1])])
				ymax = np.min([np.max(loss[0][:,1]), self.mymodel.beta*1E4])
				ax12.set_ylim(ymin - 0.05*(ymax - ymin), ymax + 0.05*(ymax - ymin))

				ymin = np.min([np.min(loss[0][:,0]), np.min(loss[1][:,0]), np.min(loss[2][:,0])])
				ymax = np.min([np.max(loss[0][:,0]), 15])
				ax13.set_ylim(ymin - 0.05*(ymax - ymin), ymax + 0.05*(ymax - ymin))

		# PLOTTING
		for i, (loss_i, epoch_i, patience_i) in enumerate(zip(loss, epoch, patience)):

			if "MLP" in self.mymodel.mod:
				ax11.plot(epoch_i, loss_i, "-", color=colors[i%2], linewidth=1)
				ax11.plot(epoch_i, loss_i, ".", color=colors[i%2], markersize=3)
				e_p = np.array([[epoch_i[p], loss_i[p]] for p in range(len(patience_i)) if patience_i[p] != 10])
				if len(e_p) != 0:
					ax11.plot(e_p[:,0], e_p[:,1], ".", color="black", markersize=3)


			if ("VAE" in self.mymodel.mod) or ("flow" in self.mymodel.mod): 
				ax11.plot(epoch_i, loss_i[:,0], "-", color=colors[i%2], linewidth=1)
				ax12.plot(epoch_i, loss_i[:,1], "-", color=colors[i%2], linewidth=1)
				ax13.plot(epoch_i, loss_i[:,2], "-", color=colors[i%2], linewidth=1)
				ax11.plot(epoch_i, loss_i[:,0], ".", color=colors[i%2], markersize=3)
				ax12.plot(epoch_i, loss_i[:,1], ".", color=colors[i%2], markersize=3)
				ax13.plot(epoch_i, loss_i[:,2], ".", color=colors[i%2], markersize=3)
				e_p0 = np.array([[epoch_i[p], loss_i[p,0]] for p in range(len(patience_i)) if patience_i[p] != 10])
				e_p1 = np.array([[epoch_i[p], loss_i[p,1]] for p in range(len(patience_i)) if patience_i[p] != 10])
				e_p2 = np.array([[epoch_i[p], loss_i[p,2]] for p in range(len(patience_i)) if patience_i[p] != 10])
				if len(e_p0) != 0:
					ax11.plot(e_p0[:,0], e_p0[:,1], ".", color="black", markersize=3)
				if len(e_p1) != 0:
					ax12.plot(e_p1[:,0], e_p1[:,1], ".", color="black", markersize=3)
				if len(e_p2) != 0:
					ax13.plot(e_p2[:,0], e_p2[:,1], ".", color="black", markersize=3)

		# SAVE PLOT
		fig1.tight_layout()
		fig1.savefig(self.plot_loss + ".pdf", format="pdf", bbox_inches='tight')

		plt.cla()
		fig1.clf()
		del ax11, fig1

		return None

	def plot_complete(self):
		"""
		loss, patience = torch.load()
		len(loss) = nº restarts
		len(loss[i]) = nº epochs in each restart
		len(loss[i][j]) = nº of losses in this model
		(same for patience)

				ALL RESTART1 ... RESTARTN
		LOSS1
		...
		LOSSN

		"""
		# CHECKING
		plt.cla()
		if len(self.loss_graph) == [[]]: 
			error = self.load_data()
			if error != None: return error
		if len(self.loss_graph) == [[]]: return "ERROR (TRAINING_LOSS.plot_complete): Empty data"

		loss, patience = self.loss_graph, self.patience_graph
		if loss == [[]]: return
		if len(loss[-1])==0: loss=loss[:-1]; patience=patience[:-1]

		#DATA
		Nrestarts = len(loss)
		Nlosses = len(loss[0][0])
		loss = [np.array(loss_i) for loss_i in loss]
		colors = ["blue", "red"]
		epoch = [np.array(range(len(loss[0]))) + 1]
		for i in np.arange(1, len(loss)): 
			epoch += [np.array(range(len(loss[i]))) + epoch[-1][-1] + 1]

		# CREATES AXES
		fig1 = plt.figure(4, figsize=[7*(Nrestarts + 1),5*Nlosses])
		axes = []
		for i in range(Nlosses*(Nrestarts+1)):
			axes += [fig1.add_subplot(Nlosses, Nrestarts+1, i+1)]

		# SETS AXES LIMITS
		axes_i = 0
		for i in range(Nlosses):
			axes[axes_i].set_xlim(epoch[0][0], epoch[-1][-1])
			axes_i += 1
			for j in np.arange(1, Nrestarts+1, 1):
				axes[axes_i].set_xlim(epoch[j-1][0], epoch[j-1][-1])
				axes_i += 1

		axes_i = 0
		for i in range(Nlosses):
			# PLOT ALL
			for j in range(Nrestarts):
				axes[axes_i].plot(epoch[j], loss[j][:,i], "-", color=colors[j%2], linewidth=1)
				axes[axes_i].plot(epoch[j], loss[j][:,i], ".", color=colors[j%2], markersize=3)
				e_p = np.array([[epoch[j][p], loss[j][p,i]] for p in range(len(patience[j])) if patience[j][p] != 10])
				if len(e_p) != 0:
					axes[axes_i].plot(e_p[:,0], e_p[:,1], ".", color="black", markersize=3)
			axes_i += 1

			for j in range(Nrestarts):
				# PLOT RESTART_i
				axes[axes_i].plot(epoch[j], loss[j][:,i], "-", color=colors[j%2], linewidth=1)
				axes[axes_i].plot(epoch[j], loss[j][:,i], ".", color=colors[j%2], markersize=3)
				e_p = np.array([[epoch[j][p], loss[j][p,i]] for p in range(len(patience[j])) if patience[j][p] != 10])
				if len(e_p) != 0:
					axes[axes_i].plot(e_p[:,0], e_p[:,1], ".", color="black", markersize=3)
				axes_i += 1

		# SAVE PLOT
		fig1.tight_layout()
		fig1.savefig(self.plot_loss_complete + ".pdf", format="pdf", bbox_inches='tight')

		plt.cla()
		fig1.clf()
		del axes, fig1

		return None

##############################################################################################################

class LOSS():
	"""
	Loss functions for the optimizer of the model. 
	"""
	def __init__(self, mymodel):
		self.Nsongs = mymodel.Nsongs
		self.beta = mymodel.beta
		self.dim = mymodel.dim
		self.BCELoss = torch.nn.BCELoss() # default reduction=mean
		self.GAUSS_CTE = -0.5*self.dim[-1]*np.log(2*np.pi).item()
		self.weight_epoch = lambda epoch: [1.,self.beta] # weights for losses in function of the epoch
		return

	def BCE_loss(self, y, ynew, pars):
		y = y.float()
		ynew = ynew.float()
		ynew = torch.sigmoid(ynew) #+ 1E-10 #from 0 to 1 for log
		loss = self.BCELoss(ynew, y)*self.Nsongs #input, target
		return [loss, torch.Tensor([loss.item()])]

	def BCE_loss2(self, y, ynew, pars):
		y = y.float()
		ynew = ynew.float()
		ynew = torch.sigmoid(ynew) #+ 1E-10 #from 0 to 1 for log
		loss = -torch.sum(y*torch.log(ynew + 1E-10) + (1-y)*torch.log(1-ynew + 1E-10), 1) / y.sum(1)
		return [torch.mean(loss), torch.Tensor([torch.mean(loss).item()])]

	def CE_loss(self, y, ynew, pars):
		y = y.float()
		ynew = ynew.float()
		ynew = torch.softmax(ynew, 1) #normalization for ynew being a pdf
		loss = -torch.sum(y*torch.log(ynew + 1E-10), 1)/y.sum(1)
		return [torch.mean(loss), torch.Tensor([torch.mean(loss).item()])]

	def BCEKLD_loss(self, y, ynew, pars, epoch=1):
		mu, logvar = pars
		lossBCE = self.weight_epoch(epoch)[0]*self.BCE_loss(y, ynew, pars)[0]
		lossKLD = self.weight_epoch(epoch)[1]*(-0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)).mean()
		return [lossBCE + lossKLD, torch.Tensor([lossBCE.item(), lossKLD.item(), lossBCE.item() + lossKLD.item()])]

	def BCEKLD_loss2(self, y, ynew, pars, epoch=1):
		mu, logvar = pars
		lossBCE = self.weight_epoch(epoch)[0]*self.BCE_loss2(y, ynew, pars)[0]
		lossKLD = self.weight_epoch(epoch)[1]*(-0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)).mean()
		return [lossBCE + lossKLD, torch.Tensor([lossBCE.item(), lossKLD.item(), lossBCE.item() + lossKLD.item()])]

	def CEKLD_loss(self, y, ynew, pars, epoch=1):
		mu, logvar = pars
		lossCE = self.weight_epoch(epoch)[0]*self.CE_loss(y, ynew, pars)[0]
		lossKLD = self.weight_epoch(epoch)[1]*(-0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)).mean()
		return [lossCE + lossKLD, torch.Tensor([lossCE.item(), lossKLD.item(), lossCE.item() + lossKLD.item()])]

	def FLOW_loss(self, y, h, logdet): #y, ynew, pars
		size = h.size()[1]
		lossp = (self.GAUSS_CTE - 0.5*h.pow(2).sum(1).mean())/size #sum all dim elements, mean for all batches
		lossdet = logdet/size
		return [- lossp - lossdet, torch.Tensor([- lossp.item(), -lossdet.item(), - lossp.item() - lossdet.item()])]

	def FOCAL_loss(self, y, x, pars): #y, ynew, pars
		alpha = 0.25 #controls data imbalance, 0.5 if balanced
		gamma = 2.0

		t = y.float()

		p = x.sigmoid().detach()
		pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
		w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
		w = w * (1-pt).pow(gamma)
		loss = torch.nn.functional.binary_cross_entropy_with_logits(x, t, w, reduction='mean') #input, target, weight
		loss = loss*self.Nsongs
		return [loss, torch.Tensor([loss.item()])]

	def FOCAL_loss2(self, y, x, pars): #y, ynew, pars
		alpha = 0.25 #controls data imbalance, 0.5 if balanced
		gamma = 2.0

		t = y.float()
		x = torch.sigmoid(x.float())

		p = x.detach()
		pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
		w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
		w = w * (1-pt).pow(gamma)
		loss = -torch.sum(w*(t*torch.log(x + 1E-10) + (1-t)*torch.log(1-x + 1E-10)), 1) / t.sum(1)
		return [torch.mean(loss), torch.Tensor([torch.mean(loss).item()])]

	def FOCALKLD_loss(self, y, x, pars, epoch=1): #y, ynew, pars
		mu, logvar = pars #VAE
		lossF = self.weight_epoch(epoch)[0]*self.FOCAL_loss(y, x, pars)[0]
		lossKLD = self.weight_epoch(epoch)[1]*(-0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)).mean()
		return [lossF + lossKLD, torch.Tensor([lossF.item(), lossKLD.item(), lossF.item() + lossKLD.item()])]

	def FOCALKLD_loss2(self, y, x, pars, epoch=1): #y, ynew, pars
		mu, logvar = pars #VAE
		lossF = self.weight_epoch(epoch)[0]*self.FOCAL_loss2(y, x, pars)[0]
		lossKLD = self.weight_epoch(epoch)[1]*(-0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)).mean()
		return [lossF + lossKLD, torch.Tensor([lossF.item(), lossKLD.item(), lossF.item() + lossKLD.item()])]

	def set_weight_epoch(self, f):
		self.weight_epoch = f
		return

##############################################################################################################