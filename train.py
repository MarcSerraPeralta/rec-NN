import torch
from torch.utils.data import DataLoader
import argparse
import sys, os
import numpy as np

import main.dataset as dataset
import main.wrapper as wrapper
from main.myclasses import MODEL, METADATA, DATASET
from main.utils import TRAINING_LOSS
from main.evaluation import average_precision, NDCG_F
from main.z_data import Z_MEAN, Z_CLUSTER, PLOT_Z_DATA


def EPOCH_f(t_v, model, data_dl, evaluate_flag = False):
	
	if t_v == 'train':
		model.train()
	else:
		model.eval()

	dat_AP = torch.tensor([]).to(args.device)
	dat_NDCG = torch.tensor([]).to(args.device)
	total = len(data_dl) 

	running_loss = 0
	plot_loss = 0
	for b,(x, y) in enumerate(data_dl):
		x, y = x.to(args.device), y.to(args.device)

		if t_v == 'train':
			optimizer.zero_grad()
			ynew, pars = model.forward(x)
			loss, plot = mymodel.loss_f(y, ynew, pars)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() #necessary for best model in training set
			plot_loss += plot

			if args.train_tun:
				optimizer.zero_grad()
				tags = data_dl.dataset.get_tags(Nusers=len(x), Ntags=args.Nmeta_classes)
				ynew_t, pars = model.forward(x, tag=tags)
				loss, plot = mymodel.loss_f(y, ynew_t, pars)
				loss.backward()
				optimizer.step()
				running_loss += loss.item() #necessary for best model in training set
				plot_loss += plot
		else:
			with torch.no_grad():
				ynew, pars = model.forward(x)
				loss, plot = mymodel.loss_f(y, ynew, pars)
				running_loss += loss.item() #necessary for best model in training set
				plot_loss += plot

				if args.train_tun:
					tags = data_dl.dataset.get_tags(Nusers=len(x), Ntags=args.Nmeta_classes)
					ynew_t, pars = model.forward(x, tag=tags)
					loss, plot = mymodel.loss_f(y, ynew_t, pars)
					running_loss += loss.item() #necessary for best model in training set
					plot_loss += plot
		
		if args.train_tun:
			aveloss = running_loss / (2*b+2)
			ave_plot = plot_loss / (2*b+2)
		else:
			aveloss = running_loss / (b+1)
			ave_plot = plot_loss / (b+1)
		
		msg = "\r      {0:7.3f}%   loss = ".format((b+1)/total*100) + " / ".join(["{0:10.6f}".format(i) for i in ave_plot]) + " / {0:10.6f}".format(aveloss)

		# mAP
		if evaluate_flag:
			# normal
			AP = average_precision(y.float(), ynew.float(), reduce_mean=False, k=1000) #y_true, y_pred
			dat_AP = torch.cat((dat_AP, AP))
			mAP = torch.mean(dat_AP)

			msg += "   mAP = {0:0.7f}     ".format(mAP)

			# tuning
			if args.train_tun:
				curr_reli = class2vector[tags]*(0.9*y + 0.1)
				NDCG = NDCG_F(prob.view((1,mymodel.Nsongs)), curr_reli, reduce_mean=True, k=args.topN)
				dat_NDCG = torch.cat((dat_NDCG, NDCG))
				mNDCG = torch.mean(dat_NDCG)

				msg += "   mNDCG = {0:0.7f}     ".format(mNDCG)

		print(msg, end="")

	print(" ")

	if t_v == 'val' and not evaluate_flag:
		return aveloss, ave_plot.data.tolist()
	
	return

#################################################################################################

def POST_CALCULATIONS(model, force_calculation=False):

	model, error = mymodel.load_model_params(model) #load best model
	if error is not None: print(error); sys.exit(0)

	# SAVE LOSS GRAPH
	if 'PLOTLOSS' in args.TODO:
		print("Plotting loss...")
		if mymodel.loss != "dummy": 
			error = myloss.load_data()
			if error is not None: print(error); sys.exit(0)
			error = myloss.plot()
			if error is not None: print(error); sys.exit(0)
			error = myloss.plot_complete()
			if error is not None: print(error); sys.exit(0)

	# CALCULATE mAP
	if 'MAP' in args.TODO:
		EPOCH_f('val', model, train_dl, evaluate_flag = True) #'val' així no calcular el gradient
		EPOCH_f('val', model, val_dl, evaluate_flag = True)
		EPOCH_f('val', model, test_dl, evaluate_flag = True)

	print("Model=", mymodel.mod, "loss=", mymodel.loss, "embdim=", mymodel.embdim)

	# CALCUALTE Z_MEAN
	if 'Z_MEAN' in args.TODO:
		for t in args.z_type_zdata:
			print("Calculating z_mean_{} ...".format(t))
			error = myzmean.calculate(model, meta_inp_out=t, force_calculation=force_calculation)
			if error is not None: print(error); sys.exit(0)

	# PLOTTING Z_DATA
	if 'PLOTZ' in args.TODO:
		for t in args.z_type_zdata:
			for part in args.partition:
				myzplot.update_names()
				error = myzplot.plot_tags(["all"], tags_mean=[], model=model, N_users=args.N_users, meta_inp_out=t, legend=False, minN=1, partition=part)
				if error != None: print(error); sys.exit(0)
	if args.tags_zdata is not None and len(args.tags_zdata) != 0 and args.tags_separated_zdata == "n":
		for t in args.z_type_zdata:
			for part in args.partition:
				error = myzplot.plot_tags(args.tags_zdata, model=model, N_users=args.N_users, meta_inp_out=t, legend=False, minN=1, partition=part)
				if error != None: print(error); sys.exit(0)
	if args.tags_zdata is not None and len(args.tags_zdata) != 0 and args.tags_separated_zdata != "n":
		for t in args.z_type_zdata:
			for part in args.partition:
				error = myzplot.plot_tags(args.tags_zdata, separated=True, model=model, N_users=args.N_users, meta_inp_out=t, legend=True, partition=part)
				if error != None: print(error); sys.exit(0)
	print(" ")

	return


#################################################################################################3

if __name__ == '__main__':

	# GET ARGUMENTS
	INPUT_PARAMS = wrapper.PARAM_WRAPPER()
	args, error = INPUT_PARAMS.input_params()
	if error is not None: print(error); sys.exit(0)

	#translate
	if len(args.TODO) == 0: print("Select actions to perform (--TODO)"); print("TRAIN", "PLOTLOSS", 'MAP', 'Z_MEAN', 'PLOTZ'); sys.exit(0)
	if args.TODO == ['all']: args.TODO = ["TRAIN", "PLOTLOSS", 'MAP', 'Z_MEAN', 'PLOTZ']
	if ('TRAIN' in args.TODO) and (args.loss != 'dummy'): training = True
	else: training = False

	# SEED
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device=='cuda':
	    torch.backends.cudnn.deterministic=True
	    torch.backends.cudnn.benchmark=False
	    torch.cuda.manual_seed(args.seed)

	# LOAD MODEL
	print("Setting model...")
	mymodel = MODEL()
	mymodel.set_params(**INPUT_PARAMS.get_params("mymodel"))
	error = mymodel.update_names()
	if error is not None: print(error); sys.exit(0)
	if mymodel.betastart is not None: mymodel.beta = 0 # for bKLD_start
	if args.train_tun: mymodel.name += "_" + args.metadata_name

	print("Setting model parameters...")
	error = mymodel.load_loss()
	if error != None: print(error); sys.exit(0)

	model, error = mymodel.load_model()
	if error != None: print(error); sys.exit(0)

	# LOAD LOSS
	print("Setting myloss...")
	myloss = TRAINING_LOSS()
	myloss.set_params(**INPUT_PARAMS.get_params("training_loss"))
	myloss.set_params(mymodel=mymodel)
	error = myloss.update_names()
	if error is not None: print(error); sys.exit(0)

	# LOAD METADATA
	print("Setting mymetadata...")
	mymetadata = METADATA()
	mymetadata.set_params(**INPUT_PARAMS.get_params("mymetadata"))
	error = mymetadata.update_names()
	if error is not None: print(error); sys.exit(0)

	# LOAD Z_DATA
	print("Setting myzmean...")
	myzmean = Z_MEAN()
	myzmean.set_params(**INPUT_PARAMS.get_params("z_data"))
	myzmean.set_params(mymodel=mymodel, mymetadata=mymetadata)
	error = myzmean.update_names()
	if error is not None: print(error); sys.exit(0)

	print("Setting myzcluster...")
	myzcluster = Z_CLUSTER()
	myzcluster.set_params(**INPUT_PARAMS.get_params("z_data"))
	myzcluster.set_params(mymodel=mymodel, mymetadata=mymetadata)
	error = myzcluster.update_names()
	if error is not None: print(error); sys.exit(0)

	print("Setting myzplot...")
	myzplot = PLOT_Z_DATA()
	myzplot.set_params(**INPUT_PARAMS.get_params("z_data"))
	myzplot.set_params(mymodel=mymodel, mymetadata=mymetadata)
	error = myzplot.update_names()
	if error is not None: print(error); sys.exit(0)

	print("Setting dataset...")
	mydataset = DATASET()
	mydataset.set_params(**INPUT_PARAMS.get_params("mydataset"))
	myzmean.set_params(mydataset=mydataset)
	myzcluster.set_params(mydataset=mydataset)
	myzplot.set_params(mydataset=mydataset)

	print("\nMODEL NAME: ", mymodel.name)
	print("TRAINING+TUNING: ", args.train_tun)
	print("TODO: ", ", ".join(args.TODO), "\n")

	# NOT TRAINING
	if not training:
		if 'MAP' in args.TODO:
			print("Loading dataset...")

			train_ds = dataset.UserSet(mymetadata.userset, tsplit='train', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)
			val_ds = dataset.UserSet(mymetadata.userset, tsplit='val', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)
			test_ds = dataset.UserSet(mymetadata.userset, tsplit='test', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)

			train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
			val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
			test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

			# LOAD class2vector (one-hot vector) FOR LOSS 
			class2vector = torch.load(mymetadata.metadata + "2vector")

		POST_CALCULATIONS(model)
		sys.exit(0)

	# TRAINING
	torch.save(model.state_dict(), mymodel.param_path + "/" + mymodel.name)

	patience = args.patience
	restarts = args.restarts
	lr = args.lr
	best_loss = np.inf

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	print("Loading dataset...")

	train_ds = dataset.UserSet(mymetadata.userset, tsplit='train', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)
	val_ds = dataset.UserSet(mymetadata.userset, tsplit='val', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)
	test_ds = dataset.UserSet(mymetadata.userset, tsplit='test', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)

	train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
	val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
	test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

	# LOAD class2vector (one-hot vector) FOR LOSS 
	class2vector = torch.load(mymetadata.metadata + "2vector")

	for epoch in range(args.n_epochs):
		print("Model=", mymodel.mod, "loss=", mymodel.loss, "embdim=", mymodel.embdim, "name=", mymodel.name)
		print("Epoch {0:0d}/{1:0d}   lr={2:g}   p={3:d}".format(epoch+1, args.n_epochs, lr, patience))

		# set betaKLD
		if mymodel.betastart is not None and mymodel.betastart == epoch+1: 
			mymodel.beta = args.beta
			mymodel.update_loss()

		# TRAIN
		EPOCH_f('train', model, train_dl)

		# VALIDATION
		loss, plot = EPOCH_f('val', model, val_dl)
		myloss.add(plot, patience)

		# CHANGE LR
		if loss < best_loss:
			best_loss = loss
			patience = args.patience
			torch.save(model.state_dict(), mymodel.param_path + "/" + mymodel.name)
		else:
			patience -= 1
			if patience <= 0:
				restarts -= 1
				myloss.restart()
				if restarts < 0:
					print("Maximum number of restarts")
					break

				lr = lr*args.lr_factor

				for param_group in optimizer.param_groups:
					param_group['lr'] = lr

				patience = args.patience

	myloss.save()

	POST_CALCULATIONS(model, force_calculation=True)