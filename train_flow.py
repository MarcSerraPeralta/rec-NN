import torch
from torch.utils.data import DataLoader
import argparse
import sys, os
import numpy as np

import main.dataset as dataset
import main.wrapper as wrapper
from main.myclasses import MODEL, METADATA, DATASET
from main.utils import TRAINING_LOSS
from main.evaluation import average_precision
from main.z_data import Z_MEAN, Z_CLUSTER, PLOT_Z_DATA


def EPOCH_f_emb(t_v, model_emb, data_dl, mAP_flag = False):

	if t_v == 'train':
		model_emb.train()
	else:
		model_emb.eval()

	dat_AP = torch.tensor([]).to(args.device)
	total = len(data_dl) 
	running_loss = 0
	plot_loss = 0

	for b,(inp_idim, inp_idx, out_idim, out_idx) in enumerate(data_dl):
		inp_idim, inp_idx, out_idim, out_idx = inp_idim.to(args.device), inp_idx.to(args.device), out_idim.to(args.device), out_idx.to(args.device)

		if t_v == 'train':
			optimizer.zero_grad()
			xnew, pars = model_emb.forward(inp_idim)
			loss, plot = embmodel.loss_f(inp_idx, xnew, pars)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			plot_loss += plot

			optimizer.zero_grad()
			ynew, pars = model_emb.forward(out_idim)
			loss, plot = embmodel.loss_f(out_idx, ynew, pars)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			plot_loss += plot

		else:
			with torch.no_grad():
				xnew, pars = model_emb.forward(inp_idim)
				loss, plot = embmodel.loss_f(inp_idx, xnew, pars)
				running_loss += loss.item()
				plot_loss += plot

				ynew, pars = model_emb.forward(out_idim)
				loss, plot = embmodel.loss_f(out_idx, ynew, pars)
				running_loss += loss.item()
				plot_loss += plot

		# loss
		aveloss = running_loss / (2*b+2)
		ave_plot = plot_loss / (2*b+2)
		print("\r      {0:7.3f}%   loss = ".format((b+1)/total*100) + " / ".join(["{0:10.6f}".format(i) for i in ave_plot]) + " / {0:10.6f}".format(aveloss), end="")

		# mAP
		if mAP_flag:
			inp_idx = inp_idx.float()
			xnew = xnew.float()
			out_idx = out_idx.float()
			ynew = ynew.float()

			AP = average_precision(inp_idx, xnew, reduce_mean=False, k=1000) #y_true, y_pred
			dat_AP = torch.cat((dat_AP, AP))
			AP = average_precision(out_idx, ynew, reduce_mean=False, k=1000) #y_true, y_pred
			dat_AP = torch.cat((dat_AP, AP))

			mAP = torch.mean(dat_AP)
			print("\r      {0:7.3f}%   loss = ".format((b+1)/total*100) + " / ".join(["{0:0.6f}".format(i) for i in ave_plot]) + "   mAP = {0:0.7f}     ".format(mAP), end="")

	print(" ")

	if t_v == 'val' and not mAP_flag:
		return aveloss, (plot_loss/(2*b+2)).data.tolist()
	
	return


def EPOCH_f_flow(t_v, model_flow, data_dl, mAP_flag = False):
	
	if t_v == 'train':
		model_flow.train()
	else:
		model_flow.eval()

	dat_AP = torch.tensor([]).to(args.device)
	total = len(data_dl) 
	running_loss = 0
	plot_loss = 0

	for b,(inp_idim, inp_idx, out_idim, out_idx) in enumerate(data_dl):
		inp_idim, inp_idx, out_idim, out_idx = inp_idim.to(args.device), inp_idx.to(args.device), out_idim.to(args.device), out_idx.to(args.device)

		if t_v == 'train':
			optimizer.zero_grad()
			xnew, pars = model_emb.encoder(inp_idim)
			h_inp, logdet = model_flow.forward(xnew.detach())
			loss, plot = flowmodel.loss_f(inp_idx, h_inp, logdet)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			plot_loss += plot

			optimizer.zero_grad()
			ynew, pars = model_emb.encoder(out_idim)
			h_out, logdet = model_flow.forward(ynew.detach())
			loss, plot = flowmodel.loss_f(out_idx, h_out, logdet)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			plot_loss += plot

		else:
			with torch.no_grad():
				xnew, pars = model_emb.encoder(inp_idim)
				h_inp, logdet = model_flow.forward(xnew.detach())
				loss, plot = flowmodel.loss_f(inp_idx, h_inp, logdet)
				running_loss += loss.item()
				plot_loss += plot

				ynew, pars = model_emb.encoder(out_idim)
				h_out, logdet = model_flow.forward(ynew.detach())
				loss, plot = flowmodel.loss_f(out_idx, h_out, logdet)
				running_loss += loss.item()
				plot_loss += plot

		# loss
		aveloss = running_loss / (2*b+2)
		ave_plot = plot_loss / (2*b+2)
		print("\r      {0:7.3f}%   loss = ".format((b+1)/total*100) + " / ".join(["{0:10.6f}".format(i) for i in ave_plot]) + " / {0:10.6f}".format(aveloss), end="")

		if mAP_flag:
			inp_idx = inp_idx.float()
			xnew = model_emb.decoder(model_flow.decoder(h_inp))
			out_idx = out_idx.float()
			ynew = model_emb.decoder(model_flow.decoder(h_out))

			AP = average_precision(inp_idx, xnew, reduce_mean=False, k=1000) #y_true, y_pred
			dat_AP = torch.cat((dat_AP, AP))
			AP = average_precision(out_idx, ynew, reduce_mean=False, k=1000) #y_true, y_pred
			dat_AP = torch.cat((dat_AP, AP))

			mAP = torch.mean(dat_AP)
			print("\r      {0:7.3f}%   loss = ".format((b+1)/total*100) + " / ".join(["{0:0.6f}".format(i) for i in ave_plot]) + " / {0:10.6f}".format(aveloss) + "   mAP = {0:0.7f}     ".format(mAP), end="")


	print(" ")

	if t_v == 'val' and not mAP_flag:
		return aveloss, (plot_loss/(2*b+2)).data.tolist()
	
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
		if "emb" == args.train_model:
			EPOCH_f_emb('val', model, train_dl, mAP_flag = True)
			EPOCH_f_emb('val', model, val_dl, mAP_flag = True)
			EPOCH_f_emb('val', model, test_dl, mAP_flag = True)
		if "flow" == args.train_model:
			EPOCH_f_flow('val', model, train_dl, mAP_flag = True)
			EPOCH_f_flow('val', model, val_dl, mAP_flag = True)
			EPOCH_f_flow('val', model, test_dl, mAP_flag = True)
		if "MLP" == args.train_model:
			EPOCH_f_MLP('val', model, train_dl, mAP_flag = True)
			EPOCH_f_MLP('val', model, val_dl, mAP_flag = True)
			EPOCH_f_MLP('val', model, test_dl, mAP_flag = True)

	print("Model=", mymodel.mod, "loss=", mymodel.loss, "embdim=", mymodel.embdim)

	# CALCUALTE Z_MEAN
	if 'Z_MEAN' in args.TODO:
		for t in args.z_type_zdata:
			print("Calculating z_mean_{} ...".format(t))
			error = myzmean.calculate([model_emb, model_flow], meta_inp_out=t, force_calculation=force_calculation)
			if error is not None: print(error); sys.exit(0)

	# CALCUALTE Z_CLUSTER
	if 'Z_CLUSTER' in args.TODO:
		for t in args.z_type_zdata:
			print("Calculating z_cluster_{} ...".format(t))
			error = myzcluster.calculate(model, meta_inp_out=t, force_calculation=force_calculation)
			if error is not None: print(error); sys.exit(0)
			error = myzcluster.write_cluster2tags(topNtag=args.topNtag)
			if error is not None: print(error); sys.exit(0)
			error = myzcluster.plot_cluster2tags(topNtag=args.topNtag)
			if error is not None: print(error); sys.exit(0)


	# PLOTTING Z_DATA
	if 'PLOTZ' in args.TODO:
		for t in args.z_type_zdata:
			for part in args.partition:
				myzplot.update_names()
				error = myzplot.plot_tags(["all"], tags_mean=[], model=[model_emb, model_flow], N_users=args.N_users, meta_inp_out=t, legend=False, minN=1, partition=part)
				if error != None: print(error); sys.exit(0)
	if args.tags_zdata is not None and len(args.tags_zdata) != 0 and args.tags_separated_zdata == "n":
		for t in args.z_type_zdata:
			for part in args.partition:
				error = myzplot.plot_tags(args.tags_zdata, model=[model_emb, model_flow], N_users=args.N_users, meta_inp_out=t, legend=False, minN=1, partition=part)
				if error != None: print(error); sys.exit(0)
	if args.tags_zdata is not None and len(args.tags_zdata) != 0 and args.tags_separated_zdata != "n":
		for t in args.z_type_zdata:
			for part in args.partition:
				error = myzplot.plot_tags(args.tags_zdata, separated=True, model=[model_emb, model_flow], N_users=args.N_users, meta_inp_out=t, legend=True, partition=part)
				if error != None: print(error); sys.exit(0)
	print(" ")

	return


#################################################################################################3

if __name__ == '__main__':

	# GET ARGUMENTS
	INPUT_PARAMS = wrapper.PARAM_WRAPPER()
	args, error = INPUT_PARAMS.input_params()
	if error is not None: print(error); sys.exit(0)

	TODO_all = ["TRAIN", "PLOTLOSS", 'MAP', 'Z_MEAN', 'Z_CLUSTER', 'PLOTZ']

	#translate
	if len(args.TODO) == 0: print("Select actions to perform (--TODO)"); print(" ".join([str(i) for i in TODO_all])); sys.exit(0)
	if len(args.train_model) == 0: print("Select which part of the flow to train (--train_model [emb or flow or MLP])"); sys.exit(0)
	if "emb" == args.train_model:
		if args.TODO == ['all']: args.TODO = ["TRAIN", "PLOTLOSS", "MAP"]
	if "flow" == args.train_model:
		if args.TODO == ['all']: args.TODO = ["TRAIN", "PLOTLOSS", 'Z_MEAN', 'PLOTZ']
	if "MLP" == args.train_model:
		if args.TODO == ['all']: args.TODO = ["TRAIN", "PLOTLOSS", 'Z_MEAN', 'PLOTZ', 'MAP']
	if ('TRAIN' in args.TODO) and (args.loss != 'dummy'): training = True
	else: training = False

	# SEED
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device=='cuda':
		torch.backends.cudnn.deterministic=True
		torch.backends.cudnn.benchmark=False
		torch.cuda.manual_seed(args.seed)

	# LOAD METADATA
	print("Setting mymetadata...")
	mymetadata = METADATA()
	mymetadata.set_params(**INPUT_PARAMS.get_params("mymetadata"))
	error = mymetadata.update_names()
	if error is not None: print(error); sys.exit(0)

	# LOAD MODEL
	if args.train_model == 'emb':
		print("Setting model...")
		embmodel = MODEL() # model to train
		embmodel.set_params(**INPUT_PARAMS.get_params("mymodel"))
		error = embmodel.update_names()
		if error is not None: print(error); sys.exit(0)

		print("Setting model parameters...")
		error = embmodel.load_loss()
		if error != None: print(error); sys.exit(0)

		model_emb, error = embmodel.load_model()
		if error != None: print(error); sys.exit(0)

		if 'TRAIN' in args.TODO or 'MAP' in args.TODO:
			print("Loading dataset...")
			train_ds = dataset.EmbSet(mymetadata.userset, tsplit='train', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)
			val_ds = dataset.EmbSet(mymetadata.userset, tsplit='val', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)
			test_ds = dataset.EmbSet(mymetadata.userset, tsplit='test', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)

			train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
			val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
			test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

		print("\tEMBNAME: " + embmodel.name)
		mymodel = embmodel # for metadata, training_loss...


	elif args.train_model == 'flow':
		print("Setting model...")
		flowmodel = MODEL() # model to train
		flowmodel.set_params(**INPUT_PARAMS.get_params("mymodel"))
		error = flowmodel.update_names()
		if error is not None: print(error); sys.exit(0)

		embmodel = MODEL()
		embmodel.set_params(mod='MLPh0_emb', dim=args.dim, reduction_emb=args.reduction_emb, param_path=args.param_path)
		embmodel.name = args.embname

		print("Setting model parameters...")
		error = flowmodel.load_loss()
		if error != None: print(error); sys.exit(0)
		
		model_emb, error = embmodel.load_model_params()
		if error != None: print(error); sys.exit(0)
		model_flow, error = flowmodel.load_model()
		if error != None: print(error); sys.exit(0)

		if 'TRAIN' in args.TODO or 'MAP' in args.TODO:
			print("Loading dataset...")
			train_ds = dataset.EmbSet(mymetadata.userset, tsplit='train', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)
			val_ds = dataset.EmbSet(mymetadata.userset, tsplit='val', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)
			test_ds = dataset.EmbSet(mymetadata.userset, tsplit='test', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)

			train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
			val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
			test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

		print("\tEMBNAME: " + embmodel.name)
		print("\tFLOWNAME: " + flowmodel.name)
		mymodel = flowmodel # for metadata, training_loss...


	elif args.train_model == 'MLP':
		print("Setting model...")
		MLPmodel = MODEL() # model to train
		MLPmodel.set_params(mod='MLP_flow', dim=[args.dim], param_path=args.param_path, loss=args.loss, lr=args.lr)
		error = MLPmodel.update_names()
		if error is not None: print(error); sys.exit(0)

		embmodel = MODEL()
		embmodel.set_params(mod='MLP_emb', dim=args.dim, reduction_emb=args.reduction_emb, param_path=args.param_path)
		embmodel.name = args.embname

		flowmodel = MODEL()
		flowmodel.set_params(**INPUT_PARAMS.get_params("mymodel"))
		flowmodel.set_params(loss='FLOW')
		error = flowmodel.update_names() # to load flow
		if error is not None: print(error); sys.exit(0)

		print("Setting model parameters...")
		error = MLPmodel.load_loss()
		if error != None: print(error); sys.exit(0)
		
		model_emb, error = embmodel.load_model_params()
		if error != None: print(error); sys.exit(0)
		model_flow, error = flowmodel.load_model_params()
		if error != None: print(error); sys.exit(0)
		model_MLP, error = MLPmodel.load_model()
		if error != None: print(error); sys.exit(0)


		if 'TRAIN' in args.TODO or 'MAP' in args.TODO:
			print("Loading dataset...")
			train_ds = dataset.UserSet(mymetadata.userset, tsplit='train', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)
			val_ds = dataset.UserSet(mymetadata.userset, tsplit='val', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)
			test_ds = dataset.UserSet(mymetadata.userset, tsplit='test', idim=args.idim, seed=args.seed, Nsongs=args.Nsongs, pc_split=args.pc_split)

			train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
			val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
			test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

		print("\tEMBNAME: " + embmodel.name)
		print("\tFLOWNAME: " + flowmodel.name)
		print("\tMLPNAME: " + MLPmodel.name)
		mymodel = MLPmodel # for metadata, training_loss...

	else:
		print("ERROR: --train_model has to be 'emb' or 'flow' or 'MLP'")
		sys.exit(0)

	# LOAD LOSS
	print("Setting myloss...")
	myloss = TRAINING_LOSS()
	myloss.set_params(**INPUT_PARAMS.get_params("training_loss"))
	myloss.set_params(mymodel=mymodel)
	error = myloss.update_names()
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
	print("TODO: ", ", ".join(args.TODO), "\n")

	# NOT TRAINING
	if not training:
		if "emb" == args.train_model: 
			POST_CALCULATIONS(model_emb)
		if "flow" == args.train_model: 
			POST_CALCULATIONS(model_flow)
		if "MLP" == args.train_model: 
			POST_CALCULATIONS(model_MLP)
		sys.exit(0)

	# TRAINING
	# save model parameters
	if "emb" == args.train_model: 
		torch.save(model_emb.state_dict(), mymodel.param_path + "/" + mymodel.name)
	if "flow" == args.train_model: 
		torch.save(model_flow.state_dict(), mymodel.param_path + "/" + mymodel.name)
	if "MLP" == args.train_model: 
		torch.save(model_MLP.state_dict(), mymodel.param_path + "/" + mymodel.name)

	patience = args.patience
	restarts = args.restarts
	lr = args.lr
	best_loss = np.inf

	if "emb" == args.train_model:
		optimizer = torch.optim.Adam(model_emb.parameters(), lr=args.lr)
	if "flow" == args.train_model:
		optimizer = torch.optim.Adam(model_flow.parameters(), lr=args.lr)
	if "MLP" == args.train_model:
		optimizer = torch.optim.Adam(model_MLP.parameters(), lr=args.lr)

	for epoch in range(args.n_epochs):
		print("Model=", mymodel.mod, "loss=", mymodel.loss, "embdim=", mymodel.embdim, "name=", mymodel.name)
		print("Epoch {0:0d}/{1:0d}   lr={2:g}   p={3:d}".format(epoch+1, args.n_epochs, lr, patience))

		# TRAIN
		if "emb" == args.train_model:
			EPOCH_f_emb('train', model_emb, train_dl)
		if "flow" == args.train_model:
			EPOCH_f_flow('train', model_flow, train_dl)
		if "MLP" == args.train_model:
			EPOCH_f_MLP('train', model_MLP, train_dl)

		# VALIDATION
		if "emb" == args.train_model:
			loss, plot = EPOCH_f_emb('val', model_emb, val_dl)
			myloss.add(plot, patience)
		if "flow" == args.train_model:
			loss, plot = EPOCH_f_flow('val', model_flow, val_dl)
			myloss.add(plot, patience)
		if "MLP" == args.train_model:
			loss, plot = EPOCH_f_MLP('val', model_MLP, val_dl)
			myloss.add(plot, patience)

		# CHANGE LR
		if loss < best_loss:
			best_loss = loss
			patience = args.patience
			# save model parameters
			if "emb" == args.train_model: 
				torch.save(model_emb.state_dict(), mymodel.param_path + "/" + mymodel.name)
			if "flow" == args.train_model: 
				torch.save(model_flow.state_dict(), mymodel.param_path + "/" + mymodel.name)
			if "MLP" == args.train_model: 
				torch.save(model_MLP.state_dict(), mymodel.param_path + "/" + mymodel.name)
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

	if "emb" == args.train_model: 
		POST_CALCULATIONS(model_emb, force_calculation=True)
	if "flow" == args.train_model: 
		POST_CALCULATIONS(model_flow, force_calculation=True)
	if "MLP" == args.train_model: 
		POST_CALCULATIONS(model_MLP, force_calculation=True)