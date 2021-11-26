import torch
from torch.utils.data import DataLoader
import argparse
import sys, os
import numpy as np
import tqdm

import main.dataset as dataset
import main.wrapper as wrapper
from recommend import RECOMMENDER, RECOM_PARAMS
from main.myclasses import MODEL, METADATA
from main.z_data import Z_MEAN, Z_CLUSTER, PLOT_Z_DATA
from main.evaluation import NDCG_F, NDCG_CLASS

def PREDICT(data_dl, alpha, reli):

	NDCG_class = torch.zeros(len(class2idx)).to(recom.mymodel.device)
	COUNTS_class = torch.zeros(3, len(class2idx)).to(recom.mymodel.device)
	DIST_class = torch.zeros(2, len(class2idx)).to(recom.mymodel.device)
	LEN_class = torch.zeros(len(class2idx)).to(recom.mymodel.device) #counts number of time each class appears, to calculate means

	# CHECKS
	if args.NDCG_post_sat is not None and recom.myrecom.recomtype == 'postfiltering':
		output_epoch = {"LEN":LEN_class}
		if "NDCG" in args.TODO:
			out = {c:args.NDCG_post_sat for c in class2idx.keys()}
			output_epoch["NDCG"] = out
		if "COUNTS" in args.TODO:
			out = {c:COUNTS_class[:,class2idx[c]] for c in class2idx.keys()}
			output_epoch["COUNTS"] = out
		if "DIST" in args.TODO:
			out = {c:DIST_class[:,class2idx[c]] for c in class2idx.keys()}
			output_epoch["DIST"] = out
			return output_epoch

	if args.tunpost_factor is None and args.alpha_post_sat is not None: 
		if alpha != 0:
			tunpost_factor = args.alpha_post_sat / alpha
		else:
			tunpost_factor = 1.
	else:
		if args.tunpost_factor is None:
			tunpost_factor = 1.
		else:
			tunpost_factor = args.tunpost_factor


	# CALCULATIONS
	total = len(data_dl) 
	for inp, out, c in tqdm.tqdm(data_dl):
		if not class2consider[c]: continue #skips this user

		c_idx = class2idx[c]
		LEN_class[c_idx] += 1

		# GET YPRED/RECOMMENDATION
		error, prob = recom.recommend(inp, betas={c:1.}, alpha=alpha, PROB=True, factor=tunpost_factor) # tag[0] for dataloader(batch_size=1), type(r_songs)=tensor of probabilities
		if error != None: print(error); sys.exit(0)

		if "NDCG" in args.TODO:
			curr_reli = torch.zeros(recom.mymodel.Nsongs).to(recom.mymodel.device)
			curr_reli[meta2song[c]] = reli
			curr_reli[out] = 1
			curr_reli = curr_reli.view((1,recom.mymodel.Nsongs))

			NDCG = NDCG_F(prob.view((1,recom.mymodel.Nsongs)), curr_reli, reduce_mean=True, k=args.topN) #mean does not affect as it is one element
			NDCG_class[c_idx] += NDCG

		if "COUNTS" in args.TODO:
			relis, ones = 0, 0
			_,r_songs = torch.topk(prob, args.topN) # not need to do all Nsongs, only topN
			r_songs = r_songs.flatten().tolist()
			for song in r_songs:
				if recom.metadata[song] != -1:
					if song in out:
						ones += 1
					elif c in recom.metadata[song]: 
						relis += 1

			COUNTS = torch.tensor([args.topN - relis - ones, relis, ones]).to(recom.mymodel.device).float().view(3) # zeros, relis, ones
			COUNTS_class[:,c_idx] += COUNTS

		if "DIST" in args.TODO and recom.myrecom.recomtype == "tunning":
			DIST_inp_class = torch.dist(recom.tunned, recom.z_data[recom.meta2idx[c]]).view(1).data.tolist()
			DIST_inp_out = torch.dist(recom.tunned, recom.user_latent(out)).view(1).data.tolist()
			DIST_class[:,c_idx] += torch.tensor([DIST_inp_class, DIST_inp_out]).view(2).to(recom.mymodel.device)

	msg = ""
	output_epoch = {"LEN":LEN_class}
	if "NDCG" in args.TODO:
		NDCG_class /= LEN_class
		out = {c:NDCG_class[class2idx[c]] for c in class2idx.keys()}
		output_epoch["NDCG"] = out
		msg += "NDCG: {0:0.5f}".format(torch.tensor([i for i in out.values()]).mean().data.tolist())
	if "COUNTS" in args.TODO:
		COUNTS_class /= LEN_class
		out = {c:COUNTS_class[:,class2idx[c]] for c in class2idx.keys()}
		output_epoch["COUNTS"] = out
		msg += " COUNTS: {:0.1f} {:0.1f} {:0.1f}".format(*torch.tensor([i.data.tolist() for i in out.values()]).mean(0).data.tolist())
	if "DIST" in args.TODO:
		DIST_class /= LEN_class
		out = {c:DIST_class[:,class2idx[c]] for c in class2idx.keys()}
		output_epoch["DIST"] = out
		msg += " DIST: {:0.4f} {:0.4f}".format(*torch.tensor([i.data.tolist() for i in out.values()]).mean(0).data.tolist())

	print(msg)

	return output_epoch

#################################################################################################

def PLOTTING(myNDCG):
	myNDCG.get()

	if 'PLOT' in args.TODO:
		print("Plotting...")
		myNDCG.plot() #legend=args.legend
	if 'PLOT_AVE' in args.TODO: 
		print("Plotting average...")
		myNDCG.plot_average()

	return

#################################################################################################

if __name__ == '__main__':

	# GET ARGUMENTS
	INPUT_PARAMS = wrapper.PARAM_WRAPPER()
	args, error = INPUT_PARAMS.input_params()
	if error is not None: print(error); sys.exit(0)

	# translate
	if args.TODO == ["all"]: args.TODO = ["NDCG", "COUNTS", "DIST", 'PLOT', 'PLOT_AVE']
	if len(args.TODO) == 0: print("Specify actions TODO: \n", "NDCG", "COUNTS", "DIST", 'PLOT', 'PLOT_AVE'); sys.exit(0)
	# checks
	if args.recomtype is None: print("ERROR: Specify a recommendation type (--recomtype)"); sys.exit(0)
	if "all" in args.recomtype:
		args.recomtype = ['postfiltering', 'tunning', 'tun+post']

	# LOAD DATASET
	print("Loading dataset... \t\t", end="")
	postset_ds = dataset.PostSet(metadata_path=args.metadata_path, metadata_name=args.metadata_name, bias_top=args.bias_top, bias_normal=args.bias_normal, calculate=False)
	topclass2Ntopclass = torch.load(args.metadata_path + "/topclass2Ntopclass_{}_t{}_n{}".format(args.metadata_name, args.bias_top, args.bias_normal))
	class2consider = {}
	class_dif = []
	for topc, Nclass in topclass2Ntopclass.items():
		class2consider[topc] = (Nclass >= args.minNclass)
		if Nclass >= args.minNclass:
			class_dif += [topc]
	class2idx = {class_dif[i]:i for i in range(len(class_dif))}
	print("POSTSET: \t", postset_ds.path)

	# LOAD MODEL
	print("Setting mymodel... \t\t", end="")
	mymodel = MODEL()
	mymodel.set_params(**INPUT_PARAMS.get_params("mymodel"))
	error = mymodel.update_names()
	if error is not None: print(error); sys.exit(0)
	print("MODEL NAME: \t", mymodel.name)

	# LOAD METADATA
	print("Setting mymetadata... \t\t", end="")
	mymetadata = METADATA()
	mymetadata.set_params(**INPUT_PARAMS.get_params("mymetadata"))
	error = mymetadata.update_names()
	if error is not None: print(error); sys.exit(0)

	meta2song = torch.load(mymetadata.meta2song) # for reli
	print("MEDATADA: \t", mymetadata.metadata)

	# LOAD RECOMMENDER
	print("Setting recommender... \t\t", end="")
	myrecom = RECOM_PARAMS()
	myrecom.set_params(**INPUT_PARAMS.get_params("myrecom"))

	recom = RECOMMENDER()
	recom.set_params(mymodel=mymodel, mymetadata=mymetadata, myrecom=myrecom, meta_inp_out=args.z_type_tun, z_data_name=args.z_data_name)
	error = recom.load() #model and data
	if error != None: print(error); sys.exit(0)
	print("Ok")

	# LOAD NDCG
	print("Setting NDCG... \t\t", end="")
	myNDCG = NDCG_CLASS({"mymodel":mymodel, "Rtype":args.recomtype, "outpath":"results/NDCG", 
				"meta_name":args.metadata_name, "meta_path":args.metadata_path, 
				"bias_top":args.bias_top, "bias_normal":args.bias_normal, 
				"alpha":args.alpha, "reli":args.reli,
				"minNclass":args.minNclass, "topN":args.topN, "Z_TYPE":args.z_type_tun})
	print("NDCG: \t", myNDCG.outname)

	print("\nTODO: \t", *args.TODO, "\n")

	# NOT CALCULATIONS
	if len(set(["NDCG", "COUNTS", "DIST"]).intersection(args.TODO)) == 0:
		PLOTTING(myNDCG)
		sys.exit(0)

	# CALCULATE NDCG
	NDCG_all = {}
	for a in args.alpha:
		for t in args.recomtype:
			for r in args.reli:
				print("Recommendation params: alpha={0:0.3f}   reli={1:0.3f}".format(a, r))
				print("Recommendation type: " + t)
				recom.myrecom.recomtype = t
				NDCG_all[a, t, r] = PREDICT(postset_ds, a, r)

	myNDCG.save(NDCG_all)

	# PLOT NDCG
	PLOTTING(myNDCG)