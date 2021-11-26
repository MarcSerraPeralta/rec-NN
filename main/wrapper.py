import argparse
import numpy as np

##############################################################################################################

class PARAM_WRAPPER():
	"""
	Manages all params for the myclasses and others. 
	input_params(): loads the params from argparse
	default_params(class_name): returns the default params for the class
	get_params(class_name): return the current params for the class

	returns variable, error_msg
	if error_msg is None, task completed succesfully. 
	"""
	def __init__(self):

		self.get_default_params(None)
		self.args = None
		self.args_dict = None

		return 

	def input_params(self):

		parser = argparse.ArgumentParser()
		# 0) GENERAL
		parser.add_argument('--device', type=str, default=self.default_params['mymodel']['device'], help="Type of device")

		# 1) DATA 
		parser.add_argument('--Nsongs', type=int, default=self.default_params['mymodel']['Nsongs'], help="Number of different songs")

		# 		1.1) PCOUNTS
		parser.add_argument('--userset_name', type=str, default=self.default_params['mymetadata']['userset_name'], help="File name for playcounts data")
		parser.add_argument('--userset_path', type=str, default=self.default_params['mymetadata']['userset_path'], help="File path for playcounts data")
		parser.add_argument('--pc_split', type=float, default=self.default_params['mydataset']['pc_split'], help="pcounts percentage of val and test")

		# 		1.2) METADATA
		parser.add_argument('--metadata_name', type=str, default=self.default_params['mymetadata']['metadata_name'], help="Name of the metadata file")
		parser.add_argument('--metadata_path', type=str, default=self.default_params['mymetadata']['metadata_path'], help="File path for metadata")
		parser.add_argument('--Nmeta_classes', type=int, default=self.default_params['miscellaneous']['Nmeta_classes'], help="Number of different metadata classes")

		# 		1.3) POSTDATA (dataset for post calculations)
		parser.add_argument('--bias_top', type=int, default=self.default_params['mymetadata']['bias_top'], help="Minimum number of songs in user_topsongs to be taken in care")
		parser.add_argument('--bias_normal', type=int, default=self.default_params['mymetadata']['bias_normal'], help="Minimum number of songs in user_normalsongs to be taken in care")

		# 		1.4) DATALOADER
		parser.add_argument('--batch_size', type=int, default=self.default_params['miscellaneous']['batch_size'], help="Batch size in one iteration")
		parser.add_argument('--num_workers', type=int, default=self.default_params['miscellaneous']['num_workers'], help="Number of workers")
		parser.add_argument('--seed', type=int, default=self.default_params['mydataset']['seed'], help="Random seed for numpy and torch")


		# 2) MODEL
		parser.add_argument('--idim', type=int, default=self.default_params['mymodel']['idim'], help="Dimension of the user input of song's indexes")
		parser.add_argument('--param_path', type=str, default=self.default_params['mymodel']['param_path'], help="File path for models parameters")

		# 		2.1) STRUCTURE
		parser.add_argument('--dim', nargs='+', type=int, default=self.default_params['mymodel']['dim'], help="Dimensions for hidden layers")
		parser.add_argument('--mod', type=str, default=self.default_params['mymodel']['mod'], help="Name of the model")
		# flow
		parser.add_argument('--bias', type=str, default=self.default_params['mymodel']['bias'], help="Bias for embedding [y/n]")
		parser.add_argument('--embname', type=str, default=self.default_params['mymodel']['embname'], help="Embedding model file name (ex. models/emb200)")
		parser.add_argument('--blocksN', type=int, default=self.default_params['mymodel']['blocksN'], help="Number of blocks in flow")
		parser.add_argument('--reduction_emb', type=str, default=self.default_params['mymodel']['reduction_emb'], help="Reduction for embedding")


		# 3) OPTIMIZER
		# 		3.1) OPTIMIZER PARAMS
		parser.add_argument('--lr', type=float, default=self.default_params['mymodel']['lr'], help="Learning rate")
		parser.add_argument('--lr_factor', type=float, default=self.default_params['miscellaneous']['lr_factor'], help="Factor for LR if patience is exceeded")

		# 		3.2) LOSS
		parser.add_argument('--loss', type=str, default=self.default_params['mymodel']['loss'], help="Name of the loss function")
		# KLD
		parser.add_argument('--beta', type=float, default=self.default_params['mymodel']['beta'], help="Loss coeficient for KLD")
		parser.add_argument('--betastart', type=int, default=self.default_params['mymodel']['betastart'], help="Number of the epoch in which the KLD is added")

		# 		3.3) TRAINING
		parser.add_argument('--n_epochs', type=int, default=self.default_params['miscellaneous']['n_epochs'], help="Number of epoch in training (n_epochs = 0 ==> random baseline)")
		parser.add_argument('--patience', type=int, default=self.default_params['miscellaneous']['patience'], help="Number of epoch while test loss is increasing")
		parser.add_argument('--restarts', type=int, default=self.default_params['miscellaneous']['restarts'], help="Number of times patience is exceeded")


		# 4) ACTIONS
		# 		4.1) TODO
		parser.add_argument('--TODO', nargs='+', type=str, default=self.default_params['miscellaneous']['TODO'], help="Actions to perform")
		parser.add_argument('--train_model', type=str, default=self.default_params['miscellaneous']['train_model'], help="Part of the flow model to train")
		parser.add_argument('--train_tun', type=str, default=self.default_params['miscellaneous']['train_tun'], help="Train model using train+tuning")

		# 5) Z_DATA
		# 		5.1) CALCULATION PARAMS
		parser.add_argument('--z_type_zdata', nargs='+', type=str, default=self.default_params['z_data']['z_type_zdata'], help="z_data types to calculate ['inp', 'out']")
		parser.add_argument('--z_data_name', type=str, default=self.default_params['miscellaneous']['z_data_name'], help="Full path for z_data")
		parser.add_argument('--N_users', type=int, default=self.default_params['z_data']['N_users'], help="Number of users used in z_data calculations")
		parser.add_argument('--Nclusters', type=int, default=self.default_params['z_data']['Nclusters'], help="Number of clusters used in z_data calculations")
		# 		5.2) PLOTING PARAMS
		parser.add_argument('--partition', nargs='+', type=str, default=self.default_params['miscellaneous']['partition'], help="Plot z partition ('test', 'train', 'val')")
		parser.add_argument('--tags_zdata', nargs='+', type=str, default=self.default_params['miscellaneous']['tags_zdata'], help="tags to plot")
		parser.add_argument('--tags_separated_zdata', type=str, default=self.default_params['miscellaneous']['tags_separated_zdata'], help="tags in different plots [y/n]")
		parser.add_argument('--topNtag', type=int, default=self.default_params['miscellaneous']['topNtag'], help="TopN tags for each cluster")


		# 6) TUNNING
		parser.add_argument('--minNclass', type=int, default=self.default_params["miscellaneous"]['minNclass'], help="Minim N for each class in to do post calculations")
		# 		6.1) PARAMS
		parser.add_argument('--z_type_tun', nargs='+', type=str, default=self.default_params["myrecom"]['z_type_tun'], help="z_data types to use in tunning ['inp', 'out']")
		parser.add_argument('--recomtype', type=str, nargs='+', default=self.default_params["myrecom"]['recomtype'], help="Type of recommendation used [all]")
		parser.add_argument('--alpha', type=float, nargs='+', default=self.default_params["myrecom"]['alpha'], help="Coeficient for tunning [z' = z+alpha*...] (2 for range)")
		parser.add_argument('--reli', type=float, nargs='+', default=self.default_params["myrecom"]['reli'], help="reli coeficient for NDCG ([1, reli, 0])")
		parser.add_argument('--topN', type=int, default=self.default_params["myrecom"]['topN'], help="Number of recommended songs")
		parser.add_argument('--alphaN', type=int, default=self.default_params["myrecom"]['alphaN'], help="Split alpha range in N elements")
		parser.add_argument('--reliN', type=int, default=self.default_params["myrecom"]['reliN'], help="Split reli range in N elements")
		parser.add_argument('--tunpost_factor', type=float, default=self.default_params['miscellaneous']['tunpost_factor'], help="Relation between alpa for tunning and postfiltering for tun+post recomtype")
		parser.add_argument('--alpha_post_sat', type=float, default=self.default_params['miscellaneous']['alpha_post_sat'], help="Value of alpha for NDCG postfiltering saturation")
		parser.add_argument('--NDCG_post_sat', type=float, default=self.default_params['miscellaneous']['NDCG_post_sat'], help="Value of NDCG postfiltering saturation")


		# 7) NDCG
		parser.add_argument('--legend_NDCG', type=str, default=self.default_params['miscellaneous']['legend_NDCG'], help="Plot legend in NDCG")
		parser.add_argument('--class_ave_NDCG', type=str, default=self.default_params['miscellaneous']['class_ave_NDCG'], help="Plot class average of NDCG")


		args = parser.parse_args()

		# TRANSLATE AND CHECKS
		if args.n_epochs == 0: args.loss = "dummy"

		if args.bias == "n": args.bias = False
		else: args.bias = True

		if args.legend_NDCG == "y": args.legend_NDCG = True
		else: args.legend_NDCG = False

		if args.alpha is not None:
			if len(args.alpha) == 1:
				pass
			elif len(args.alpha) == 2:
				args.alpha = np.linspace(args.alpha[0], args.alpha[1], args.alphaN)
			else:
				return None, "ERROR (PARAM_WRAPPER.input_params): --alpha must have len=1 or 2"

		if args.reli is not None:
			if len(args.reli) == 1:
				pass
			elif len(args.reli) == 2:
				args.reli = np.linspace(args.reli[0], args.reli[1], args.reliN)
			else:
				return None, "ERROR (PARAM_WRAPPER.input_params): --reli must have len=1 or 2"

		if args.mod=="flow" and args.embname is None: return None, "ERROR (PARAM_WRAPPER.input_params): Specify name for flow projector (--embname)"

		if args.z_type_tun == "" and args.recomtype != ["postfiltering"]: return "ERROR (PARAM_WRAPPER.input_params): Specify a z_data type to be used in tunning"

		if args.Nmeta_classes == -1:
			if args.metadata_name == "opt_tags": args.Nmeta_classes = 1000
			if args.metadata_name == "opt_tags_filtered": args.Nmeta_classes = 59
			if args.metadata_name == "opt_genre": args.Nmeta_classes = 21

		if args.train_tun == "y": args.train_tun = True
		else: args.train_tun = False

		self.args_dict = vars(args) # converts args.Namespace() to dict
		self.args = args

		return args, None

	def get_default_params(self, class_name):

		default_params = {}
		default_params["mymodel"] = {
				'idim':100,
				'Nsongs':180198,
				'device':'cuda',
				'loss':None,
				'dim':None,
				'mod':None,
				'beta':None,
				'betastart':None,
				'bias':True,
				'embname':None,
				'blocksN':4,
				'hN':4,
				'reduction_emb':'attention',
				'lr':1E-3,
				'model_path':'main/models',
				'param_path':'results/models',
				'Nmeta_classes':-1
		}
		default_params["mymetadata"] = {
				'userset_name':'opt_pcounts',
				'userset_path':'results/metadata',
				'metadata_name':'opt_tags',
				'metadata_path':'results/metadata',
				'metadata_type':'list',
				'postset_name':'postset_opt_tags',
				'postset_path':'results/metadata',
				'bias_top':10,
				'bias_normal':10,
				'z_data_path':'results/z_data'
		}
		default_params["mydataset"] = {
				'pc_split':0.1, #playcounts split for test and val
				'bias_top':1,
				'bias_normal':1,
				'seed':0
		}
		default_params["training_loss"] = {
				'plot_path':'results/loss',
				'mymodel':None
		}
		default_params["z_data"] = {
				'mymodel':None,
				'mymetadata':None,
				'mydataset':None,
				'Nclusters':10,
				'N_users':-1,
				'z_type_zdata':['out'],
				'z_data_name':None
		}
		default_params["myrecom"] = {
				'z_type_tun':['out'],
				'recomtype':[],
				'alpha':[1.],
				'reli':[0.5],
				'topN':1000,
				'alphaN':10,
				'reliN':10,
				'z_data_type':'z_cluster'
		}
		default_params["miscellaneous"] = {
				'batch_size':128,
				'num_workers':4,
				'lr_factor':0.1,
				'n_epochs':999,
				'patience':10,
				'restarts':2,
				'TODO':[],
				'partition':['train'],
				'tags_zdata':[],
				'tags_separated_zdata':False,
				'legend_NDCG':True,
				'class_ave_NDCG':True,
				'z_data_name':None,
				'topNtag':100,
				'minNclass':1,
				'tunpost_factor':None,
				'alpha_post_sat':None,
				'NDCG_post_sat':None,
				'train_model':"",
				'Nmeta_classes':-1,
				'train_tun':"y"
		}

		self.default_params = default_params

		if class_name is None: return
		else: return self.default_params[class_name]

	def get_params(self, class_name):

		param_names = self.default_params[class_name].keys()
		class_params = self.default_params[class_name].copy()
		for i,j in self.args_dict.items():
			if i in param_names: class_params[i] = j

		return class_params

##############################################################################################################