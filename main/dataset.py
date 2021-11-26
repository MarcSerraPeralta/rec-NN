import torch
from torch.utils import data
import sys
from sklearn.utils import shuffle
import numpy as np
import argparse
import matplotlib.pyplot as plt

class UserSet(data.Dataset):
	def __init__(self, path, tsplit, idim=100, seed=0, Nsongs=180198, pc_split=0.1, tag2vector_path=""):
		"""
		path : str
			   path + fname of the user-playcounts list
			   the file has the index of the songs listened by each user
		idim : int
			   maximum number of songs per user in items
			   >95% of users have listened less than 100 songs
		tsplit : str
				 type of dataset: 'train', 'val', 'test'
		loss : str
			   Name of the loss function used
		seed : int
			   Seed used for the pcounts splitting
		Nsongs : int
				 Number of different songs
		pc_split : float
				   Percentage of the val and test set 
				   (pc_split=1 corresponds to 100%)
		"""

		# LOAD DATA
		self.path = path 
		self.pcounts = torch.load(self.path) #list
		self.tsplit = tsplit
		self.pc_split = pc_split
		self.idim = idim
		self.len = len(self.pcounts)
		self.index1 = int(self.len*(1 - 2*pc_split)) 
		self.index2 = int(self.len*(1 - pc_split))
		self.seed = seed
		self.Nsongs = Nsongs

		# SPLIT DATASET
		if self.tsplit == "train":
			self.pcounts = shuffle(self.pcounts, random_state=self.seed)[:self.index1]
			self.len = len(self.pcounts)
		elif self.tsplit == "val":
			self.pcounts = shuffle(self.pcounts, random_state=self.seed)[self.index1:self.index2]
			self.len = len(self.pcounts)
		elif self.tsplit == "test":
			self.pcounts = shuffle(self.pcounts, random_state=self.seed)[self.index2:]
			self.len = len(self.pcounts)
		else:
			print("ERROR: split options = 'train', 'val', 'test'. \n", self.tsplit)
			self.len = None
			self.pcounts = None
			sys.exit(0)
		return

	def __len__(self):
		return self.len

	def __getitem__(self, idx): #given an index of user, returns two vectors of the listenned songs
		user = shuffle(self.pcounts[idx])
		idx_inp = np.random.randint(1, min(len(user)-1, self.idim))
		idx_out = np.random.randint(idx_inp + 1, min(len(user) + 1, idx_inp + self.idim))

		#INP PER EMBEDDING (song ID and -1)
		inp = -torch.ones(self.idim, dtype=torch.long)
		inp[range(idx_inp)] = torch.LongTensor(user[:idx_inp])

		#OUT (one-hot vector)
		out = torch.zeros(self.Nsongs, dtype=torch.long)
		out[user[idx_inp:idx_out]] = torch.ones(len(user[idx_inp:idx_out]), dtype=torch.long)

		return inp, out

	def get_tags(self, Nusers=0, Ntags=1):
		return torch.randint(Ntags, (Nusers, 1)).squeeze(1)

class EmbSet(data.Dataset):
	def __init__(self, path, tsplit, idim=100, seed=0, Nsongs=180198, pc_split=0.1):
		"""
		See UserSet
		This dataset is for flows. 
		"""

		self.path = path 
		self.pcounts = torch.load(self.path) #list
		self.tsplit = tsplit
		self.pc_split = pc_split
		self.idim = idim
		self.len = len(self.pcounts)
		self.index1 = int(self.len*(1 - 2*pc_split)) 
		self.index2 = int(self.len*(1 - pc_split))
		self.seed = seed
		self.Nsongs = Nsongs

		# SPLIT DATASET
		if self.tsplit == "train":
			self.pcounts = shuffle(self.pcounts, random_state=self.seed)[:self.index1]
			self.len = len(self.pcounts)
		elif self.tsplit == "val":
			self.pcounts = shuffle(self.pcounts, random_state=self.seed)[self.index1:self.index2]
			self.len = len(self.pcounts)
		elif self.tsplit == "test":
			self.pcounts = shuffle(self.pcounts, random_state=self.seed)[self.index2:]
			self.len = len(self.pcounts)
		else:
			print("ERROR: split options = 'train', 'val', 'test'. \n", self.tsplit)
			self.len = None
			self.pcounts = None
			sys.exit(0)
		return

	def __len__(self):
		return self.len

	def __getitem__(self, idx): #given an index of user, returns two vectors of the listenned songs
		user = shuffle(self.pcounts[idx])
		idx_inp = np.random.randint(1, min(len(user)-1, self.idim))
		idx_out = np.random.randint(idx_inp + 1, min(len(user) + 1, idx_inp + self.idim))

		#INP 
		inp_idim = -torch.ones(self.idim, dtype=torch.long)
		inp_idim[range(idx_inp)] = torch.LongTensor(user[:idx_inp])

		inp_idx = torch.zeros(self.Nsongs, dtype=torch.long)
		inp_idx[user[:idx_inp]] = torch.ones(len(user[:idx_inp]), dtype=torch.long)

		#OUT 
		out_idim = -torch.ones(self.idim, dtype=torch.long)
		out_idim[range(idx_out - idx_inp)] = torch.LongTensor(user[idx_inp:idx_out])

		out_idx = torch.zeros(self.Nsongs, dtype=torch.long)
		out_idx[user[idx_inp:idx_out]] = torch.ones(len(user[idx_inp:idx_out]), dtype=torch.long)

		return inp_idim, inp_idx, out_idim, out_idx


class PostSet(data.Dataset):
	"""
	Loads dataset for predict created by get_PostSet(). 
	"""
	def __init__(self, calculate=False, metadata_path="results/metadata", metadata_name="opt_tags", bias_top=1, bias_normal=1):
		if calculate:
			get_TestSetPredict()
		self.data = torch.load(metadata_path + "/postset_{}_t{}_n{}".format(metadata_name, bias_top, bias_normal))
		self.len = len(self.data)
		self.path = metadata_path + "/postset_{}_t{}_n{}".format(metadata_name, bias_top, bias_normal)
		return

	def __len__(self):
		return self.len

	def __getitem__(self, idx): 
		return self.data[idx]


def get_PostSet(pcounts_name = "opt_pcounts", pcounts_path = "results/metadata", 
						pc_split=0.1, seed = 0, 
						metadata_name = "opt_tags", metadata_path = "results/metadata", 
						bias_top=1, bias_normal=1):
	"""
	ONLY VALID FOR METADATA THAT IS A LIST FOR EACH SONG
	"""

	# LOAD PCOUNTS AND METADATA
	pcounts = torch.load(pcounts_path + "/" + pcounts_name) #list
	index2 = int(len(pcounts)*(1 - pc_split))
	pcounts = shuffle(pcounts, random_state=seed)[index2:] # Test partition
	metadata, meta = torch.load(metadata_path + "/" + metadata_name)
	Nclasses = len(meta)
	meta2idx = {meta[i]:i for i in range(Nclasses)}
	idx2meta = {i:meta[i] for i in range(Nclasses)}

	# CHANGE METADATA
	print("Metadata2num and opt_pcounts to dict...")
	idx_metadata = {} # same as metadata but using the index of meta2idx
	for i in range(len(metadata)):
		if metadata[i] == -1:
			idx_metadata[i] = -1
		else:
			idx_metadata[i] = [meta2idx[m] for m in metadata[i]]
	dict_pcounts = {}
	for i in range(len(pcounts)):
		dict_pcounts[i] = pcounts[i]

	# USER META COUNT
	print("Before filtering users without metadata,", len(pcounts))
	user2class_counts = {}
	total = len(dict_pcounts)
	for b, user in enumerate(list(dict_pcounts.keys())):
		print("  {0:0.3f}% \r".format((b+1.)*100./total), end="")
		class_counts = torch.zeros(Nclasses)
		for song in dict_pcounts[user]:
			if idx_metadata[song] != -1:
				class_counts[idx_metadata[song]] += 1
		if (class_counts != 0).any(): 
			user2class_counts[user] = class_counts.data.tolist()
		else:
			del dict_pcounts[user]

	# GET TOP CLASS
	print("After filtering users without metadata,", len(user2class_counts), len(dict_pcounts))
	user2topclass = {}
	for user in user2class_counts.keys():
		user2topclass[user] = idx2meta[torch.argmax(torch.tensor(user2class_counts[user])).data.tolist()]

	# SPLIT INTO [SONGS, TOP CLASS SONGS, TOP TAG]
	user2topsongs = {}
	user2normalsongs = {}
	total = len(dict_pcounts)
	for b, user in enumerate(dict_pcounts.keys()):
		print("  {0:0.3f}%\r".format((b+1.)/total*100), end="")
		top = []
		normal = []
		Ntop = 0
		for song in dict_pcounts[user]:
			if metadata[song] != -1:
				if (user2topclass[user] in metadata[song]) and Ntop<100:
					top += [song]
					Ntop += 1
				else:
					normal += [song]
			else:
				normal += [song]
		user2topsongs[user] = top
		user2normalsongs[user] = normal

	# DELETE USERS (BIAS_TOP, BIAS_NORMAL)
	predict_dataset = []
	for b, user in enumerate(dict_pcounts.keys()):
		print("  {0:0.3f}%\r".format((b+1.)/total*100), end="")
		if len(user2topsongs[user]) >= bias_top and len(user2normalsongs[user]) >= bias_normal:
			predict_dataset += [[user2normalsongs[user], user2topsongs[user], user2topclass[user]]]

	print("# Users (after deleting top<{}, inp<{}): ".format(bias_top, bias_normal), len(predict_dataset))

	torch.save(predict_dataset, metadata_path + "/postset_{}_t{}_n{}".format(metadata_name, bias_top, bias_normal))

	return

def get_topclass2Ntopclass(bias_top=1, bias_normal=1, metadata_path="results/metadata", metadata_name="opt_tags"):
	print("Calculating topclass2Ntopclass...")
	PostSet = torch.load(metadata_path + "/postset_{}_t{}_n{}".format(metadata_name, bias_top, bias_normal))
	topclass2Ntopclass = {}
	for b, (inp, out, c) in enumerate(PostSet):
		if c not in list(topclass2Ntopclass.keys()): topclass2Ntopclass[c] = 0
		topclass2Ntopclass[c] += 1

	torch.save(topclass2Ntopclass, metadata_path + "/topclass2Ntopclass_{}_t{}_n{}".format(metadata_name, bias_top, bias_normal)) 

	return

def get_class2song(metadata_path="results/metadata", metadata_name="opt_tags"):
	print("Calculating class2song...")
	metadata, meta = torch.load(metadata_path + "/" + metadata_name)
	class2song = {c:[] for c in meta}
	total = len(metadata)
	for i in range(total):
		print("  {0:0.3f}%\r".format((i+1.)/total*100), end="")
		if metadata[i] == -1: continue
		for c in metadata[i]:
			class2song[c] += [i]

	torch.save(class2song, metadata_path + "/{}2song".format(metadata_name))

	return

def get_class2vector(metadata_path="results/metadata", metadata_name="opt_tags", Nsongs=180198):
	print("Calculating get_class2vector...")
	class2song = torch.load(metadata_path + "/{}2song".format(metadata_name))
	_, meta = torch.load(metadata_path + "/" + metadata_name) # for idx2meta
	Nclasses = len(meta)
	meta2idx = {meta[i]:i for i in range(Nclasses)}
	idx2meta = {i:meta[i] for i in range(Nclasses)}
	total = len(class2song)

	class2vector = torch.zeros(total,Nsongs).long() 
	for i in range(total):
		print("  {0:0.3f}%\r".format((i+1.)/total*100), end="")
		class2vector[i][class2song[idx2meta[i]]] = 1

	torch.save(class2vector, metadata_path + "/{}2vector".format(metadata_name))

	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--bias_top', type=int, default=1, help="Minimum number of songs in user_topsongs to be taken in care")
	parser.add_argument('--bias_normal', type=int, default=1, help="Minimum number of songs in user_normalsongs to be taken in care")
	parser.add_argument('--Nsongs', type=int, default=180198, help="Number of different songs")

	parser.add_argument('--metadata_name', type=str, default="opt_tags", help="Name of the metadata to use")
	parser.add_argument('--metadata_path', type=str, default="results/metadata", help="Path of the metadata to use")
	parser.add_argument('--pcounts_name', type=str, default="opt_pcounts", help="Name of the pcounts to use")
	parser.add_argument('--pcounts_path', type=str, default="results/metadata", help="Path of the pcounts to use")

	parser.add_argument('--TODO', nargs='+', type=str, default=["all"], help="Things to calculate")

	args = parser.parse_args()
	if args.TODO == ["all"]: args.TODO = ["postset", "topclass2Ntopclass", "class2song", "class2vector"]

	print("METADATA: {}\nBIAS TOP: {}\nBIAS NORMAL: {}\n".format(args.metadata_name, args.bias_top, args.bias_normal))

	if "postset" in args.TODO:
		get_PostSet(bias_normal=args.bias_normal, bias_top=args.bias_top, metadata_name=args.metadata_name, metadata_path=args.metadata_path, pcounts_name=args.pcounts_name, pcounts_path=args.pcounts_path)
	if "topclass2Ntopclass" in args.TODO:
		get_topclass2Ntopclass(bias_normal=args.bias_normal, bias_top=args.bias_top, metadata_name=args.metadata_name, metadata_path=args.metadata_path)
	if "class2song" in args.TODO:
		get_class2song(metadata_name=args.metadata_name, metadata_path=args.metadata_path)
	if "class2vector" in args.TODO:
		get_class2vector(metadata_name=args.metadata_name, metadata_path=args.metadata_path, Nsongs=args.Nsongs)
