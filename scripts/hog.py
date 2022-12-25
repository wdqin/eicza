from utils import getEarImageListByEarSubjects
import skimage.io as io
from skimage.feature import hog
from tqdm import tqdm
import numpy as np
import pickle
from PIL import Image


def compute_hog(ear_subject_dict,load_dataset_path):
	hog_descriptor_dict = {}
	with open(load_dataset_path, "rb") as f: # "rb" because we want to read in binary mode
		image_dict = pickle.load(f)
	for k,v in tqdm(ear_subject_dict.items()):
		for image_name in v:

			# image = io.imread(image_name)
			image = image_dict[image_name]['img_encoded']
			feature_vector,hog_image = hog(image, orientations=5, pixels_per_cell=(16, 16),\
				cells_per_block=(7, 7), visualize=True, feature_vector = True,multichannel = True)

			# assert image_name not in hog_descriptor_dict, print("the same file {} has two feature vectors".format(image_name))
			# hog_descriptor_dict[image_name] = feature_vector
			if k not in hog_descriptor_dict:
				hog_descriptor_dict[k] = [feature_vector]
			else:
				hog_descriptor_dict[k].append(feature_vector)
	# print("hog_descriptor_dict",hog_descriptor_dict)
	return hog_descriptor_dict


class earHog:
	def __init__(self,ear_list,args):
		self.ear_subject_dict = getEarImageListByEarSubjects(ear_list,args.image_folder_path,args.dataset)
		self.hog_descriptor_dict = compute_hog(self.ear_subject_dict,args.load_dataset_path)
	def make_data(self):
		label_list = []
		all_feature_list = []
		for k,v in self.hog_descriptor_dict.items():
			for feature in v:
				all_feature_list.append(feature)
				label_list.append(int(k))
		input_matrix = np.stack(all_feature_list)
		label_vector = np.stack(label_list)
		return input_matrix,label_vector

	# def compute_hog_features(self):
	# 	self.hog_descriptor_dict = compute_hog()


