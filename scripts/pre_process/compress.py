import h5py
import numpy as np
import os
from PIL import Image
import base64
import pickle


def collect_images_from_folder(dir,save_file,dataset):
	if dataset == 'icz' or dataset == 'fgnet':
		image_dict = {}

		files = os.listdir(dir)
		for file_name in files:
			if(file_name.endswith('.JPG') or file_name.endswith('.png') or file_name.endswith('.jpg')):
				img = Image.open(dir+file_name).convert('RGB')
				img_array = np.array(img)

				image_dict[dir+file_name] = {
				'file_name': file_name,
				'path': dir+file_name,
				'img_encoded': img_array,
				}
		print("len(image_dict)",len(image_dict))
		print("image_dict.keys()",image_dict.keys())
		raise NotImplementedError
		with open(save_file, "wb") as f: # "wb" because we want to write in binary mode
			pickle.dump(image_dict, f)
	elif dataset == 'awe':
		image_dict = {}
		for folder in os.listdir(dir):
			if (folder.isnumeric()):
				files = os.listdir(dir+folder)
				for file_name in files:
					if(file_name.endswith('.JPG') or file_name.endswith('.png') or file_name.endswith('.jpg')):
						img = Image.open(dir+folder+'/'+file_name).convert('RGB')
						img_array = np.array(img)

						image_dict[dir+folder+'/'+file_name] = {
						'file_name': folder+'/'+file_name,
						'path': dir+folder+'/'+file_name,
						'img_encoded': img_array,
						}

		print("len(image_dict)",len(image_dict))
		with open(save_file, "wb") as f: # "wb" because we want to write in binary mode
			pickle.dump(image_dict, f)
	else:
		raise NotImplementedError

collect_images_from_folder('./datasets/FGNET_resized/images/','./datasets/FGNET_resized/fgnet.bin',dataset='fgnet')



# print("pwd_list",pwd_list)
