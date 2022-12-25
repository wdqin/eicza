import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader

import pickle

def awe_prepare_data_none_ml(df,args,train_transform=None):
  def load_data(info_df):
    keys = df.columns.tolist()
    data_dict = {}
    for i,idx in enumerate(df['earID']):
      single = {}
      for key in keys:
        if key == 'subject':
          single[key] = info_df[key][i]-1
        else:
          single[key] = info_df[key][i]
      data_dict[idx] = single
    return data_dict

  def split_data(data_dict):
    split_set = {'train','val','test'}
    train_list = []
    val_list = []
    test_list = []
    for k,v in data_dict.items():
      assert v['split'] in split_set, "split not valid, got {}.".format(v['split'])
      if v['split'] == 'train':
        train_list.append(v)
      elif v['split'] == 'val':
        val_list.append(v)
      elif v['split'] == 'test':
        test_list.append(v)
      else:
        raise NotImplementedError
    return train_list,val_list,test_list

  data_dict = load_data(df)
  train_list,val_list,test_list = split_data(data_dict)
  return train_list,val_list,test_list

def awe_prepare_data_ml(df,args,train_transform=None):
  def load_data(info_df):
    keys = df.columns.tolist()
    data_dict = {}
    for i,idx in enumerate(df['earID']):
      single = {}
      for key in keys:
        if key == 'subject':
          single[key] = info_df[key][i]-1
        else:
          single[key] = info_df[key][i]
      data_dict[idx] = single
    return data_dict

  def split_data(data_dict):
    split_set = {'train','val','test'}
    train_list = []
    val_list = []
    test_list = []
    for k,v in data_dict.items():
      assert v['split'] in split_set, "split not valid, got {}.".format(v['split'])
      if v['split'] == 'train':
        train_list.append(v)
      elif v['split'] == 'val':
        val_list.append(v)
      elif v['split'] == 'test':
        test_list.append(v)
      else:
        raise NotImplementedError
    return train_list,val_list,test_list

  data_dict = load_data(df)
  train_list,val_list,test_list = split_data(data_dict)

  if(not train_transform):
    preprocess_train_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485,0.456,0.406], std = [0.229,0.224,0.225]),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(30),
    ])
  else:
    raise NotImplementedError

  preprocess_val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std = [0.229,0.224,0.225]),
    ])

  awe_dataset_train = awe_dataset(train_list,load_dataset_path = args.load_dataset_path, path = args.image_folder_path,transforms = preprocess_train_transform)
  awe_dataset_val = awe_dataset(val_list, load_dataset_path = args.load_dataset_path, path = args.image_folder_path,transforms = preprocess_val_transform)
  awe_dataset_test = awe_dataset(test_list, load_dataset_path = args.load_dataset_path, path = args.image_folder_path,transforms = preprocess_val_transform)

  awe_dataloader_train = DataLoader(awe_dataset_train, batch_size=args.batch_size, shuffle=True)
  awe_dataloader_val = DataLoader(awe_dataset_val, batch_size=args.batch_size, shuffle=True)
  awe_dataloader_test = DataLoader(awe_dataset_test, batch_size=args.batch_size, shuffle=True)

  return awe_dataset_train,awe_dataset_val,awe_dataset_test,awe_dataloader_train,awe_dataloader_val,awe_dataloader_test


class awe_dataset(Dataset):
    def __init__(self, data_list, load_dataset_path, path = "",transforms = None):

        self.data_list = data_list
        self.path = path
        self.transforms = transforms
        self.unique_ids = set()

        with open(load_dataset_path, "rb") as f: # "rb" because we want to read in binary mode
            image_dict = pickle.load(f)
        self.image_dict = image_dict

        for data in data_list:
            if data['subject'] not in self.unique_ids:
                self.unique_ids.add(data['subject'])



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
      
        ear_image_name = self.data_list[idx]['imagePath']

        # img = Image.open(ear_image_name).convert('RGB')
        
        # print("self.image_dict.keys()",self.image_dict.keys())

        img = self.image_dict[ear_image_name]['img_encoded']

        if self.transforms:
            img = self.transforms(img)


        i = self.data_list[idx]['earID']

        ear_x = self.data_list[idx]['x']
        ear_y = self.data_list[idx]['y']
        ear_leftOrRight = self.data_list[idx]['leftOrRight']

        ear_w = self.data_list[idx]['w']
        ear_h = self.data_list[idx]['h']

        ear_accessories = self.data_list[idx]['accessories']
        ear_overlap  = self.data_list[idx]['overlap']

        ear_hPitch  = self.data_list[idx]['hPitch']
        ear_hYaw  = self.data_list[idx]['hYaw']
        ear_hRoll = self.data_list[idx]['hRoll']

        ear_subject = self.data_list[idx]['subject']
        ear_split = self.data_list[idx]['split']

        data_dict = {
                    "idx": int(i),\
                    "ear_image": img,\
                    "x": ear_x,\
                    "y": ear_y,\
                    "leftOrRight": ear_leftOrRight,\
                    "w": ear_w,\
                    "h": ear_h,\
                    "accessories": ear_accessories,\
                    "overlap": ear_overlap,\
                    "hPitch": ear_hPitch,\
                    "hYaw": ear_hYaw,\
                    "hRoll": ear_hRoll,\
                    "ear_subject_idx": ear_subject,\
                    "split": ear_split
                    }

        return data_dict