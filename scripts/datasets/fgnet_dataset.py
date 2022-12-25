import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader

import pickle

def fgnet_prepare_data_none_ml(df,args,train_transform=None):
  def load_data(info_df):
    keys = df.columns.tolist()
    data_dict = {}
    for i,idx in enumerate(df['idx']):
      single = {}
      for key in keys:
        if key == 'personID':
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

def fgnet_prepare_data_ml(df,args,train_transform=None):
  def load_data(info_df):
    keys = df.columns.tolist()
    data_dict = {}
    for i,idx in enumerate(df['idx']):
      single = {}
      for key in keys:
        if key == 'personID':
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

  fgnet_dataset_train = fgnet_dataset(train_list,load_dataset_path = args.load_dataset_path,path = args.image_folder_path,transforms = preprocess_train_transform)
  fgnet_dataset_val = fgnet_dataset(val_list,load_dataset_path = args.load_dataset_path, path = args.image_folder_path,transforms = preprocess_val_transform)
  fgnet_dataset_test = fgnet_dataset(test_list,load_dataset_path = args.load_dataset_path, path = args.image_folder_path,transforms = preprocess_val_transform)

  fgnet_dataloader_train = DataLoader(fgnet_dataset_train, batch_size=args.batch_size, shuffle=True)
  fgnet_dataloader_val = DataLoader(fgnet_dataset_val, batch_size=args.batch_size, shuffle=True)
  fgnet_dataloader_test = DataLoader(fgnet_dataset_test, batch_size=args.batch_size, shuffle=True)

  return fgnet_dataset_train,fgnet_dataset_val,fgnet_dataset_test,fgnet_dataloader_train,fgnet_dataloader_val,fgnet_dataloader_test


class fgnet_dataset(Dataset):
    def __init__(self, data_list, load_dataset_path, path = "",transforms = None):

        self.data_list = data_list
        self.path = path
        self.transforms = transforms
        self.unique_ids = set()

        with open(load_dataset_path, "rb") as f: # "rb" because we want to read in binary mode
            image_dict = pickle.load(f)
        self.image_dict = image_dict

        for data in data_list:
            if data['personID'] not in self.unique_ids:
                self.unique_ids.add(data['personID'])



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
      
        ear_image_name = self.path + self.data_list[idx]['jpgName']
        # img = Image.open(ear_image_name).convert('RGB')
        img = self.image_dict[ear_image_name]['img_encoded']

        if self.transforms:
            img = self.transforms(img)


        i = self.data_list[idx]['idx']

        ear_person_age = self.data_list[idx]['personAge']
        ear_subject = self.data_list[idx]['personID']
        ear_split = self.data_list[idx]['split']

        data_dict = {
                    "idx": int(i),\
                    "ear_image": img,\
                    "ear_person_age": ear_person_age,\
                    "ear_subject_idx": ear_subject,\
                    "split": ear_split
                    }

        return data_dict