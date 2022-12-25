import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle

def icz_prepare_data_none_ml(df,args,train_transform=None):
  def load_data(info_df):
    keys = df.columns.tolist()
    data_dict = {}
    for i,idx in enumerate(df['idx']):
      single = {}
      for key in keys:
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

def icz_prepare_data_ml(df,args,train_transform=None):
  def load_data(info_df):
    keys = df.columns.tolist()
    data_dict = {}
    for i,idx in enumerate(df['idx']):
      single = {}
      for key in keys:
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

  icz_dataset_train = icz_dataset(train_list,load_dataset_path = args.load_dataset_path, path = args.image_folder_path,transforms = preprocess_train_transform)
  icz_dataset_val = icz_dataset(val_list,load_dataset_path = args.load_dataset_path,path = args.image_folder_path,transforms = preprocess_val_transform)
  icz_dataset_test = icz_dataset(test_list,load_dataset_path = args.load_dataset_path,path = args.image_folder_path,transforms = preprocess_val_transform)

  icz_dataloader_train = DataLoader(icz_dataset_train, batch_size=args.batch_size, shuffle=True)
  icz_dataloader_val = DataLoader(icz_dataset_val, batch_size=args.batch_size, shuffle=True)
  icz_dataloader_test = DataLoader(icz_dataset_test, batch_size=args.batch_size, shuffle=True)

  return icz_dataset_train,icz_dataset_val,icz_dataset_test,icz_dataloader_train,icz_dataloader_val,icz_dataloader_test


class icz_dataset(Dataset):
    def __init__(self, data_list, load_dataset_path, path = "",transforms = None):

        self.data_list = data_list
        self.path = path
        self.transforms = transforms
        self.unique_ids = set()

        with open(load_dataset_path, "rb") as f: # "rb" because we want to read in binary mode
            image_dict = pickle.load(f)
        self.image_dict = image_dict

        for data in data_list:
            if data['earSubjectIdx'] not in self.unique_ids:
                self.unique_ids.add(data['earSubjectIdx'])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        

        i = self.data_list[idx]['idx']
        split = self.data_list[idx]['split']
        ear_subject_idx = self.data_list[idx]['earSubjectIdx']
        ear_image_name = self.path + self.data_list[idx]['earImageName']
        # img = Image.open(ear_image_name)
        img = self.image_dict[ear_image_name]['img_encoded']
        
        if self.transforms:
            img = self.transforms(img)

        period = self.data_list[idx]['period']
        ear_subject_name = self.data_list[idx]['earSubjectName']
        ear_left_right = self.data_list[idx]['earLeftRight']
        ear_rotated  = bool(self.data_list[idx]['earRotated'])
        
        data_dict = {"idx": int(i),\
                    "split": split,
                    "ear_subject_idx":ear_subject_idx,
                    "ear_image":img,
                    "period":period,
                    "ear_subject_name":ear_subject_name,
                    "ear_left_right":ear_left_right,
                    "ear_rotated":ear_rotated
                    }


        return data_dict