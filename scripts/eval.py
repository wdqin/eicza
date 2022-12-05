from param import args
from utils import build_model
from model import earResnet18Model,earResnet50Model,earSqueezeNet10Model,earSqueezeNet11Model
from model import train_model,evaluate_ear_model
from datasets.icz_dataset import icz_dataset,prepare_data

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import random

# set up random seed for reproducibility

random.seed(10)
np.random.seed(0)
torch.manual_seed(42)


if __name__ == "__main__":
  # print(args.info_path)
  if args.dataset == 'icz':

    
    df = pd.read_csv(args.info_path)
    icz_dataset_train,icz_dataset_val,icz_dataset_test,icz_dataloader_train,icz_dataloader_val,icz_dataloader_test = prepare_data(df,args)

    icz_dataset_trainval = torch.utils.data.Subset(icz_dataset_train,list(range(200)))
    icz_dataloader_trainval = DataLoader(icz_dataset_trainval,batch_size = 32, shuffle=True)



    if '+' in args.eval_split:
      split_list = args.eval_split.split('+')
    else:
      split_list = [args.eval_split]
    
    dataloader_list = []
    split_list_ordered = []

    if 'test' in split_list:
      dataloader_list.append(icz_dataloader_test) 
      split_list_ordered.append('test')
    if 'val' in split_list:
      dataloader_list.append(icz_dataloader_val) 
      split_list_ordered.append('val')
    if 'train' in split_list:
      dataloader_list.append(icz_dataloader_train) 
      split_list_ordered.append('train')
    if 'train_val' in split_list:
      dataloader_list.append(icz_dataloader_trainval)
      split_list_ordered.append('train_val')
    # else:
    #   print("split {} not recognized.".format(args.eval_split))
    #   raise NotImplementedError


    unique_ids = set()
    
    unique_ids.update(icz_dataset_train.unique_ids)
    unique_ids.update(icz_dataset_val.unique_ids)
    unique_ids.update(icz_dataset_test.unique_ids)

    # init model
    model = build_model(args.model,unique_ids)

    model.load_state_dict(torch.load(args.path_model_eval))

    model.eval()

    assert len(dataloader_list)>0, "split(s) to evaluate should be more than 0"
    for i,dataloader in enumerate(dataloader_list):
      acc = evaluate_ear_model(dataloader,model,split_list_ordered[i])

  else:
    print("dataset {} not recognized.".format(args.dataset))
    raise NotImplementedError

  # print(test_list)