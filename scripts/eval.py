from param import args
from utils import build_model
from model import earResnet18Model,earResnet50Model,earSqueezeNet10Model,earSqueezeNet11Model
from model import train_model,evaluate_ear_model

from sift import siftEar
from hog import earHog
from svm import earSVM

from datasets.icz_dataset import icz_dataset,icz_prepare_data_ml,icz_prepare_data_none_ml
from datasets.awe_dataset import awe_dataset,awe_prepare_data_ml,awe_prepare_data_none_ml
from datasets.fgnet_dataset import fgnet_dataset,fgnet_prepare_data_ml,fgnet_prepare_data_none_ml

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
  print(args)
  if args.debug == 1:
    print("------------------------- ATTENTION: DEBUG MODE IS ON -------------------------")
    
  if args.dataset == 'icz':

    df = pd.read_csv(args.info_path)

    if args.model == "sift":

      if '+' in args.eval_split:
        split_list = args.eval_split.split('+')
      else:
        split_list = [args.eval_split]
      
      icz_train_list,icz_val_list,icz_test_list = icz_prepare_data_none_ml(df,args)
      for split in split_list:
        if split == 'val':
          sift_val = siftEar(icz_train_list,icz_val_list,args)
          ear_source_image_path_and_descriptor_dict_val,ear_target_image_path_and_descriptor_dict_val = sift_val.compute_descriptor()
          sift_val.evaluate(ear_source_image_path_and_descriptor_dict_val,ear_target_image_path_and_descriptor_dict_val,split)
        elif split == 'test':
          sift_test = siftEar(icz_train_list,icz_test_list,args)
          ear_source_image_path_and_descriptor_dict_test,ear_target_image_path_and_descriptor_dict_test = sift_test.compute_descriptor()
          sift_test.evaluate(ear_source_image_path_and_descriptor_dict_test,ear_target_image_path_and_descriptor_dict_test,split)
        else:
          print("{} dataset not found",split)
          raise NotImplementedError
    elif args.model == "hog":

      # hog + svm

      if '+' in args.eval_split:
        split_list = args.eval_split.split('+')
      else:
        split_list = [args.eval_split]
      icz_train_list,icz_val_list,icz_test_list = icz_prepare_data_none_ml(df,args)

      hog_train = earHog(icz_train_list,args)
      hog_val = earHog(icz_val_list,args)
      hog_test = earHog(icz_test_list,args)

      input_matrix_train,label_vector_train = hog_train.make_data()
      input_matrix_val,label_vector_val = hog_val.make_data()
      input_matrix_test,label_vector_test = hog_test.make_data()

      train_xy = (input_matrix_train,label_vector_train)
      val_xy = (input_matrix_val,label_vector_val)
      test_xy = (input_matrix_test,label_vector_test)

      if args.debug == 1: # debug mode
        svm_val = earSVM(train_xy = val_xy,test_xy = val_xy, args = args)
        svm_val.train()
        svm_val.evaluate("val")
      elif args.debug == 0: # normal mode
        svm_test = earSVM(train_xy = train_xy,test_xy = test_xy, args = args)
        if args.path_model_loaded == "": # train mode
          svm_test.train()
          svm_test.evaluate("test")
        else: # eval mode
          svm_test.load(args.path_model_loaded)
          svm_test.evaluate("test")
      else: # throw error
        print("debug mode {} not recognized, need to be 0 or 1.".format(args.debug))
        raise NotImplementedError
      

    else:
      icz_dataset_train,icz_dataset_val,icz_dataset_test,icz_dataloader_train,icz_dataloader_val,icz_dataloader_test = icz_prepare_data_ml(df,args)
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

      model = build_model(args.model,unique_ids,args.path_model_loaded)

      model.load_state_dict(torch.load(args.path_model_loaded))

      model.eval()

      assert len(dataloader_list)>0, "split(s) to evaluate should be more than 0"
      for i,dataloader in enumerate(dataloader_list):
        acc = evaluate_ear_model(dataloader,model,split_list_ordered[i])
  elif args.dataset == 'awe':
    df = pd.read_csv(args.info_path)
    
    if args.model == "sift":

      if '+' in args.eval_split:
        split_list = args.eval_split.split('+')
      else:
        split_list = [args.eval_split]
      
      awe_train_list,awe_val_list,awe_test_list = awe_prepare_data_none_ml(df,args)
      for split in split_list:
        if split == 'val':
          sift_val = siftEar(awe_train_list,awe_val_list,args)
          ear_source_image_path_and_descriptor_dict_val,ear_target_image_path_and_descriptor_dict_val = sift_val.compute_descriptor()
          sift_val.evaluate(ear_source_image_path_and_descriptor_dict_val,ear_target_image_path_and_descriptor_dict_val,split)
        elif split == 'test':
          sift_test = siftEar(awe_train_list,awe_test_list,args)
          ear_source_image_path_and_descriptor_dict_test,ear_target_image_path_and_descriptor_dict_test = sift_test.compute_descriptor()
          sift_test.evaluate(ear_source_image_path_and_descriptor_dict_test,ear_target_image_path_and_descriptor_dict_test,split)
        else:
          print("{} dataset not found",split)
          raise NotImplementedError
    elif args.model == "hog":

      # hog + svm

      if '+' in args.eval_split:
        split_list = args.eval_split.split('+')
      else:
        split_list = [args.eval_split]
      awe_train_list,awe_val_list,awe_test_list = awe_prepare_data_none_ml(df,args)

      hog_train = earHog(awe_train_list,args)
      hog_val = earHog(awe_val_list,args)
      hog_test = earHog(awe_test_list,args)

      input_matrix_train,label_vector_train = hog_train.make_data()
      input_matrix_val,label_vector_val = hog_val.make_data()
      input_matrix_test,label_vector_test = hog_test.make_data()

      train_xy = (input_matrix_train,label_vector_train)
      val_xy = (input_matrix_val,label_vector_val)
      test_xy = (input_matrix_test,label_vector_test)

      if args.debug == 1: # debug mode
        svm_val = earSVM(train_xy = val_xy,test_xy = val_xy, args = args)
        svm_val.train()
        svm_val.evaluate("val")
      elif args.debug == 0: # normal mode
        svm_test = earSVM(train_xy = train_xy,test_xy = test_xy, args = args)
        if args.path_model_loaded == "": # train mode
          svm_test.train()
          svm_test.evaluate("test")
        else: # eval mode
          svm_test.load(args.path_model_loaded)
          svm_test.evaluate("test")
      else: # throw error
        print("debug mode {} not recognized, need to be 0 or 1.".format(args.debug))
        raise NotImplementedError
      
    else:
      awe_dataset_train,awe_dataset_val,awe_dataset_test,awe_dataloader_train,awe_dataloader_val,awe_dataloader_test = awe_prepare_data_ml(df,args)
      awe_dataset_trainval = torch.utils.data.Subset(awe_dataset_train,list(range(200)))
      awe_dataloader_trainval = DataLoader(awe_dataset_trainval,batch_size = 32, shuffle=True)

      if '+' in args.eval_split:
        split_list = args.eval_split.split('+')
      else:
        split_list = [args.eval_split]
      
      dataloader_list = []
      split_list_ordered = []

      if 'test' in split_list:
        dataloader_list.append(awe_dataloader_test) 
        split_list_ordered.append('test')
      if 'val' in split_list:
        dataloader_list.append(awe_dataloader_val) 
        split_list_ordered.append('val')
      if 'train' in split_list:
        dataloader_list.append(awe_dataloader_train) 
        split_list_ordered.append('train')
      if 'train_val' in split_list:
        dataloader_list.append(awe_dataloader_trainval)
        split_list_ordered.append('train_val')
      # else:
      #   print("split {} not recognized.".format(args.eval_split))
      #   raise NotImplementedError


      unique_ids = set()
      
      unique_ids.update(awe_dataset_train.unique_ids)
      unique_ids.update(awe_dataset_val.unique_ids)
      unique_ids.update(awe_dataset_test.unique_ids)

      # init model

      model = build_model(args.model,unique_ids,args.path_model_loaded)

      model.load_state_dict(torch.load(args.path_model_loaded))

      model.eval()

      assert len(dataloader_list)>0, "split(s) to evaluate should be more than 0"
      for i,dataloader in enumerate(dataloader_list):
        acc = evaluate_ear_model(dataloader,model,split_list_ordered[i])

  elif args.dataset == 'fgnet':
    df = pd.read_csv(args.info_path)

    if args.model == "sift":

      if '+' in args.eval_split:
        split_list = args.eval_split.split('+')
      else:
        split_list = [args.eval_split]
      
      fgnet_train_list,fgnet_val_list,fgnet_test_list = fgnet_prepare_data_none_ml(df,args)
      for split in split_list:
        if split == 'val':
          sift_val = siftEar(fgnet_train_list,fgnet_val_list,args)
          ear_source_image_path_and_descriptor_dict_val,ear_target_image_path_and_descriptor_dict_val = sift_val.compute_descriptor()
          sift_val.evaluate(ear_source_image_path_and_descriptor_dict_val,ear_target_image_path_and_descriptor_dict_val,split)
        elif split == 'test':
          sift_test = siftEar(fgnet_train_list,fgnet_test_list,args)
          ear_source_image_path_and_descriptor_dict_test,ear_target_image_path_and_descriptor_dict_test = sift_test.compute_descriptor()
          sift_test.evaluate(ear_source_image_path_and_descriptor_dict_test,ear_target_image_path_and_descriptor_dict_test,split)
        else:
          print("{} dataset not found",split)
          raise NotImplementedError

    elif args.model == "hog":

      # hog + svm

      if '+' in args.eval_split:
        split_list = args.eval_split.split('+')
      else:
        split_list = [args.eval_split]
      fgnet_train_list,fgnet_val_list,fgnet_test_list = fgnet_prepare_data_none_ml(df,args)

      hog_train = earHog(fgnet_train_list,args)
      hog_val = earHog(fgnet_val_list,args)
      hog_test = earHog(fgnet_test_list,args)

      input_matrix_train,label_vector_train = hog_train.make_data()
      input_matrix_val,label_vector_val = hog_val.make_data()
      input_matrix_test,label_vector_test = hog_test.make_data()

      train_xy = (input_matrix_train,label_vector_train)
      val_xy = (input_matrix_val,label_vector_val)
      test_xy = (input_matrix_test,label_vector_test)

      if args.debug == 1: # debug mode
        svm_val = earSVM(train_xy = val_xy,test_xy = val_xy, args = args)
        svm_val.train()
        svm_val.evaluate("val")
      elif args.debug == 0: # normal mode
        svm_test = earSVM(train_xy = train_xy,test_xy = test_xy, args = args)
        if args.path_model_loaded == "": # train mode
          svm_test.train()
          svm_test.evaluate("test")
        else: # eval mode
          svm_test.load(args.path_model_loaded)
          svm_test.evaluate("test")
      else: # throw error
        print("debug mode {} not recognized, need to be 0 or 1.".format(args.debug))
        raise NotImplementedError

    else:

      fgnet_dataset_train,fgnet_dataset_val,fgnet_dataset_test,fgnet_dataloader_train,fgnet_dataloader_val,fgnet_dataloader_test = fgnet_prepare_data_ml(df,args)
      fgnet_dataset_trainval = torch.utils.data.Subset(fgnet_dataset_train,list(range(200)))
      fgnet_dataloader_trainval = DataLoader(fgnet_dataset_trainval,batch_size = 32, shuffle=True)

      if '+' in args.eval_split:
        split_list = args.eval_split.split('+')
      else:
        split_list = [args.eval_split]
      dataloader_list = []
      split_list_ordered = []

      if 'test' in split_list:
        dataloader_list.append(fgnet_dataloader_test) 
        split_list_ordered.append('test')
      if 'val' in split_list:
        dataloader_list.append(fgnet_dataloader_val) 
        split_list_ordered.append('val')
      if 'train' in split_list:
        dataloader_list.append(fgnet_dataloader_train) 
        split_list_ordered.append('train')
      if 'train_val' in split_list:
        dataloader_list.append(fgnet_dataloader_trainval)
        split_list_ordered.append('train_val')
      if len(dataloader_list) == 0:
        print("split {} not recognized.".format(args.eval_split))
        raise NotImplementedError


      unique_ids = set()
      
      unique_ids.update(fgnet_dataset_train.unique_ids)
      unique_ids.update(fgnet_dataset_val.unique_ids)
      unique_ids.update(fgnet_dataset_test.unique_ids)

      # init model

      model = build_model(args.model,unique_ids,args.path_model_loaded)

      model.load_state_dict(torch.load(args.path_model_loaded))

      model.eval()

      assert len(dataloader_list)>0, "split(s) to evaluate should be more than 0"
      for i,dataloader in enumerate(dataloader_list):
        acc = evaluate_ear_model(dataloader,model,split_list_ordered[i])

  else:
    print("dataset {} not recognized.".format(args.dataset))
    raise NotImplementedError

  # print(test_list)