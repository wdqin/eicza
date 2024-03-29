from param import args
from utils import build_model
from model import train_model,evaluate_ear_model
from hog import earHog
from svm import earSVM
from datasets.icz_dataset import icz_dataset,icz_prepare_data_ml,icz_prepare_data_none_ml
from datasets.awe_dataset import awe_dataset,awe_prepare_data_ml,awe_prepare_data_none_ml
from datasets.fgnet_dataset import fgnet_dataset,fgnet_prepare_data_ml,fgnet_prepare_data_none_ml

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random
from datetime import datetime
import logging

# set up random seed for reproducibility

# get date-time

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

#

def build_optimizer(model,optimizer_name,):
  if optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
  else:
    raise NotImplementedError
  return optimizer

def train(dataloader_train,dataloader_val_list,model,epochs = 100,save_path_best = ""):
  
  running_loss = 0.0
  loss_epoch = 0.0
  evaluate_every = 200
  print_loss_count = 0
  print_loss_every = 100
  iter_count = 1
  best_top1_acc = 0.0
  best_top5_acc = 0.0

  optimizer = build_optimizer(model,args.optim)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(epochs):
    print("epoch {} starts ...".format(epoch))
    for i,data in enumerate(dataloader_train):
      iter_count+=1
      optimizer.zero_grad()
      loss = train_model(data,model,optimizer,criterion,dataset = args.dataset)
      # print("loss",loss)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      print_loss_count+=1
      loss_epoch+= loss.item()

      if i % print_loss_every == 1:
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_loss_count:.3f}')
        running_loss = 0.0
        print_loss_count = 0

      if iter_count % evaluate_every == 0:
        accuracyList = []
        split = ['trainval','val']
        for i,val_dataloader in enumerate(dataloader_val_list):
          top1_acc,top5_acc = evaluate_ear_model(val_dataloader,model,split[i])

          writer.add_scalar("acc_top1_acc/"+split[i], top1_acc, iter_count)
          writer.add_scalar("acc_top5_acc/"+split[i], top5_acc, iter_count)

          if split[i]=='val': #val
            acc_top1_this_interval = top1_acc

            if acc_top1_this_interval > best_top1_acc:
              best_top1_acc = acc_top1_this_interval
              print("new top-1 best Acc: {:.2f}".format(best_top1_acc))
              torch.save(model.state_dict(), save_path_best+"_top1")

            acc_top5_this_interval = top5_acc
            if acc_top5_this_interval > best_top5_acc or (acc_top5_this_interval == best_top5_acc and top1_acc>best_top1_acc):
              best_top5_acc = acc_top5_this_interval
              print("new top-5 best Acc: {:.2f}".format(best_top5_acc))
              torch.save(model.state_dict(), save_path_best+"_top5")
    print(f'[{epoch + 1}] loss: {loss_epoch / len(dataloader_train):.3f}')
    writer.add_scalar("Loss/train_epoch", loss_epoch / len(dataloader_train), epoch)
    loss_epoch = 0.0
  split = ['trainval','val']
  for i,val_dataloader in enumerate(dataloader_val_list):
    top1_acc,top5_acc = evaluate_ear_model(val_dataloader,model,split[i])
    torch.save(model.state_dict(), save_path_best+"_"+split[i]+"_last")
  writer.flush()

  

if __name__ == "__main__":
  


  if args.debug == 1:
    print("------------------------- ATTENTION: DEBUG MODE IS ON -------------------------")

  # editing logging

  # set up random seed
  random.seed(10)
  np.random.seed(0)
  torch.manual_seed(42)
  
  # set up logger

  # print(args.info_path)
  print(args)
  writer = SummaryWriter(log_dir='./runs/log/'+args.log_name+"_"+dt_string)

  logging.basicConfig(level=logging.INFO, format='%(message)s')
  logger = logging.getLogger()
  logger.addHandler(logging.FileHandler('./runs/log/'+args.log_name+"_"+dt_string+'/'+args.log_name+"_"+dt_string, 'a+'))
  print = logger.info

  if args.dataset == 'icz':

    df = pd.read_csv(args.info_path)
    icz_dataset_train,icz_dataset_val,icz_dataset_test,icz_dataloader_train,\
                                    icz_dataloader_val,icz_dataloader_test = \
                                    icz_prepare_data_ml(df,args)


    # make a train validation data loader
    icz_dataset_trainval = torch.utils.data.Subset(icz_dataset_train,list(range(200)))
    icz_dataloader_trainval = DataLoader(icz_dataset_trainval,batch_size = 32, shuffle=True)

    # count unique ids

    unique_ids = set()
    
    unique_ids.update(icz_dataset_train.unique_ids)
    unique_ids.update(icz_dataset_val.unique_ids)
    unique_ids.update(icz_dataset_test.unique_ids)

    if args.path_model_loaded == "":
      path_model_loaded = None
    else:
      path_model_loaded = args.path_model_loaded

    if args.model == "sift":
      print("the sift ear matching is implemented in eval.py.")
      raise NotImplementedError
    elif args.model == "hog":
      print("the hog + svm ear matching is implemented in eval.py.")
      raise NotImplementedError
      
    else:
      model = build_model(args.model,unique_ids,path_model_loaded)

    icz_dataloader_val_list = [icz_dataloader_trainval,icz_dataloader_val]
    if args.debug == 1:
      train(icz_dataloader_trainval,icz_dataloader_val_list,model,epochs = args.epochs,save_path_best = args.path_best)
    elif args.debug == 0:
      train(icz_dataloader_train,icz_dataloader_val_list,model,epochs = args.epochs,save_path_best = args.path_best)
    else:
      print("the command is neither a debug or non-debug mode.")
      raise NotImplementedError  
    print("training completed.")
  elif args.dataset == 'awe':
    df = pd.read_csv(args.info_path)
    awe_dataset_train,awe_dataset_val,awe_dataset_test,awe_dataloader_train,\
                                    awe_dataloader_val,awe_dataloader_test = \
                                    awe_prepare_data_ml(df,args)

    # make a train validation data loader
    awe_dataset_trainval = torch.utils.data.Subset(awe_dataset_train,list(range(200)))
    awe_dataloader_trainval = DataLoader(awe_dataset_trainval,batch_size = 32, shuffle=True)

    # count unique ids

    unique_ids = set()
    
    unique_ids.update(awe_dataset_train.unique_ids)
    unique_ids.update(awe_dataset_val.unique_ids)
    unique_ids.update(awe_dataset_test.unique_ids)

    if args.path_model_loaded == "":
      path_model_loaded = None
    else:
      path_model_loaded = args.path_model_loaded

    if args.model == "sift":
      print("the sift ear matching is implemented in eval.py.")
      raise NotImplementedError
    elif args.model == "hog":
      print("the hog + svm ear matching is implemented in eval.py.")
      raise NotImplementedError
    else:
      model = build_model(args.model,unique_ids,path_model_loaded)

    awe_dataloader_val_list = [awe_dataloader_trainval,awe_dataloader_val]
    if args.debug == 1:
      train(awe_dataloader_trainval,awe_dataloader_val_list,model,epochs = args.epochs,save_path_best = args.path_best)
    elif args.debug == 0:
      train(awe_dataloader_train,awe_dataloader_val_list,model,epochs = args.epochs,save_path_best = args.path_best)
    else:
      print("the command is neither a debug or non-debug mode.")
      raise NotImplementedError  
    print("training completed.")
      
  elif args.dataset == 'fgnet':
    df = pd.read_csv(args.info_path)
    fgnet_dataset_train,fgnet_dataset_val,fgnet_dataset_test,fgnet_dataloader_train,\
                                    fgnet_dataloader_val,fgnet_dataloader_test = \
                                    fgnet_prepare_data_ml(df,args)

    # make a train validation data loader
    fgnet_dataset_trainval = torch.utils.data.Subset(fgnet_dataset_train,list(range(200)))
    fgnet_dataloader_trainval = DataLoader(fgnet_dataset_trainval,batch_size = 32, shuffle=True)

    unique_ids = set()
    
    unique_ids.update(fgnet_dataset_train.unique_ids)
    unique_ids.update(fgnet_dataset_val.unique_ids)
    unique_ids.update(fgnet_dataset_test.unique_ids)

    if args.path_model_loaded == "":
      path_model_loaded = None
    else:
      path_model_loaded = args.path_model_loaded

    model = build_model(args.model,unique_ids,path_model_loaded)

    fgnet_dataloader_val_list = [fgnet_dataloader_trainval,fgnet_dataloader_val]
    if args.debug == 1:
      train(fgnet_dataloader_trainval,fgnet_dataloader_val_list,model,epochs = args.epochs,save_path_best = args.path_best)
    elif args.debug == 0:
      train(fgnet_dataloader_train,fgnet_dataloader_val_list,model,epochs = args.epochs,save_path_best = args.path_best)
    else:
      print("the command is neither a debug or non-debug mode.")
      raise NotImplementedError  
    print("training completed.")


  else:
    raise NotImplementedError

  writer.close()