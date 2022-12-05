import torch
import torch.nn as nn

from tqdm import tqdm
import torch.optim as optim

import torch.hub as hub
import torchvision

hub.set_dir("./pre_trained/")

class earResnet50Model(torch.nn.Module):

    def __init__(self,uniqueSubjectCount):
        super(earResnet50Model, self).__init__()

        resNetModel = hub.load('pytorch/vision:v0.10.0', 'resnet50',pretrained=True)
        self.resNet = nn.Sequential(*list(resNetModel.children())[:-1])

        self.linear = torch.nn.Linear(2048, uniqueSubjectCount+1)
        # self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.resNet(x)
        x = torch.flatten(x,1)
        x = self.linear(x)
        # x = self.softmax(x)
        return x

class earSqueezeNet10Model(torch.nn.Module):
    # squeezenet1_0 model

    def __init__(self,uniqueSubjectCount):
        super(earSqueezeNet10Model, self).__init__()

        squeezeNet = torchvision.models.squeezenet1_0(pretrained=True)
        self.squeezeNet_extract = nn.Sequential(*list(squeezeNet.children())[:-1])

        self.classifier = nn.Sequential(
          nn.Dropout(p=0.5, inplace=False),
          nn.Conv2d(512, uniqueSubjectCount+1, kernel_size=(1, 1), stride=(1, 1)),
          nn.ReLU(inplace=True),
          nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        # self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):

        feature = self.squeezeNet_extract(x)
        x = self.classifier(feature)
        x = x.squeeze(-1).squeeze(-1)
        # x = self.softmax(x)

        return x

class earSqueezeNet11Model(torch.nn.Module):
    # squeezenet1_1 model

    def __init__(self,uniqueSubjectCount):
        super(earSqueezeNet11Model, self).__init__()

        squeezeNet = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
        self.squeezeNet_extract = nn.Sequential(*list(squeezeNet.children())[:-1])

        self.classifier = nn.Sequential(
          nn.Dropout(p=0.5, inplace=False),
          nn.Conv2d(512, uniqueSubjectCount+1, kernel_size=(1, 1), stride=(1, 1)),
          nn.ReLU(inplace=True),
          nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        # self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):

        feature = self.squeezeNet_extract(x)
        x = self.classifier(feature)
        x = x.squeeze(-1).squeeze(-1)
        # x = self.softmax(x)

        return x

class earResnet18Model(torch.nn.Module):

    def __init__(self,uniqueSubjectCount):
        super(earResnet18Model, self).__init__()

        resNetModel = hub.load('pytorch/vision:v0.10.0', 'resnet18',pretrained=True)
        self.resNet = nn.Sequential(*list(resNetModel.children())[:-1])

        self.linear = torch.nn.Linear(512, uniqueSubjectCount+1)
        # self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.resNet(x)
        x = torch.flatten(x,1)
        x = self.linear(x)
        # x = self.softmax(x)
        return x

class earSEModel(torch.nn.Module):

    def __init__(self,uniqueSubjectCount):

        super(earSEModel, self).__init__()
        self.se_model = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', num_classes=uniqueSubjectCount+1)

    def forward(self, x):
        x = self.se_model(x)
        return x

def train_model(data,model,optimizer,criterion,dataset=""):
  ear_images = data['ear_image'].cuda()
  logits = model(ear_images)
  targets = data['ear_subject_idx'].long().cuda()
  loss = criterion(logits,targets)
  
  return loss

def evaluate_ear_model(evalDataloader,earModel,split):
  total = 0
  correct = 0
  top5 = 0
  with torch.no_grad():
    for i,data in enumerate(evalDataloader):
      ear_images = data['ear_image'].cuda()
      logits = earModel(ear_images)
      targets = data['ear_subject_idx'].long().cuda()

      logits_ = logits.clone()
      logits_sorted, indices = torch.sort(logits_,descending=True)
      _, predicted = torch.max(logits, 1)

      total += targets.size(0)
      correct += (predicted == targets).sum().item()
      
      for i,target in enumerate(targets):
        if target in indices[i,:5]:
            top5+=1

    print('Top-1 accuracy of the network on the {} ears: {:.2f} %'.format(split,100 * correct / total))
    print('Top-5 accuracy of the network on the {} ears: {:.2f} %'.format(split,100 * top5 / total))

  return correct / total, top5 / total

# def evaluateEarModel(evalDataloader,earModel,split):
#   total = 0
#   correct = 0
#   top5 = 0
#   with torch.no_grad():
#     for i,earInputBatch in enumerate(evalDataloader):
#       logits = earModel(earInputBatch['earImage'])
#       targets = torch.tensor(earInputBatch['earSubjectIdx']).long().cuda()

#       logits_ = logits.clone()
#       logits_sorted, indices = torch.sort(logits_,descending=True)
#       _, predicted = torch.max(logits, 1)

#       total += targets.size(0)
#       correct += (predicted == targets).sum().item()
      
#       for i,target in enumerate(targets):
#         if target in indices[i,:5]:
#             top5+=1

#     print('Top-1 accuracy of the network on the {} ears: {:.2f} %'.format(split,100 * correct / total))
#     print('Top-5 accuracy of the network on the {} ears: {:.2f} %'.format(split,100 * top5 / total))
#   return correct / total, top5 / total

# def trainEarModel(trainDataloader,earModel,optimizer,criterion,setEpoch = 100,valDataloaderList=[],savePathBest="",savePathLast=""):
#   running_loss = 0.0
#   evaluateEvery = 100 #iterations
#   iterCount = 0
#   bestTop1Acc = 0
#   bestTop5Acc = 0
#   for epoch in range(setEpoch):
#     print("epoch {} starts...".format(epoch))
#     for i,earInputBatch in enumerate(trainDataloader):
#       iterCount+=1
#       optimizer.zero_grad()
#       earInputBatch = earInputBatch
#       logits = earModel(earInputBatch['earImage'])
#       targets = torch.tensor(earInputBatch['earSubjectIdx']).long().cuda()
#       loss = criterion(logits,targets)
#       loss.backward()
#       optimizer.step()
#       running_loss += loss.item()
#       if i % 20 == 1:    # print every 5 mini-batches
#         print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
#         running_loss = 0.0
#       if iterCount % evaluateEvery == 0:
#         accuracyList = []
#         split = ['trainVal','val']
#         for i,vallDataloader in enumerate(valDataloaderList):
#           top1Acc,top5Acc = evaluateEarModel(vallDataloader,earModel,split[i])
#           if split[i]=='val': #val
#             accTop1ThisInterval = top1Acc
#             if accTop1ThisInterval > bestTop1Acc:
#               bestTop1Acc = accTop1ThisInterval
#               print("new top-1 best Acc: {}".format(bestTop1Acc))
#               torch.save(earModel.state_dict(), savePathBest+"Top1")

#             accTop5ThisInterval = top5Acc
#             if accTop5ThisInterval > bestTop5Acc or (accTop5ThisInterval == bestTop5Acc and top1Acc>bestTop1Acc):
#               bestTop5Acc = accTop5ThisInterval
#               print("new top-5 best Acc: {}".format(bestTop5Acc))
#               torch.save(earModel.state_dict(), savePathBest+"Top5")

#   split = ['trainVal','val']
#   for i,vallDataloader in enumerate(valDataloaderList):
#     top1Acc,top5Acc = evaluateEarModel(vallDataloader,earModel,split[i])
#   torch.save(earModel.state_dict(), savePathLast)