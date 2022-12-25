import torch
import torch.nn as nn
import torch.optim as optim
import torch.hub as hub
import torchvision

from tqdm import tqdm
import pickle

from vggface2.models.resnet import resnet50
from vggface2.models.senet import senet50

hub.set_dir("./pre_trained/")

N_IDENTITY = 8631  # total number of identities in VGG Face2


def load_vggface2_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))

class earVGGFaceResNet50Model(torch.nn.Module):

    def __init__(self,uniqueSubjectCount,pretrained = None):
        super(earVGGFaceResNet50Model, self).__init__()

        
        if(pretrained is not None):
            if (pretrained.endswith('.pkl')):
                self.resnet = resnet50(num_classes=N_IDENTITY, include_top=False)
                load_vggface2_state_dict(self.resnet,pretrained)
                print("resnet loaded from {}".format(pretrained))
            else:
                self.resnet = resnet50(num_classes=N_IDENTITY, include_top=False)
                print("senet loaded from {}".format(pretrained))
        else:
            self.resnet = resnet50(num_classes=N_IDENTITY, include_top=False)
            print("resnet trained from scratch or loaded from saved snapshot.")

        self.fc = nn.Linear(2048, uniqueSubjectCount+1)

    def forward(self, x):
        
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class earVGGFaceSENet50Model(torch.nn.Module):

    def __init__(self,uniqueSubjectCount,pretrained = None):
        super(earVGGFaceSENet50Model, self).__init__()

        if(pretrained is not None):
            if pretrained.endswith('.pkl'):
                self.senet = senet50(num_classes=N_IDENTITY, include_top=False)
                load_vggface2_state_dict(self.senet,pretrained)
                print("senet loaded from {}".format(pretrained))
            else:
                self.senet = senet50(num_classes=N_IDENTITY, include_top=False)
                print("senet loaded from {}".format(pretrained))
        else:
            self.senet = senet50(num_classes=N_IDENTITY, include_top=False)
            print("senet trained from scratch or loaded from saved snapshot.")
        self.fc = nn.Linear(2048, uniqueSubjectCount+1)

    def forward(self, x):
        
        x = self.senet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class earResnet50Model(torch.nn.Module):

    def __init__(self,uniqueSubjectCount,pretrained = None):
        super(earResnet50Model, self).__init__()

        if pretrained == 'ImageNet':
            is_pretrained = True
        else:
            is_pretrained = False

        resNetModel = hub.load('pytorch/vision:v0.10.0', 'resnet50',pretrained=is_pretrained)

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

    def __init__(self,uniqueSubjectCount,pretrained = None):
        super(earSqueezeNet10Model, self).__init__()

        if pretrained == 'ImageNet':
            is_pretrained = True
        else:
            is_pretrained = False

        squeezeNet = torchvision.models.squeezenet1_0(pretrained=is_pretrained)
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

    def __init__(self,uniqueSubjectCount,pretrained = None):
        super(earSqueezeNet11Model, self).__init__()

        if pretrained == 'ImageNet':
            is_pretrained = True
        else:
            is_pretrained = False

        squeezeNet = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=is_pretrained)
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

    def __init__(self,uniqueSubjectCount,pretrained = None):
        super(earResnet18Model, self).__init__()

        if pretrained == 'ImageNet':
            is_pretrained = True
        else:
            is_pretrained = False

        resNetModel = hub.load('pytorch/vision:v0.10.0', 'resnet18',pretrained=is_pretrained)
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
