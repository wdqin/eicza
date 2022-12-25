from model import earResnet18Model,earResnet50Model
from model import earSqueezeNet10Model,earSqueezeNet11Model,earSEModel
from model import earVGGFaceResNet50Model,earVGGFaceSENet50Model

from tqdm import tqdm

def build_model(model_name, unique_ids,pretrain_file=None):
	
	# init model

    if model_name == 'resnet18':
      model = earResnet18Model(uniqueSubjectCount = len(unique_ids),pretrained = pretrain_file).cuda()
    elif model_name == 'resnet50':
      model = earResnet50Model(uniqueSubjectCount = len(unique_ids),pretrained = pretrain_file).cuda()
    elif model_name == 'squeezenet10':
      model = earSqueezeNet10Model(uniqueSubjectCount = len(unique_ids),pretrained = pretrain_file).cuda()
    elif model_name == 'squeezenet11':
      model = earSqueezeNet11Model(uniqueSubjectCount = len(unique_ids),pretrained = pretrain_file).cuda()
    elif model_name == 'senet':
      model = earSEModel(uniqueSubjectCount = len(unique_ids)).cuda()
    elif model_name == 'vggface2_resnet50':
      model = earVGGFaceResNet50Model(uniqueSubjectCount = len(unique_ids),pretrained = pretrain_file).cuda()
    elif model_name == 'vggface2_senet50':
      model = earVGGFaceSENet50Model(uniqueSubjectCount = len(unique_ids),pretrained = pretrain_file).cuda()
    else:
      print("the model name {} does not match any model available.".format(model_name))
      raise NotImplementedError

    return model

def getEarImageListByEarSubjects(ear_list,image_folder_path,dataset_name):
  earSubjectDict = {}
  for earIdx in tqdm(range(len(ear_list))):
    if dataset_name == 'icz':
      earImagePath = image_folder_path+ear_list[earIdx]['earImageName']
      if ear_list[earIdx]['earSubjectIdx'] not in earSubjectDict:
        ears = []
        ears.append(earImagePath)
        earSubjectDict[ear_list[earIdx]['earSubjectIdx']] = ears
      else:
        earSubjectDict[ear_list[earIdx]['earSubjectIdx']].append(earImagePath)
    elif dataset_name == 'awe':
      earImagePath = ear_list[earIdx]['imagePath']
      if ear_list[earIdx]['subject'] not in earSubjectDict:
        ears = []
        ears.append(earImagePath)
        earSubjectDict[ear_list[earIdx]['subject']] = ears
      else:
        earSubjectDict[ear_list[earIdx]['subject']].append(earImagePath)
    elif dataset_name == 'fgnet':
      earImagePath = image_folder_path+ear_list[earIdx]['jpgName']
      if ear_list[earIdx]['personID'] not in earSubjectDict:
        ears = []
        ears.append(earImagePath)
        earSubjectDict[ear_list[earIdx]['personID']] = ears
      else:
        earSubjectDict[ear_list[earIdx]['personID']].append(earImagePath)
    else:
      raise NotImplementedError

    

  return earSubjectDict