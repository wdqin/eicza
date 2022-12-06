from model import earResnet18Model,earResnet50Model
from model import earSqueezeNet10Model,earSqueezeNet11Model,earSEModel,earVGGFaceResNet50Model



def build_model(model_name, unique_ids,pretrain_file=None):
	
	# init model

    if model_name == 'resnet18':
      model = earResnet18Model(uniqueSubjectCount = len(unique_ids)).cuda()
    elif model_name == 'resnet50':
      model = earResnet50Model(uniqueSubjectCount = len(unique_ids)).cuda()
    elif model_name == 'squeezenet10':
      model = earSqueezeNet10Model(uniqueSubjectCount = len(unique_ids)).cuda()
    elif model_name == 'squeezenet11':
      model = earSqueezeNet11Model(uniqueSubjectCount = len(unique_ids)).cuda()
    elif model_name == 'senet':
      model = earSEModel(uniqueSubjectCount = len(unique_ids)).cuda()
    elif model_name == 'vggface2_resnet50':
      model = earVGGFaceResNet50Model(uniqueSubjectCount = len(unique_ids),pretrained = pretrain_file).cuda()
    else:
      print("the model name {} does not match any model available.".format(model_name))
      raise NotImplementedError

    return model