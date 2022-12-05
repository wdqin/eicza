from model import earResnet18Model,earResnet50Model,earSqueezeNet10Model,earSqueezeNet11Model,earSEModel

def build_model(model_name, unique_ids):
	
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
    else:
      print("the model name {} does not match any model available.".format(args.model))
      raise NotImplementedError

    return model