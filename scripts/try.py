import torch
import torch.nn as nn

from param import args
import pandas as pd

from tqdm import tqdm
import torch.optim as optim

import torch.hub as hub
import torchvision.models as models

from datasets.icz_dataset import icz_dataset,prepare_data

hub.set_dir("./pre_trained/")

args.image_folder_path = './datasets/infantCohortZambia/jpgs/'
args.info_path = './datasets/infantCohortZambia/info.csv'
args.batch_size = 8
# args['image_folder_path']
df = pd.read_csv(args.info_path)
icz_dataset_train,icz_dataset_val,icz_dataset_test,icz_dataloader_train,icz_dataloader_val,icz_dataloader_test = prepare_data(df,args)

model = models.squeezenet1_0(pretrained=True).cuda()
model_extract = nn.Sequential(*list(model.children())[:-1])

# 
# Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
#     (2): ReLU(inplace=True)
#     (3): AdaptiveAvgPool2d(output_size=(1, 1))
#   )
# 


classifier = nn.Sequential(
          nn.Dropout(p=0.5, inplace=False),
          nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)),
          nn.ReLU(inplace=True),
          nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ).cuda()

print("model",model)
# print("model_extract",model_extract)

for data in icz_dataloader_train:
	ear_images = data['ear_image'].cuda()
	out = model_extract(ear_images)
	# print(out.shape)
	out2 = classifier(out)
	out2 = out2.squeeze(-1).squeeze(-1)
	print(out2.shape)
	raise NotImplementedError