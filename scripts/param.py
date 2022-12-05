import argparse

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")
        self.parser.add_argument('--debug', type=int, default=0, help='enable debug mode.')
        self.parser.add_argument('--dataset', type=str, default="", help='choose the dataset to work with.')
        self.parser.add_argument('--eval_split', type=str, default="", help='split(s) to evaluate, split by plus symbol +')

        self.parser.add_argument('--model', type=str, default="", help='choose the model to be trained.')

        self.parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train the model.')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

        self.parser.add_argument('--log_name', type=str, default="", help='log name to save.')
        self.parser.add_argument('--path_model_eval', type=str, default="", help='path of the model to be evaluated.')
        self.parser.add_argument('--path_best', type=str, default="", help='path to save model snapshots.')
        self.parser.add_argument('--info_path', type=str, default="", help='an info .csv file path.')
        self.parser.add_argument('--image_folder_path', type=str, default="./", help='image (jpgs) folder path.')
        self.parser.add_argument('--optim', type=str, default="", help='optimizer to train the model.')

        self.args = self.parser.parse_args()

param = Param()
args = param.args