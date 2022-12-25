import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class earSVM:
	def __init__(self,train_xy,test_xy,args = None):
		self.train_x,self.train_y = train_xy
		self.test_x,self.test_y = test_xy
		self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
		self.saved_path = args.path_best
	
	def train(self):
		print("training...")
		self.clf.fit(self.train_x,self.train_y)
		with open(self.saved_path+'.pkl','wb') as f:
			pickle.dump(self.clf,f)
		print("training is done.")
		print("model is saved at {}".format(self.saved_path+'.pkl'))

	def load(self,loaded_path):
		with open(loaded_path, 'rb') as f:
			self.clf = pickle.load(f)
		print("the model is loaded from {}".format(loaded_path))
	def evaluate(self,split,k = 5):
		print("evaluating ...")
		accuracy = self.clf.score(self.test_x,self.test_y)
		
		
		probs = self.clf.predict_proba(self.test_x)
		
		sort_index = np.argsort(probs)
		top_k_correct  = 0
		top_k_total = 0
		for i,row in enumerate(sort_index):
			topk = row[::-1][:k]
			print("topk",topk)
			print("self.test_y[i]",self.test_y[i])
			print("--------------------------------")
			if self.test_y[i] in topk:
				top_k_correct +=1
			top_k_total+=1
		print('Top-1 accuracy of the network on the {} ears: {:.2f} %'.format(split,100 * accuracy))
		print('Top-{} accuracy of the network on the {} ears: {:.2f} %'.format(k,split,100 * (top_k_correct/top_k_total)))

		print("evaluation is done.")
