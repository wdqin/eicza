import cv2
import numpy as np
from tqdm import tqdm
from utils import getEarImageListByEarSubjects
import pickle

def extractDesriptorFromImage(earImagePath,SIFT,image_dict):
	img = image_dict[earImagePath]['img_encoded']
	# img = cv2.imread(earImagePath)

	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	kp, descriptor = SIFT.detectAndCompute(gray,None)
	return descriptor
def extractDescriptorListFromImageList(earImageList,SIFT,image_dict):
	descriptorList = []
	for earImagePath in earImageList:
		descriptorList.append(extractDesriptorFromImage(earImagePath,SIFT,image_dict))
	return descriptorList

def buildDescriptorListForImageDict(earImageListDict,SIFT,image_dict):
	earImagePathAndDescriptorDict = {}
	print("extracting descriptor list into dict")
	for key in tqdm(earImageListDict.keys()):
		earImageDict = {}
		earImageList = earImageListDict[key]
		descriptorList = extractDescriptorListFromImageList(earImageList,SIFT,image_dict)

		earImageDict['imageList'] = earImageList
		earImageDict['descriptorList'] = descriptorList
		earImagePathAndDescriptorDict[key] = earImageDict

	return earImagePathAndDescriptorDict

def getBestMatchingScoreFromDescriptorList(desListA,desListB):
	bestMatchingScore = None
	for desA in desListA:
		for desB in desListB:
			score = getTopKMatchScoresAverage(desA,desB)
			if bestMatchingScore is None or bestMatchingScore>score:
				bestMatchingScore = score
	return bestMatchingScore


def getTopKMatchScoresAverage(desA,desB,topK = 10):
	def computeDistanceScoreBetweenTwoDescriptor(desA,desB):

		bf = cv2.BFMatcher()
		matches = bf.knnMatch(desA,desB,k=1)

		return matches

	matches = computeDistanceScoreBetweenTwoDescriptor(desA,desB)
	distanceScores = []
	for keyPoint in matches:
		for bestMatch in keyPoint:
			distanceScores.append(bestMatch.distance)
	distanceScores.sort()
	if(len(distanceScores)<topK):
		topK = len(distanceScores)

	topKScores = distanceScores[:topK]

	return np.average(topKScores)

def getEarMatchingPrediction(earImagePathAndDescriptorDictTrain,earImagePathAndDescriptorDictTest,k=5):
	predictions = {}
	prediction_scores = {}
	prediction_topk_scores = {}
	print("generating predictions...")
	for keyTrain in tqdm(earImagePathAndDescriptorDictTrain):
		bestMatchScore = None
		bestMatchId = None

		topNMatchScore = None
		topNMatchIds = None

		desListA = earImagePathAndDescriptorDictTrain[keyTrain]['descriptorList']

		for keyTest in earImagePathAndDescriptorDictTest:
			desListB = earImagePathAndDescriptorDictTest[keyTest]['descriptorList']

			score = getBestMatchingScoreFromDescriptorList(desListA,desListB)

			if(bestMatchScore is None or bestMatchScore>score):
				bestMatchScore = score
				bestMatchId = keyTest
			if topNMatchIds is None:
				topNMatchIds = [keyTest]
				topNMatchScore = [score]
			else:
				topNMatchIds.append(keyTest)
				topNMatchScore.append(score)

		topNScores = np.array(topNMatchScore)
		topNIndexes = np.argsort(topNScores)
		topNCandidate = [topNMatchIds[i] for i in topNIndexes]
		topk = topNCandidate[:k]

		prediction_topk_scores[keyTrain] = topk
		predictions[keyTrain] = bestMatchId
		prediction_scores[keyTrain] = bestMatchScore

	return predictions,prediction_scores,prediction_topk_scores

def calculateAccuracy(predictions,topkidx):
	total = 0
	count = 0
	countTopk = 0
	for key in predictions:
		value = predictions[key]
		if(key == value):
			count+=1
		if (key in topkidx[key]):
			countTopk+=1
		total+=1
	acc = count/total
	accTopk = countTopk/total
	return acc,accTopk

class siftEar:
	def __init__(self, source_list,target_list,args):
		
		self.ear_source_subject_dict = getEarImageListByEarSubjects(source_list,args.image_folder_path,args.dataset)
		self.ear_target_subject_dict = getEarImageListByEarSubjects(target_list,args.image_folder_path,args.dataset)

		with open(args.load_dataset_path, "rb") as f: # "rb" because we want to read in binary mode
			self.image_dict = pickle.load(f)
	def compute_descriptor(self):
		SIFT = cv2.SIFT_create()
		ear_source_image_path_and_descriptor_dict = buildDescriptorListForImageDict(self.ear_source_subject_dict,SIFT,self.image_dict)
		ear_target_image_path_and_descriptor_dict = buildDescriptorListForImageDict(self.ear_target_subject_dict,SIFT,self.image_dict)
		return ear_source_image_path_and_descriptor_dict,ear_target_image_path_and_descriptor_dict
	def evaluate(self,ear_source_image_path_and_descriptor_dict,ear_target_image_path_and_descriptor_dict,split):
		predictions,prediction_scores,topkidx = getEarMatchingPrediction(ear_source_image_path_and_descriptor_dict,ear_target_image_path_and_descriptor_dict)
		acc,accTopk = calculateAccuracy(predictions,topkidx)
		print('Top-1 accuracy of the network on the {} ears: {:.2f} %'.format(split,100 * acc))
		print('Top-5 accuracy of the network on the {} ears: {:.2f} %'.format(split,100 * accTopk))
		return predictions,prediction_scores,topkidx

		