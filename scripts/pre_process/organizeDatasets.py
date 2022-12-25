import os 
import csv
import re
import json
import sys
import copy
import random

# generate data info .csv

FGNET_DIR= './datasets/FGNET/images/'
FGNET_DATA_INFO_PATH = './datasets/FGNET/info' # leave suffixes

AWE_DIR = './datasets/awe_resized/'
AWE_DATA_INFO_PATH = './datasets/awe_resized/info.csv'

ICZ_DIR = './datasets/infantCohortZambia/'
ICZ_DATA_INFO_DIR = './datasets/infantCohortZambia/info'

def organizeDatasetICZ(dir,writeCSVInfoPath):
	
	# need directory './datasets/infantCohortZambia/'

	def getEarInfo(row,needIdx=True,needIsTrain=True,needEarSubjectIdx=True,needEarImageName=True,needPeriod=True,needDetail=True):
		earInfoDict = {}
		if needIdx:
			earInfoDict['idx'] = int(row['idx'])
		if needIsTrain:
			earInfoDict['isTrain'] = int(row['is_train'])
		if needEarSubjectIdx:
			earInfoDict['earSubjectIdx'] = int(row['ear_subject_idx'])
		if needEarImageName:
			earInfoDict['earImageName'] = str(row['ear_img_name'])
		if needPeriod:
			earInfoDict['period'] = str(row['period'])
		if needDetail:
			earDetailList = row['ear_img_name'].split('_')
			earInfoDict['earSubjectName'] = earDetailList[1]
			earInfoDict['earLeftRight'] = earDetailList[3]
			if earDetailList[4]== 'True':
				earInfoDict['earRotated'] = True
			else:
				earInfoDict['earRotated'] = False  
		return earInfoDict

	def getEarDictByIdx(path):
		earDict = {}
		with open(path, newline='') as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				earInfo = getEarInfo(row,needIdx=False)
				earDict[int(row['idx'])] = earInfo
		return earDict

	def getEarDictBySubjectIdx(path,returnSubjectCount=False):
		earDict = {}
		periodSet = set()
		with open(path, newline='') as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				earSubIdx = int(row['ear_subject_idx'])
				if earSubIdx not in earDict:
					earList = []
					earInfo = getEarInfo(row) #,needEarSubjectIdx=False
					if earInfo['period'] not in periodSet:
						periodSet.add(earInfo['period'])
					earList.append(earInfo)
					earDict[earSubIdx] = earList
				else: 
					earInfo = getEarInfo(row) #,needEarSubjectIdx=False
					if earInfo['period'] not in periodSet:
						periodSet.add(earInfo['period'])
					earDict[earSubIdx].append(earInfo)
		if(returnSubjectCount):
			return earDict,len(earDict),periodSet
		return earDict

	def getEarDictByPeriod(path,organizeByEarSubjectIdx=True,returnPeriodList=False):
		earDict = {}
		if organizeByEarSubjectIdx:
			with open(path, newline='') as csvfile:
				reader = csv.DictReader(csvfile)
				for row in reader:
					earPeriod = str(row['period'])
					if earPeriod not in earDict:
						earSubIdx = int(row['ear_subject_idx'])
						earSubIdxDict = {}
						earList = []
						earInfo = getEarInfo(row,needEarSubjectIdx=False,needPeriod=False)
						earList.append(earInfo)
						earSubIdxDict[earSubIdx] = earList
						earDict[earPeriod] = earSubIdxDict
					else:
						earSubIdx = int(row['ear_subject_idx'])
						if earSubIdx not in earDict[earPeriod]:
							earSubIdxDict = {}
							earList = []

							earInfo = getEarInfo(row,needEarSubjectIdx=False,needPeriod=False)
							earList.append(earInfo)
							earSubIdxDict[earSubIdx] = earList
						else:
							earInfo = getEarInfo(row,needEarSubjectIdx=False,needPeriod=False)
							earDict[earPeriod][earSubIdx].append(earInfo)
		else:
			with open(path, newline='') as csvfile:
				reader = csv.DictReader(csvfile)
				for row in reader:
					earPeriod = str(row['period'])
					if earPeriod not in earDict:
						earList = []
						earInfo = getEarInfo(row,needPeriod=False)
						earList.append(earInfo)
						earDict[earPeriod] = earList
					else:
						earInfo = getEarInfo(row,needPeriod=False)
						earDict[earPeriod].append(earInfo)
		if(returnPeriodList):
			return earDict,list(earDict.keys())
		return earDict

	def changeSubjectKeyIndexes(earList):
		earCount = 0
		earIndexDict = {}
		for earDict in earList:
			if (earDict['earSubjectIdx'] not in earIndexDict):
				earIndexDict[earDict['earSubjectIdx']] = earCount
				earCount+=1
		return earIndexDict

	def generateInfoCSVNoAgeProgression(csvPath,dataDirectory,earCSVOutFileName):
		IMAGE_THRESHOLD_TO_INCLUDE = 7

		earDataPath = dataDirectory+csvPath
		earBySubjectDict,subjectCount,periodSet = getEarDictBySubjectIdx(earDataPath,True)
		# print("There are {} objects in the dataset".format(subjectCount))

		maxSubjectEars = 0
		minSubjectEars = 9999

		maxSubjectEarsId = -1
		minSubjectEarsId = -1

		allDataDict = {}
		for earSubjectKey in earBySubjectDict.keys():
			earListOfASubject = earBySubjectDict[earSubjectKey]
			if(len(earListOfASubject)>IMAGE_THRESHOLD_TO_INCLUDE):
				allDataDict[earSubjectKey] = earBySubjectDict[earSubjectKey]
			if(len(earListOfASubject)>maxSubjectEars):
				maxSubjectEars = len(earListOfASubject)
				maxSubjectEarsId = earSubjectKey
			if(len(earListOfASubject)<minSubjectEars):
				minSubjectEars = len(earListOfASubject)
				minSubjectEarsId = earSubjectKey

		validationEarList = []
		testEarList = []
		trainingEarList = []

		for earSubject in allDataDict:
			earSubjectList = copy.deepcopy(allDataDict[earSubject])
			random.shuffle(earSubjectList)
			validationEarList += earSubjectList[:2]
			testEarList += earSubjectList[2:4]
			trainingEarList += earSubjectList[4:]

		earIndexDict = changeSubjectKeyIndexes(trainingEarList)
		#write data back into .csv

		# validationEarList = []
		# testEarList = []
		# trainingEarList = []

		with open(earCSVOutFileName, 'w', newline='') as csvfile:
			fieldnames = ['idx', 'split','earSubjectIdx','earImageName','period','earSubjectName','earLeftRight','earRotated']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()
			for ear in validationEarList:
				writer.writerow({'idx': ear['idx'], 'split': 'val','earSubjectIdx':earIndexDict[ear['earSubjectIdx']],'earImageName':ear['earImageName'],\
								'period':ear['period'],'earSubjectName':ear['earSubjectName'],'earLeftRight':ear['earLeftRight'],'earRotated':ear['earRotated']})
			for ear in testEarList:
				writer.writerow({'idx': ear['idx'], 'split': 'test','earSubjectIdx':earIndexDict[ear['earSubjectIdx']],'earImageName':ear['earImageName'],\
								'period':ear['period'],'earSubjectName':ear['earSubjectName'],'earLeftRight':ear['earLeftRight'],'earRotated':ear['earRotated']})
			for ear in trainingEarList:
				writer.writerow({'idx': ear['idx'], 'split': 'train','earSubjectIdx':earIndexDict[ear['earSubjectIdx']],'earImageName':ear['earImageName'],\
								'period':ear['period'],'earSubjectName':ear['earSubjectName'],'earLeftRight':ear['earLeftRight'],'earRotated':ear['earRotated']})  
	
	def generateInfoCSVWithAgeProgression(csvPath,dataDirectory,earCSVOutFileName):
		def sortPeriodList(periodList):
			periodListOut = [0,0,0,0,0,0,0,0]
			if '6W' in periodList:
				periodListOut[0] = '6W'
			if '10W' in periodList:
				periodListOut[1] = '10W'
			if '14W' in periodList:
				periodListOut[2] = '14W'
			if '5M' in periodList:
				periodListOut[3] = '5M'
			if '6M' in periodList:
				periodListOut[4] = '6M'
			if '7M' in periodList:
				periodListOut[5] = '7M'
			if '8M' in periodList:
				periodListOut[6] = '8M'
			if '9M' in periodList:
				periodListOut[7] = '9M'
			idx = 0
			while idx<len(periodListOut):
				if periodListOut[idx] == 0:
					periodListOut.pop(idx)
				else:
					idx+=1
			return periodListOut

		def getLongPeriodsDict(earBySubjectDict):
			earByLongPeriodDict = {}
			def checkPeriodSize(subjectList):
				periods = set()
				for earIndividual in subjectList:
					period = earIndividual['period']
					if period not in periods:
						periods.add(period)
				return periods
			for earSubjectKey in earBySubjectDict.keys():
				earSubjectList = earBySubjectDict[earSubjectKey]
				periods = checkPeriodSize(earSubjectList)
				if len(periods)>=4:
					earByLongPeriodDict[earSubjectKey] = earBySubjectDict[earSubjectKey]
					for earDict in earByLongPeriodDict[earSubjectKey]:
						earDict['periodList'] = sortPeriodList(list(periods))
			return earByLongPeriodDict

		earDataPath = dataDirectory+csvPath
		earBySubjectDict,subjectCount,periodSet = getEarDictBySubjectIdx(earDataPath,True)
		PERIODSETCORRECT = {'6W','10W','14W','5M','6M','7M','8M','9M'}

		earByLongPeriodDict = getLongPeriodsDict(earBySubjectDict)
		validationEarList = []
		testEarList = []
		trainingEarList = []

		for earSubjectKey in earByLongPeriodDict.keys():
			for earIndividual in earByLongPeriodDict[earSubjectKey]:
				periodList = earIndividual['periodList']
				validationEarPeriod = earIndividual['periodList'][-2]
				testEarPeriod = earIndividual['periodList'][-1]
				if(earIndividual['period'] == validationEarPeriod):
					validationEarList.append(earIndividual)
				elif(earIndividual['period'] == testEarPeriod):
					testEarList.append(earIndividual)
				else:
					trainingEarList.append(earIndividual)

		earIndexDict = changeSubjectKeyIndexes(trainingEarList)

		with open(earCSVOutFileName, 'w', newline='') as csvfile:
			fieldnames = ['idx', 'split','earSubjectIdx','earImageName','period','earSubjectName','earLeftRight','earRotated']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()
			for ear in validationEarList:
				writer.writerow({'idx': ear['idx'], 'split': 'val','earSubjectIdx':earIndexDict[ear['earSubjectIdx']],'earImageName':ear['earImageName'],\
					'period':ear['period'],'earSubjectName':ear['earSubjectName'],'earLeftRight':ear['earLeftRight'],'earRotated':ear['earRotated']})
			for ear in testEarList:
				writer.writerow({'idx': ear['idx'], 'split': 'test','earSubjectIdx':earIndexDict[ear['earSubjectIdx']],'earImageName':ear['earImageName'],\
					'period':ear['period'],'earSubjectName':ear['earSubjectName'],'earLeftRight':ear['earLeftRight'],'earRotated':ear['earRotated']})
			for ear in trainingEarList:
				writer.writerow({'idx': ear['idx'], 'split': 'train','earSubjectIdx':earIndexDict[ear['earSubjectIdx']],'earImageName':ear['earImageName'],\
					'period':ear['period'],'earSubjectName':ear['earSubjectName'],'earLeftRight':ear['earLeftRight'],'earRotated':ear['earRotated']})  

	earDataPath = dir+"annotations.csv"
	# print("earDataPath",earDataPath)
	earBySubjectDict,subjectCount,periodSet = getEarDictBySubjectIdx(earDataPath,True)
	generateInfoCSVNoAgeProgression("annotations.csv",dir,writeCSVInfoPath+"NoAgeProgression.csv")
	generateInfoCSVWithAgeProgression("annotations.csv",dir,writeCSVInfoPath+"WithAgeProgression.csv")

	return earBySubjectDict

def organizeDatasetAWE(dir,writeCSVInfoPath):
	# need directory '/awe/'	
	def buildEarDict(dirs,dir = dir):
		earDict = {}
		earSubjectDict = {}
		for dirName in dirs:
			if dirName.isdigit():

				imageDirs = os.listdir(dir+dirName)

				with open(dir+dirName+'/annotations.json', 'r') as openFile:
					# Reading from json file
					annotations = json.load(openFile)
				gender = annotations['gender']
				ethnicity = annotations['ethnicity']
				subjectIDList = []
				for fileName in imageDirs:
					
					if fileName.endswith('.png'):

						filePath = dir+dirName+'/'+fileName
						earName = fileName.split('.')[0]
						x = annotations['data'][earName]['x']
						y = annotations['data'][earName]['y']
						leftOrRight = 'left' if annotations['data'][earName]['d'] == 'l' else 'right'
						w = annotations['data'][earName]['w']
						h = annotations['data'][earName]['h']
						accessories = False if int(annotations['data'][earName]['accessories']) == 0 else True
						overlap = False if int(annotations['data'][earName]['overlap']) == 0 else True
						hPitch = annotations['data'][earName]['hPitch']
						hYaw = annotations['data'][earName]['hYaw']
						hRoll = annotations['data'][earName]['hRoll']

						singleEarDict = {
										'imagePath': filePath,\
										'x':x,\
										'y':y,\
										'leftOrRight':leftOrRight,\
										'w':w,\
										'h':h,\
										'accessories':accessories,\
										'overlap':overlap,\
										'hPitch':hPitch,\
										'hYaw':hYaw,\
										'hRoll':hRoll,\
										'subject':int(dirName),\
										}

						hashToken = hash(filePath)
						hashToken += sys.maxsize + 1
						assert str(hashToken) not in earDict, "hash conflict: {}".format(str(hashToken))
						earDict[str(hashToken)] = singleEarDict
						subjectIDList.append(str(hashToken))

				earSubjectDict[int(dirName)] = subjectIDList

		return earDict,earSubjectDict

	def decideTrainValOrTest(dir = dir):

		trainFile = dir+"train.txt"
		testFile = dir + "test.txt"

		with open(trainFile) as f:
			trainSubjects = []
			contents = f.read().split('\n')
			for content in contents:
				trainSubjects+= [int(idx) for idx in content.split(' ')]
			# fileList = contents.split(' ')
		
		with open(testFile) as f:
			testSubjects = []
			contents = f.read().split('\n')
			for content in contents:
				testSubjects+= [int(idx) for idx in content.split(' ')]
		
		return trainSubjects,testSubjects

	dirs = os.listdir(dir)
	earDict,earSubjectDict =  buildEarDict(dirs)
	trainSubjects,testSubjects = decideTrainValOrTest()

	for k in earSubjectDict.keys():
		earList = earSubjectDict[k]
		random.shuffle(earList)

		trainEars = earList[:6]
		for trainEar in trainEars:
			earDict[trainEar]['split'] = 'train'

		valEars = earList[6:8]
		for valEar in valEars:
			earDict[valEar]['split'] = 'val'

		testEars = earList[8:]
		for testEar in testEars:
			earDict[testEar]['split'] = 'test'
		# print(len(earSubjectDict[k]))

	with open(writeCSVInfoPath, 'w', newline='') as csvfile:
		fieldnames = ['earID','imagePath', 'x','y','leftOrRight','w','h','accessories','overlap','hPitch','hYaw','hRoll','subject','split']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()
		count = 0
		for k in earDict.keys():
			writer.writerow({'earID': count, \
							 'imagePath':earDict[k]['imagePath'], \
							 'x': earDict[k]['x'], \
							 'y':earDict[k]['y'], \
							 'leftOrRight': earDict[k]['leftOrRight'], \
							 'w': earDict[k]['w'], \
							 'h': earDict[k]['h'], \
							 'accessories': earDict[k]['accessories'], \
							 'overlap': earDict[k]['overlap'], \
							 'hPitch': earDict[k]['hPitch'], \
							 'hYaw': earDict[k]['hYaw'], \
							 'hRoll': earDict[k]['hRoll'], \
							 'subject': earDict[k]['subject'], \
							 'split': earDict[k]['split'],
							 })
			count+=1

	return earDict


def organizeDatasetFGNET(dir,writeCSVInfoPath):
	
	# need directory 'FGNET/images/'

	def getJpgOnly(fileNameList):
		jpgFiles = [jpg for jpg in fileNameList if jpg.endswith('.JPG')]
		return jpgFiles

	def extractInfoFromFileName(jpgName):
		personID = jpgName.split('A')[0]
		personAge = jpgName.split('A')[1].split('.')[0]
		personAge = re.sub('[a-z]','',personAge)
		return int(personID),int(personAge)

	dirs = os.listdir(dir)
	jpgs = getJpgOnly(dirs)
	
	# ./images folder: all human face images. The groundtruth is used to name each image. 
	# For example, 078A11.JPG, means that this is the No.'78' person's image when he/she was 11 years old. 
	# 'A' is short for Age.

	jpgDict = {}
	personDict = {}


	for i,jpgName in enumerate(jpgs):
		personID,personAge = extractInfoFromFileName(jpgName)
		jpgDict[jpgName] = {'idx':i,'personID':personID,'personAge':personAge}
		if personID not in personDict:
			personDict[personID] = [{'idx':i,'personAge':personAge,'jpgName':jpgName}]
		else:
			personDict[personID].append({'idx':i,'personAge':personAge,'jpgName':jpgName})
			
	# print(personDict)

	# train-testing split 1: age progression
	for k in personDict:
		testSize = len(personDict[k])//3
		trainSize = len(personDict[k]) - testSize


		while personDict[k][trainSize-1]['personAge'] == personDict[k][trainSize]['personAge']:
			trainSize -=1
			# print("k {} triggered {}".format(k,trainSize))


		for earID in personDict[k][:trainSize]: #train
			jpgDict[earID['jpgName']]['split'] = 'train'
		for earID in personDict[k][trainSize:]: #test
			jpgDict[earID['jpgName']]['split'] = 'test'

	with open(writeCSVInfoPath+'WithAgeProgression.csv', 'w', newline='') as csvfile:
		fieldnames = ['idx', 'jpgName','personID','personAge','split']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		count = 0
		for k in jpgDict.keys():
			writer.writerow({'idx': count, 'jpgName': k, 'personID': jpgDict[k]['personID'],\
							'personAge': jpgDict[k]['personAge'],'split':jpgDict[k]['split']})
			count+=1

	# train-testing split 2: no age progression

	for k in personDict:
		random.shuffle(personDict[k])
		testSize = len(personDict[k])//4
		valSize = len(personDict[k])//4
		trainSize = len(personDict[k]) - testSize - valSize
		assert trainSize >= testSize - valSize, \
			"train size: {}, val size: {} and test size: {}".format(trainSize,valSize,testSize)

		for earID in personDict[k][:trainSize]: #train
			jpgDict[earID['jpgName']]['split'] = 'train'
		for earID in personDict[k][trainSize:trainSize+valSize]: #val
			jpgDict[earID['jpgName']]['split'] = 'val'
		for earID in personDict[k][trainSize+valSize:]: #test
			jpgDict[earID['jpgName']]['split'] = 'test'
		

	with open(writeCSVInfoPath+'NoAgeProgression.csv', 'w', newline='') as csvfile:
		fieldnames = ['idx', 'jpgName','personID','personAge','split']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		count = 0
		for k in jpgDict.keys():
			writer.writerow({'idx': count, 'jpgName': k, 'personID': jpgDict[k]['personID'],\
							'personAge': jpgDict[k]['personAge'],'split':jpgDict[k]['split']})
			count+=1


	return jpgDict


jpgFGNETDict = organizeDatasetFGNET(FGNET_DIR,writeCSVInfoPath=FGNET_DATA_INFO_PATH)
# earICZDict = organizeDatasetICZ(ICZ_DIR,ICZ_DATA_INFO_DIR)
# earAWEDict = organizeDatasetAWE(AWE_DIR,writeCSVInfoPath=AWE_DATA_INFO_PATH)


