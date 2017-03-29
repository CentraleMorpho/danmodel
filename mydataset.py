import os
import random
import math


def getImagesPathsFromLabelsFile():
	posesFile = '/mnt/SSD/stagiaire/VisagePoseEstimation/Img/Cropped/024_poseonly_normalised180.txt'
	#posesFile = '../Cropped/024_poseonly_normalised180.txt'
	listPathsTrain = []
	listPathsVal = []
	
	with open(posesFile,'r') as f:
			i=0
			for line in f:
				path = line.split(' ')[0]
				if(path.split('/')[0]=='020' or path.split('/')[0]=='022' or path.split('/')[0]=='014b'):
					
					if(path.split('/')[1]=='ibug'):
						path = os.path.join("/mnt/SSD/stagiaire/VisagePoseEstimation/Img/Cropped",path)
						listPathsVal.append(path)
					elif(path.split('/')[1]=='B73') :
						path = os.path.join("/mnt/SSD/stagiaire/VisagePoseEstimation/Img/Cropped",path)
						listPathsTrain.append(path)
					#path = os.path.join("../Cropped",path)
				print(i)
				i=i+1
				
	return (listPathsTrain,listPathsVal)
	
	


if __name__=='__main__':
	print('Getting all the images paths...')
	(listPathsTrain, listPathsVal) = getImagesPathsFromLabelsFile()
	random.seed(123)
	print('Shuffling the list')
	random.shuffle(listPathsTrain)
	random.shuffle(listPathsVal)
	#nbTrainingSet = int(math.floor(len(listPaths)*90/100))
	#print('Creating the training set')
	#trainingSetPaths = listPaths[:nbTrainingSet]
	#print('Creating the validation set')
	#validationSetPaths = listPaths[nbTrainingSet+1:]
	
	print('Saving the training paths to a txt file')
	trainingSetPathsFile = open('trainingImagesPaths.txt','w')
	for item in listPathsTrain:
		trainingSetPathsFile.write("%s\n" % item)
	trainingSetPathsFile.close()

	print('Saving the validation paths to a txt file')
	validationSetPathsFile = open('validationImagesPaths.txt','w')
	for item in listPathsVal:
		validationSetPathsFile.write("%s\n" % item)
	validationSetPathsFile.close()
	print('Completed')

	
