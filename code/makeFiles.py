import sys, os
import time
from collections import defaultdict
def removeStops(essay,stopWords):
	newEssay = []
	for word in essay:
		if word not in stopWords:
			newEssay.append(word)
	return newEssay

def createStopList():
	stopWords = []
	f0 = open("stop.txt","r")
	for line in f0:
		word = line.replace("\n","")
		stopWords.append(word)
	return stopWords
	f0.close()
def createDictionaryForFile(file):
	answer = {}
	for line in file:
		for word in line.strip().split():
			if word not in answer:
				answer[word] = 0
			answer[word] += 1
	print(answer)
	for word in answer:
		print(answer[word], '\t',word)
	sortedDict = sorted(answer.items(), key=lambda item: item[1],reverse = True)
	for i in range(len(sortedDict[:20])):
		print(i)
		print(sortedDict[i])
def getAllEssays():
	dir = os.listdir("Data2014")
	allEssays = []
	for folders in dir:
		if folders != ".DS_Store":
			path = "Data2014/"+folders
			dirs = os.listdir( path )
			for currfile in dirs:
				file = path+"/"+currfile
				if "final" in file:
					currentEssay = []
					openEssay = open(file,"r")
					for line in openEssay:
						line2 = line.translate(str.maketrans('','', "(,),\',\",!,@,#,$,%,^,&,*,{,},{,},-,_,=,+,;,:,,,.,<,>,/,?,\\,|,`,~,\n"))
						splitLine = line2.split(" ")
						if splitLine != ['\n']:
							for word in splitLine:
								currentEssay.append(word.lower())
					allEssays.append(currentEssay)
					openEssay.close()
	return allEssays

def main():	
	start = time.time()

	#get List of all stop words
	stopWords = createStopList()
	#get 2d list of all words in all
	allEssays = getAllEssays()

	for i in range(len(allEssays)):
		#print(len(allEssays[i]))
		allEssays[i] = removeStops(allEssays[i],stopWords)
		#print(len(allEssays[i]))
		#print("-0----")
	for i in range(1):
		print(allEssays[i])
		createDictionaryForFile(allEssays[i])






	print("essays checked:\t",len(allEssays))
	end = time.time()
	print("Elapsed Time(seconds): ",end - start)







main()
