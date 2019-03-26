import sys, os
import time
from collections import defaultdict
def removeStops(essay,stopWords):
	newEssay = []
	for word in essay:
		if word not in stopWords and word != "":
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
	freq = {}
	for line in file:
		for word in line.strip().split():
			if word not in freq:
				freq[word] = 0
			freq[word] += 1
			'''if word not in fullWordList:
				fullWordList.append(word)'''
	sortedDict = sorted(freq.items(), key=lambda item: item[1],reverse = True)
	return sortedDict
def getAllEssays():
	dirrec1 = os.listdir("Data2014")
	allEssays = []
	'''for i in range(1):
		if dirrec1[i] != ".DS_Store":
			path = "Data2014/"+dirrec1[i]'''
	for folders in dirrec1:
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
	dotWord = "...................................................."


	#get List of all stop words
	statement = "create stop word list"
	print(statement+dotWord[:50-len(statement)], end =" ")
	stopWords = createStopList()
	end1 = time.time()
	print("done")
	print("Time to load stop words  (seconds):\t",round(end1 - start,15),'\telap:\t',round(end1-start,15))
	

	#get 2d list of all words in all essays
	statement = "Loading all Student Essays"
	print(statement+dotWord[:50-len(statement)], end =" ")
	allEssays = getAllEssays()
	print("done")
	end2 = time.time()
	print("Time to load all essays  (seconds):\t",round(end2 - start,15),'\telap:\t',round(end2-end1,15))
	

	#remove all of the stop words within each essay
	statement = "Remove all stop words from Essays"
	print(statement+dotWord[:50-len(statement)], end =" ")
	for i in range(len(allEssays)):
		allEssays[i] = removeStops(allEssays[i],stopWords)
	print("done")
	end3 = time.time()
	print("Time to remove stop words(seconds):\t",round(end3 - start,15),'\telap:\t',round(end3-end2,15))


	#convert each essay into a list of (word,freq) tuples
	statement = "Convert each essay to freqCounts"
	print(statement+dotWord[:50-len(statement)], end =" ")
	freqCounts = []
	for essay in allEssays:
		freqCounts.append(createDictionaryForFile(essay))
	print("done")
	end4 = time.time()
	print("Time to create freq count(seconds):\t",round(end4 - start,15),'\telap:\t',round(end4-end3,15))


	#create a list of all unique words used in the essays
	statement = "Create list of unique words"
	print(statement+dotWord[:50-len(statement)], end =" "),
	fullWordList = []
	for essay in freqCounts:
		for word in essay:
			if word[0] not in fullWordList:
				fullWordList.append(word[0])
	print("done")
	end5 = time.time()
	print("Time to find unique words(seconds):\t",round(end5 - start,15),'\telap:\t',round(end5-end4,15))






	count = 0
	for essay in freqCounts:
		for word in essay:
			count +=1
	




	print("amount of total words :\t",count)
	print("Amount of unique words:\t",len(fullWordList))
	print("essays checked        :\t",len(allEssays))
	end = time.time()
	print("Total Elapsed Time       (seconds):\t",round(end - start,15))







main()
