import sys, os
import time
import math
from collections import defaultdict
import matplotlib.pyplot as plt


#look at this and figure out how it works
from sklearn.decomposition import PCA


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np

import numpy
from scipy import sparse

fullListFreq = {}


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
			if word not in fullListFreq:
				fullListFreq[word] = 0
			fullListFreq[word] += 1
			'''if word not in fullWordList:
				fullWordList.append(word)'''
	sortedDict = sorted(freq.items(), key=lambda item: item[1],reverse = True)
	return freq
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
					lines = ""
					for line in openEssay:
						#lines +=line
						#sentenceList.append(line)
						line2 = line.translate(str.maketrans('','', "(,),\',\",!,@,#,$,%,^,&,*,{,},{,},-,_,=,+,;,:,,,.,<,>,/,?,\\,|,`,~,\n"))
						splitLine = line2.split(" ")
						if splitLine != ['\n']:
							for word in splitLine:
								currentEssay.append(word.lower())
					allEssays.append(currentEssay)
					openEssay.close()
					#sentenceList.append(lines)
	return allEssays
def tfComputation(docFreqList, document):
	tf = {}
	length = len(document)
	for word, count in docFreqList.items():
		tf[word] = count/length
	return tf
def computeTDF(essayList):
	n = len(essayList)
	idf = {}
	for l in essayList:
		for word, count in l.items():
			if count > 0:
				if word not in idf:
					idf[word] = 0
				idf[word] += 1
	
	for word, v in idf.items():
		idf[word] = math.log(n / float(v))
	return idf
def computeTFIDF(tf,idf):
    tf_idf = dict.fromkeys(idf.keys(), 0)
    for word, v in tf.items():
        tf_idf[word] = v * idf[word]
    return tf_idf.items()





def main():	
	start = time.time()
	dotWord = "...................................................."


	#get List of all stop words
	statement = "create stop word list"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	stopWords = createStopList()
	end1 = time.time()
	print("DONE\n")
	print("Time to load stop words  (seconds):\t",round(end1 - start,15),'\telap:\t',round(end1-start,15))
	

	#get 2d list of all words in all essays
	statement = "Loading all Student Essays"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	allEssays = getAllEssays()
	print("DONE\n")
	end2 = time.time()
	print("Time to load all essays  (seconds):\t",round(end2 - start,15),'\telap:\t',round(end2-end1,15))
		
	
	#remove all of the stop words within each essay
	statement = "Remove all stop words from Essays"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	for i in range(len(allEssays)):
		allEssays[i] = removeStops(allEssays[i],stopWords)
	print("DONE\n")
	end3 = time.time()
	print("Time to remove stop words(seconds):\t",round(end3 - start,15),'\telap:\t',round(end3-end2,15))

	
	#convert each essay into a list of (word,freq) tuples
	statement = "Convert each essay to freqCounts"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	freqCounts = []
	for essay in allEssays:
		freqCounts.append(createDictionaryForFile(essay))
	print("DONE\n")
	end4 = time.time()
	print("Time to create freq count(seconds):\t",round(end4 - start,15),'\telap:\t',round(end4-end3,15))

	#sortedDict = sorted(fullListFreq.items(), key=lambda item: item[1],reverse = True)



	#Get the IDF of all the terms
	statement = "Get the IDF computation"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	idf = computeTDF(freqCounts)
	print("DONE\n")
	end5 = time.time()
	print("Time to create IDF       (seconds):\t",round(end5 - start,15),'\telap:\t',round(end5-end4,15))



	#Get the IDF of all the terms
	statement = "Get the TF computations"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	tfComputations = []
	for i in range(len(allEssays)):
		tfComputations.append(tfComputation(freqCounts[i],allEssays[i]))
	print("DONE\n")
	end6 = time.time()
	print("Time to create TF's      (seconds):\t",round(end6 - start,15),'\telap:\t',round(end6-end5,15))

	#Get the TF-IDF of each document
	statement = "Get the TF-IDF computations"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	tfIDFs = []
	for tf in tfComputations:
		tfIDFs.append(computeTFIDF(tf,idf))
	print("DONE\n")
	end7 = time.time()
	print("Time to create  TF-IDF's (seconds):\t",round(end7 - start,15),'\telap:\t',round(end7-end6,15))



	#Get the full matrix
	statement = "Generate complete matrix's"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)
	matrix = []
	for essay in tfIDFs:
		docLine = []
		for word in essay:
			#print(word[1])
			docLine.append(word[1])
		matrix.append(docLine)
	docMatrixDense = numpy.matrixlib.defmatrix.matrix(matrix)
	docMatrixSparse = sparse.csr_matrix(docMatrixDense)
	print("DONE\n")
	end8 = time.time()
	print("Time to create Matrix    (seconds):\t",round(end8 - start,15),'\telap:\t',round(end8-end7,15))



	reduced_data1 = PCA(n_components=2).fit_transform(docMatrixDense)
	reduced_data = TSNE(n_components=2).fit_transform(docMatrixDense)
	K = 3
	kmeans_model = KMeans(n_clusters=8).fit(reduced_data)
	print(reduced_data)


	

	points = []
	
	#for a in reduced_data:
#		points.append((list(a)[0],list(a)[1]))
#	plt.scatter(*zip(*points))


	
	#doing kmeans
	statement = "doing kMeans"
	print(statement+dotWord[:50-len(statement)], end =" ",flush=True)

	clustering_model = KMeans(
	    n_clusters=8,
	    max_iter=100000
	)
	#labels = clustering_model.fit_predict(reduced_data)
	labels = kmeans_model.labels_
	centers = np.array(kmeans_model.cluster_centers_)
	print(labels)
	print()

	print("DONE\n")
	end9 = time.time()
	print("Time to create kMeans    (seconds):\t",round(end9 - start,15),'\telap:\t',round(end9-end8,15))


	X = docMatrixDense

	# ----------------------------------------------------------------------------------------------------------------------

	#reduced_data = PCA(n_components=2).fit_transform(X)
	# print reduced_data
	labels_color_map = {
	    0: '#c687c5', 1: '#ffbb00', 2: '#e1ff00', 3: '#94ff00', 4: '#00ffb2',
	    5: '#0090ff', 6: '#7200ff', 7: '#f600ff', 8: '#44440d', 9: '#7c8ea5'
	}
	fig, ax = plt.subplots()
	for index, instance in enumerate(reduced_data):
	    # print instance, index, labels[index]
	    pca_comp_1, pca_comp_2 = reduced_data[index]
	    color = labels_color_map[labels[index]]
	    ax.scatter(pca_comp_1, pca_comp_2, c=color, s=45)

	plt.scatter(centers[:,0], centers[:,1], marker="x", color='r',s = 100)
	'''
	# t-SNE plot
	embeddings = TSNE(n_components=tsne_num_components)
	Y = embeddings.fit_transform(X)
	plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
	plt.show()
	'''


	count = 0
	for essay in allEssays:
		count+=len(essay)
	print("amount of total words :\t",count)
	print("Amount of unique word2:\t",len(fullListFreq))
	print("essays checked        :\t",len(allEssays))
	end = time.time()
	print("Total Elapsed Time       (seconds):\t",round(end - start,15))
	plt.show()







main()
