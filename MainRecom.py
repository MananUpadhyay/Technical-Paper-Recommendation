import sys,queue,os,pickle
import DocumentVectors as dvec
import cosine_similarity as csim

def writeOutput(result):
	print("\n" + "RESULTS ----->" + "\n")
	while(not result.empty()):
		print(result.get())

if __name__ == '__main__':
	dataPath = sys.argv[1]
	seedDocument = sys.argv[2]
	rank = int(sys.argv[3])

	pickleTf = "./tf.pickle"
	pickleIdf = "./idf.pickle"
	reComputeTfIdfFlag = False
	
	fileToRecommend = seedDocument.split("/").pop()
	print("\nSeed File: " +fileToRecommend + "\n")

	resultQueue = queue.PriorityQueue()

	tf = {}
	idf = {}
	
	if( os.path.exists(pickleTf) ):
		tFile = open(pickleTf,'rb')
		tf = pickle.load(tFile)
		tFile.close()
		reComputeTfIdfFlag = True
	
	if( os.path.exists(pickleIdf) ):
		iFile = open(pickleIdf,'rb')
		idf = pickle.load(iFile)
		iFile.close()
		reComputeTfIdfFlag = True
	
	if( reComputeTfIdfFlag ):
		print("Computing TF-IDF scores...")

	if(not reComputeTfIdfFlag):
		print("Computing TF-IDF scores...")
		tf,idf = dvec.getPreProcData(dataPath)
		
		#dump tf dictionary;
		with open(pickleTf,'wb') as tfOutFile:
			pickle.dump(tf,tfOutFile)

		#dump idf dictionary;
		with open(pickleIdf,'wb') as idfOutFile:
			pickle.dump(idf,idfOutFile)



	docVectors = dvec.createVectors(tf,idf)
	# print(docVectors.keys())

	seedVector = docVectors[fileToRecommend]

	resultQueue = csim.cosine_similarity(fileToRecommend,seedVector,docVectors,rank)
	writeOutput(resultQueue)



