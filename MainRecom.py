import DocumentVectors as dvec, cosine_similarity as csim,sys,queue


def writeOutput(result):

	while(not result.empty()):
		print((result.get())






if __name__ == '__main__':
	dataPath = sys.argv[1]
	seedDocument = sys.argv[2]
	resultQueue = queue.PriorityQueue()

	tf,idf = dvec.getPreProcData(dataPath)
	docVectors = dvec.createVectors(tf,idf)
	resultQueue = csim.cosine_similarity(seedDocument,dataPath)
	writeOutput(resultQueue)

