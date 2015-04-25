import sys,queue
import DocumentVectors as dvec
import cosine_similarity as csim

def writeOutput(result):
	# print("in outpur")
	while(not result.empty()):
		print(result.get())

if __name__ == '__main__':
	dataPath = sys.argv[1]
	seedDocument = sys.argv[2]
	
	fileToRecommend = seedDocument.split("/").pop()
	print("Seed File: " +fileToRecommend + "\n")

	resultQueue = queue.PriorityQueue()

	tf,idf = dvec.getPreProcData(dataPath)
	docVectors = dvec.createVectors(tf,idf)
	# print(docVectors.keys())

	seedVector = docVectors[fileToRecommend]

	resultQueue = csim.cosine_similarity(fileToRecommend,seedVector,docVectors)
	writeOutput(resultQueue)

