__author__ = 'harshmshah'
import queue


def cosine_similarity(vector_doc1,docs_dataset) :
    """
    Checks the cosine similarity betweeen all the documents in the dataset to the given document.
        Parameters : vector_doc1 - The input document to which the similar documents need to be recommended.
                     docs_dataset - The documents dataset within which we need to recommend the the
                                   most 5 similar documents to the given document.
    """
    RANK  = 5

    # This Priority Queue maintains the rank of the top ranked  similar documents.
    # It store the items in the format of a tuple as (score, document_name)
    rank_q = queue.PriorityQueue(RANK)
    
    # Check to seee if the word contains in both the documents.
    # If it does then include its tf-idf normalized weights in the computation of cosine similarity score.

    max_score = 0
    for doc in docs_dataset.keys():
        doc_dict = docs_dataset[doc];
        sum = 0
        for words_doc1 in vector_doc1:
            if words_doc1 in doc_dict:
                dot_product = vector_doc1[words_doc1] * doc_dict[words_doc1]
                sum = sum + dot_product
        #get_similar_doc(sum,doc, rank_q)

        # Maintain a rank queue with the best 5 documents rank related to the given document
        if (not rank_q.empty()):
            smallest_elem = rank_q.get()
            if (sum > smallest_elem[0]):
                rank_q.put(sum, doc)
            else:
                rank_q.put(smallest_elem)

"""
def get_similar_doc(sum, doc, queue_q):

    smallest_elem = queue_q.get()
    if ( sum > smallest_elem):
        queue_q.put(10,'ten')
    else :
        queue_q.put(smallest_elem)

"""