# import
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd


def calc_measures(matrix, similarity_function, ref_set, n):
    sim = np.array(similarity_function(matrix[n - 1], matrix[0:(n - 1)])[0])
    topRelevant = set(sim.argsort()[-len(relevant_set):][::-1] + 1)
    intersec = topRelevant & ref_set
    precision = len(intersec) / len(topRelevant)
    recall = len(intersec) / len(ref_set)
    f_measure = 0
    if len(intersec)==0:
        f_measure = 0
    else:
        f_measure = 2 * (precision * recall) / (precision + recall)
    return [precision, recall, f_measure]


# prepare corpus
ref_corpus = []
for d in range(1400):
    f = open("./cranfield/d/" + str(d + 1) + ".txt")
    ref_corpus.append(f.read())

queries = []
for q in range(225):
    f = open("./cranfield/q/" + str(q + 1) + ".txt")
    queries.append(f.read())

relevant_documents = []
for q in range(225):
    f = open("./cranfield/r/" + str(q + 1) + ".txt")
    document = []
    for line in f:
        document.append(int(line))
    document = set(document)
    relevant_documents.append(document)

res_df = pd.DataFrame(columns=["Query", "Precision", "Recall", "F-measure", "Method"])
for i in range(0, 20):
    query = queries[i]
    corpus = ref_corpus[:]
    relevant_set = relevant_documents[i]
    corpus.append(query)

    # init pure TF
    count_vectorizer = CountVectorizer()

    tf_matrix = count_vectorizer.fit_transform(corpus)

    # init TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # prepare matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    similarity_functions = [cosine_similarity, euclidean_distances]
    matrices = [tf_matrix, tfidf_matrix]
    for matrix in matrices:
        for func in similarity_functions:
            res_df.loc[len(res_df)] = [i] + calc_measures(matrix, func, relevant_set, len(corpus)) + ["TF-IDF/Cosine"]

print(res_df)

res_df.to_csv("out.csv", sep=',')