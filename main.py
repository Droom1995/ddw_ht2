# import
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd


def calc_cosine_measures(matrix, ref_set, n):
    sim = np.array(cosine_similarity(matrix[n - 1], matrix[0:(n - 1)])[0])
    topRelevant = set(sim.argsort()[-10:][::-1] + 1)
    intersec = topRelevant & ref_set
    precision = len(intersec) / len(topRelevant)
    recall = len(intersec) / len(ref_set)
    f_measure = 0
    if len(intersec) == 0:
        f_measure = 0
    else:
        f_measure = 2 * (precision * recall) / (precision + recall)
    return [precision, recall, f_measure]


def calc_euclidean_measures(matrix, ref_set, n):
    sim = np.array(euclidean_distances(matrix[n - 1], matrix[0:(n - 1)])[0])
    topRelevant = set(sim.argsort()[:10][::-1] + 1)
    intersec = topRelevant & ref_set
    precision = len(intersec) / len(topRelevant)
    recall = len(intersec) / len(ref_set)
    f_measure = 0
    if len(intersec) == 0:
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

df_arr = []
for i in range(0, 6):
    df_arr.append(pd.DataFrame(columns=["Query", "Precision", "Recall", "F-measure"]))
df_names = ["Bin-Euclid","Bin-Cosine","TF-Euclid", "TF-Cosine", "TFIDF-Euclid", "TFIDF-Cosine"]

for i in range(0, 20):
    query = queries[i]
    corpus = ref_corpus[:]
    relevant_set = relevant_documents[i]
    corpus.append(query)

    # init pure TF
    count_vectorizer = CountVectorizer()
    binary_vectorizer = CountVectorizer(binary=True)

    tf_matrix = count_vectorizer.fit_transform(corpus)
    bin_matrix = binary_vectorizer.fit_transform(corpus)
    # init TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # prepare matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    df_arr[0].loc[len(df_arr[0])] = [i] + calc_euclidean_measures(bin_matrix, relevant_set, len(corpus))
    df_arr[1].loc[len(df_arr[1])] = [i] + calc_cosine_measures(bin_matrix, relevant_set, len(corpus))
    df_arr[2].loc[len(df_arr[2])] = [i] + calc_euclidean_measures(tf_matrix, relevant_set, len(corpus))
    df_arr[3].loc[len(df_arr[3])] = [i] + calc_cosine_measures(tf_matrix, relevant_set, len(corpus))
    df_arr[4].loc[len(df_arr[4])] = [i] + calc_euclidean_measures(tfidf_matrix, relevant_set, len(corpus))
    df_arr[5].loc[len(df_arr[5])] = [i] + calc_cosine_measures(tfidf_matrix, relevant_set, len(corpus))

for i in range(0,len(df_arr)):
    df_arr[i].to_csv(df_names[i] + ".csv", sep=',')
