import re
import numpy as np
import gensim.downloader as api

"""
Creates a folder with a number of .npy files containing adjacency graphs of every connections game's words cosine similarity.
"""


def extract(file_path):
    connections = []

    with open(file_path, encoding="utf8") as file:
        lines = file.readlines()
        index = False
        temp = []

        for line in lines:

            if index == False:
                index = True
                continue

            else:

                pattern = r" - (.*)"
                temp += re.search(pattern, line).group(1).split(", ")

            if len(temp) == 16:
                connections.append(temp)
                index = False
                temp = []
                
    return connections

def word_similarity(word1, word2, model):

    word1 = word1.lower()
    word2 = word2.lower()

    try:
        similarity_score = model.similarity(word1, word2)
        return similarity_score
    except KeyError as e:
        return None


def create(word_data, model):
    matrices = []  
    index = []    
    simil = None

    for words in word_data:
        matrix = []
        for target in words:
            temp = []
            for word in words:
                simil = word_similarity(target, word, model)
                if simil is None:
                    break
                temp.append(float(simil))

            if simil is None:
                break
            matrix.append(temp)

        if simil is not None:
            matrices.append(np.array(matrix))
            index.append(words)  


    np.save(MODEL_NAME + "/data.npy", np.array(matrices))
    np.save(MODEL_NAME + "/word_data.npy", np.array(index))



def clean(word_data, model):

    for words in word_data:

        for target in words:

            target = target.lower()

            try:
                similarity_score = model.similarity(target, target)
            except KeyError as e:
                print(f"{e}")


MODEL_NAME = "fasttext"
data = extract("extract/full_words.txt")
model = api.load("fasttext-wiki-news-subwords-300")
create(data, model)
