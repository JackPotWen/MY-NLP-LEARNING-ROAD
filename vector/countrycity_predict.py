import pickle
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

data = pd.read_csv('capitals.txt',delimiter=' ')
data.columns=['city1','country1','city2','country2']
word_embeddings=pickle.load(open('./word_embeddings_subset.p','rb'))
f = open('D:/training/w3vector/capitals.txt', 'r').read()
set_words = set(nltk.word_tokenize(f))
select_words = words = ['king', 'queen', 'oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']
for w in select_words:
    set_words.add(w)

def get_word_embeddings(embeddings):

    word_embeddings = {}
    for word in embeddings.vocab:
        if word in set_words:
            word_embeddings[word] = embeddings[word]
    return word_embeddings


def cosine_similarity(A, B):
    dot = np.dot(A, B)

    norma = np.sqrt((np.dot(A, A)))
    normb = np.sqrt((np.dot(B, B)))
    cos = dot / (norma * normb)

    return cos


def euclidean(A,B):
    return np.linalg.norm(A-B)


def get_country(city1, country1, city2, embeddings):
    group = set((city1, country1, city2))
    city1_emb = word_embeddings[city1]
    country1_emb = word_embeddings[country1]
    city2_emb = word_embeddings[city2]
    vec = country1_emb - city1_emb + city2_emb
    similarity = -1
    country = ""
    for word in embeddings.keys():
        if word not in group:
            word_emb = word_embeddings[word]
            cur_similarity = cosine_similarity(vec, word_emb)
            if cur_similarity > similarity:
                similarity = cur_similarity
                country = (word, similarity)
    return country


def get_accuracy(word_embedding, data):
    num_correct = 0

    for i, row in data.iterrows():
        city1 = row['city1']
        country1 = row['country1']
        city2 = row['city2']
        country2 = row['country2']

        # get_country function to find the predicted country2
        predicted_country2, _ = get_country(city1, country1, city2, word_embedding)

        if predicted_country2 == country2:
            num_correct += 1

    accuracy = num_correct / len(data)

    return accuracy


print(get_country('Athens', 'Greece', 'Cairo', word_embeddings))
accuracy = get_accuracy(word_embeddings, data)
print(f"Accuracy is {accuracy:.2f}")
"""
# Testing your function
word_embeddings=pickle.load(open('./word_embeddings_subset.p','rb'))
print(len(word_embeddings))
pickle.dump( word_embeddings, open( "word_embeddings_subset.p", "wb" ) )
print("dimension: {}".format(word_embeddings['Spain'].shape[0]))
"""