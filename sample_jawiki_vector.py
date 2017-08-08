import MeCab
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from gensim.models import KeyedVectors

def tokenize(text):
    wakati = MeCab.Tagger("-O wakati")
    wakati.parse("")
    return wakati.parse(text).rstrip()

text_list = ["あらゆる現実をすべて自分のほうへねじ曲げたのだ。"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts([tokenize(t) for t in text_list])

embeddings_model = KeyedVectors.load_word2vec_format('entity_vector/entity_vector.model.bin', binary=True)

word_index = tokenizer.word_index
num_words = len(word_index)

embedding_matrix = np.zeros((num_words+1, 200))
for word, i in word_index.items():
    if word in embeddings_model.index2word:
        embedding_matrix[i] = embeddings_model[word]

print(embedding_matrix)
