from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np
import logging
import time
import math

start_time = time.time()

# Diantara ''' dibawah Ini untuk membuat model model word2vec baru, hapus saja ''' untuk menjalankannya
'''
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('./Corpus/corpus_all.txt')     # ambil corpus di folder Corpus/
model = word2vec.Word2Vec(sentences, size=300, sg=0)

model.wv.save_word2vec_format('./input/text8.model.bin', binary=True)

print(model['manusia'])         # mengeluarkan model representasi dari kata "manusia"
'''

# Try to load vector model
'''
# Load Word2vec model
print("=== Load Word2vec ===")
model = KeyedVectors.load_word2vec_format('./input/skipgram/ws_model100.bin',  binary=True)

# print(model['manusia'])

# print(norm(model['manusia']-)
print("Waktu Running Program 3 : ", "--- %s seconds ---" % (time.time() - start_time))
hasil = [model['manusia'], model['manusia']]
# hasil = model['manusia']
for i in hasil:
    result = 0
    for j in i:
        result += math.pow(j, 2)
    print(result)
# print(model['dunia'])
# cross_validation('./Input/data_coba/try_data.csv')
'''

# VISUALISASI VEKTOR MODEL.
'''
from sklearn.manifold import TSNE
import re
import matplotlib.pyplot as plt
import pandas as pd

vocab = list(model.wv.vocab)
X = model[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
print (df)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)

plt.show()
'''

# BUILD FAST TEXT
'''
import fasttext

# CBOW model
model = fasttext.cbow('./Corpus/korpus.csv', 'model')
print (model.words) # list of words in dictionary

print (model['dunia']) # get the vector of the word 'machine'
'''