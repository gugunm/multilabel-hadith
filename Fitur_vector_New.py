# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np

def load_w2vec_model(path_model):
    model = KeyedVectors.load_word2vec_format(path_model,  binary=True)
    return model

def cek_vocab_in_model(fitur_perdoc, model):
    for i, sentence in enumerate(fitur_perdoc):
        for key in list(sentence.keys()):
            if key not in model.vocab: # cek apakah kata tersebut ada di model?
                del sentence[key]
    return fitur_perdoc

def fitur_vector(fitur_perdoc, model):
    panjang_vector = len(model['manusia'])
    matriks_w2vec = np.zeros((len(fitur_perdoc),panjang_vector), dtype=float)
    for i, sentence in enumerate(fitur_perdoc):
        # print(len(sentence), (sentence))
        arr_feature = np.zeros((1,panjang_vector), dtype=float)
        if(len(sentence) > 0):
            for key, value in sentence.items():
                # print(key)
                w_vector = model[key]
                for k in range(1):
                    for l, el in enumerate(w_vector):
                        arr_feature[k][l] += el
            for m in range(1):
                for n in range(panjang_vector):
                    arr_feature[m][n] = arr_feature[m][n]/len(sentence)
            for j, num in enumerate(arr_feature[0]):
                matriks_w2vec[i][j] = num
        else: # jika suatu dokumen tidak memiliki fitur, isi vectornya nol semua.
            for j, num in enumerate(arr_feature[0]):
                matriks_w2vec[i][j] = num
    return matriks_w2vec

def feature_extraction(model, fitur_perdoc, feature_path):
    fitur_hasil_cek = cek_vocab_in_model(fitur_perdoc, model)
    hasil_extract = fitur_vector(fitur_hasil_cek, model)
    df = pd.DataFrame(hasil_extract)
    df.to_csv(feature_path, index=False)
    return df