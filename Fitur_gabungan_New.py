# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 

def weight_tfidf_vector(model, data_tfidf, bow, output_path):
    data = data_tfidf.values                 # akses data baris ke 1-end dan kolom ke 1-end
    out_gabungan = []
    panjang_kolom = len(model['manusia'])   # buat ambil panjang vectornya aja
    for a, row in enumerate(data):
        arr_feature = np.zeros((1,panjang_kolom), dtype=float)
        for b, token in enumerate(row):
            if bow[b] in model.vocab:
                mat_token = np.array(np.mat(token))
                mat_vector = np.array([model[bow[b]]])
                # looping untuk matriks 1x1 
                for i in range(len(mat_token)):
                    # looping untuk panjangnya matriks vector dari kata sekarang (100)
                    for j in range(len(mat_vector[0])):
                        # looping untuk semua fitur
                        for k in range(len(mat_vector)):
                            arr_feature[i][j] += mat_token[i][k] * mat_vector[k][j]
        data_row = list(data[a])
        feature_row = list(arr_feature[0])
        data_row.extend(feature_row)
        out_gabungan.append(data_row)
    df = pd.DataFrame(out_gabungan)
    df.to_csv(output_path, index=False)
    return df

def load_weight_gabungan(tfidf_path, bow, model, output_path):
    data = pd.read_csv(tfidf_path)
    feature_gabungan = weight_tfidf_vector(model, data, bow, output_path)
    return feature_gabungan
