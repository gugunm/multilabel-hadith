# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math

# ambil unique words
def bagofword(fitur_alldoc):
    bow = []
    for fitur in fitur_alldoc:
        bow.append(fitur)
    return bow

# bikin table term frequency
def tf(bow, fitur_onedoc):
    tf_table = np.zeros((len(fitur_onedoc), len(bow)), dtype=int)
    for no_doc, doc in enumerate(fitur_onedoc):
        for no_fitur, fitur in enumerate(bow):
            # print(fitur)
            if(fitur in doc):
                tf_table[no_doc,no_fitur] = doc[fitur]
    return tf_table

# Calculate buat table idf
def idf(all_fitur, size_doc):
    for fitur in all_fitur:
        all_fitur[fitur] = math.log10(size_doc/all_fitur[fitur])
    return all_fitur

def tf_idf(tf, idf, bow):
    tfidf = np.zeros((len(tf), len(idf)), dtype=float)
    for i in range(len(tf)):
        for j, fitur in enumerate(bow):
            tfidf[i,j] = tf[i,j]*idf[fitur]
    return tfidf

# Save to CSV
def save_tocsv(data_array, nfile):
    df = pd.DataFrame(data_array)
    df.to_csv(nfile,  sep=',', encoding='utf-8', index=False)

# Main Program
def main(fitur_onedoc, fitur_alldoc, result_tfidf):
    bow = bagofword(fitur_alldoc)
    tfreq = tf(bow, fitur_onedoc)
    tidf = idf(fitur_alldoc, len(fitur_onedoc))
    tfidf = tf_idf(tfreq, tidf, bow)
    save_tocsv(tfidf, result_tfidf)
    return tfidf, bow
