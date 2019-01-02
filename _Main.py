# -*- coding: utf-8 -*-
import time
import numpy as np
import Preprocessing as pr
import CrossValidation as cr
import Regex
import Fitur_tfidf_New as tfidf
import Fitur_vector_New as vector
import Fitur_gabungan_New as gabungan

waktu = open('output/wakturun.txt', 'w')
start_time = time.time()

# PATH DATA REAL
# data_input      = './input/hadits2k.csv' 
# data_label      = './input/label2k.csv' 
# data_clean      = './input/data_clean.csv' 
# result_word2vec = './output/hasil_word2vec.csv'
# result_tfidf    = './output/hasil_tfidf_fix.csv'
# result_gabungan = './output/hasil_gabungan.csv'
# path_model      = './input/skipgram/ws_model100.bin' # baris ini buat ganti2 model yang ingin digunakan
# path_hasil_tfidf    = './output/Report_tfidf.txt'
# path_hasil_word2vec = './output/Report_word2vec.txt'
# path_hasil_gabungan = './output/Report_gabungan.txt'
# metode_tfidf    = 'TF-IDF'
# metode_w2vec    = 'WORD2VEC'
# metode_gabungan = 'GABUNGAN'

# PATH DATA DUMMY
data_input      = './dummy/data_dummy.csv'
data_label      = './dummy/dummy_label.csv'
data_clean      = './dummy/data_clean.csv'
result_word2vec = './dummy/hasil_word2vec.csv'
result_tfidf    = './dummy/hasil_tfidf.csv'
result_gabungan = './dummy/hasil_gabungan.csv'
path_model      = './input/skipgram/ws_model100.bin' # baris ini buat ganti2 model yang ingin digunakan
path_hasil_tfidf    = './dummy/Report_tfidf.txt'
path_hasil_word2vec = './dummy/Report_word2vec.txt'
path_hasil_gabungan = './dummy/Report_gabungan.txt'
metode_tfidf    = 'TF-IDF'
metode_w2vec    = 'WORD2VEC'
metode_gabungan = 'GABUNGAN'

# --- Load Preprocessing ---
print("=== Preprocessing ===")
pr.praproses_data(data_input, data_clean)

# --- Load Kamus Kata (unique word) ---
print("=== Fitur Freq Perdoc & Alldoc ===")
fitur_onedoc, fitur_alldoc = Regex.load_fitur_postag(data_clean)

# '''
print("=== Bag Of Words ===")
bow = tfidf.bagofword(fitur_alldoc)

start_time1 = time.time()
# --- Load Feature Extraction Using TF IDF---
print("=== NEW Feature Extraction TfIdf ===")
hasil_ekstraksi_tfidf, bow = tfidf.main(fitur_onedoc, fitur_alldoc, result_tfidf)
h_loss_tfidf = cr.cross_validation(result_tfidf, data_label, path_hasil_tfidf, metode_tfidf)
waktu.write("TF-IDF " +  "--- %s seconds ---" % (time.time() - start_time1) + '\n')

start_time1 = time.time()
# --- Load Feature Extraction Using Vector ---
print("=== NEW Feature Extraction Vector ===")
model = vector.load_w2vec_model(path_model)
hasil_ekstraksi_w2vec = vector.feature_extraction(model, fitur_onedoc, result_word2vec)
h_loss_vector = cr.cross_validation(result_word2vec, data_label, path_hasil_word2vec, metode_w2vec)
waktu.write("W2VEC " +  "--- %s seconds ---" % (time.time() - start_time1) + '\n')

start_time1 = time.time()
# --- Load Feature Extraction Using TF IDF Concat Vector---
print("=== NEW Feature Extraction TfIdf & Vector ===") 
hasil_ekstraksi_gabungan = gabungan.load_weight_gabungan(result_tfidf, bow, model, result_gabungan)
h_loss_gabungan = cr.cross_validation(result_gabungan, data_label, path_hasil_gabungan, metode_gabungan)
waktu.write("W2VEC TFIDF" + "--- %s seconds ---" % (time.time() - start_time1) + '\n')

# --- Load Evaluation ---
print("=== Multilabel Evaluation ===")
print('')

print('Result Hamming Loss TFIDF            : ', h_loss_tfidf)
print('Result Hamming Loss WORD2VEC         : ', h_loss_vector)
print('Result Hamming Loss WORD2VEC+TFIDF   : ', h_loss_gabungan)

print('')
print("Time of Running Program : ", "--- %s seconds ---" % (time.time() - start_time))

waktu.write("Time of Running Program : " + str(time.time() - start_time) )
waktu.close()