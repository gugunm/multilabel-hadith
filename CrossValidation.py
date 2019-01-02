import pandas as pd
import numpy as np
import Evaluation as ev
import Classification as cl
from sklearn.model_selection import KFold

def save_info_kfold(label_pred, label_real, indeks_tes, hasil_hammingloss, output_path):
    out = open(output_path, 'w')
    jeniskelas = ['ANJURAN','LARANGAN','INFORMASI']
    for n, elemen in enumerate(indeks_tes):
        out.write('# KELAS ' + jeniskelas[n]  + ' # \n')
        for k, i in enumerate(elemen):
            out.write('- Fold Ke ' + str(k+1) + ' - \n')
            out.write('Indeks Data test : ')
            for j in i:
                out.write('{} '.format(j))
            out.write('\n')
            out.write('Label True       : ')
            for l in label_real[n][k]:
                out.write('{} '.format(l))
            out.write('\n')
            out.write('Label Prediksi   : ')
            for m in label_pred[n][k]:
                out.write('{} '.format(m))
            out.write('\n')
            out.write('Hamming Loss ke-'+ str(k+1) + ': ' +  str(hasil_hammingloss[k]) + '\n')
        out.write('\n')
        n_HLoss = np.mean(hasil_hammingloss)
    out.write('====================================== \n ')
    out.write('HAMMING LOSS     : ' + '{} '.format(n_HLoss) + '\n')
    out.write('====================================== ')
    out.close()

def cross_validation(result_tfidf, data_label, output_path, nama_metode):
    df = pd.read_csv(result_tfidf)  # tabel hasil ekstraksi
    label = pd.read_csv(data_label) # tabel label sesuai data ekstraksi
    X = df.values                   # tabel ekstraksi dijadikan array biar mudah
    anjuran = label['anjuran'].values           # array label anjuran
    larangan = label['larangan'].values         # array label larangan
    informasi = label['informasi'].values       # array label informasi
    arr_label = [anjuran, larangan, informasi]  # gabungan dari 3 array label

    label_pred = [] # inisialisasi array untuk prediksi label
    label_real = [] # inisialisasi array untuk label yang sesungguhnya
    indeks_tes = []

    kf = KFold(n_splits=10, shuffle=False) # data di bagi 5 fold dan 1x perulangan
    jeniskelas = ['ANJURAN','LARANGAN','INFORMASI']

    for c, y in enumerate(arr_label): # untuk setiap label (ada length = 3)
        single_label_pred = [] # inisialisasi array lebel_pred perlabel
        single_label_real = [] # inisialisasi array lebel_real perlabel
        indeks_tesperfold = []
        count = 0
        for train_index, test_index in kf.split(X): # looping untuk setiap fold yang telah dibagi menjadi 5
            count += 1
            print('FOLD KE-', count , ' | KELAS-', jeniskelas[c], ' | METODE-', nama_metode)
            print("TEST:",test_index)
            X_train, X_test = X[train_index], X[test_index] 
            y_train, y_test = y[train_index], y[test_index]
            
            # PROSES KLASIFIKASINYA DISINI
            predictions = cl.klasifikasi_svm(X_train, y_train, X_test)
            # ============================
            
            print (predictions, ' --- ', y_test)
            single_label_pred.append(predictions)
            single_label_real.append(y_test)
            indeks_tesperfold.append(test_index)
        label_pred.append(single_label_pred)
        label_real.append(single_label_real)
        indeks_tes.append(indeks_tesperfold)

    hasil_hammingloss = []
    for i in range(kf.get_n_splits()):
        label_prediksi = []
        label_benar    = []
        for j in range(len(label_pred)):
            label_prediksi.append(label_pred[j][i])
            label_benar.append(label_real[j][i])
        # INI PERHITUNGAN HAMMING LOSSNYA
        nilai_hl = ev.hamming_loss(label_prediksi, label_benar, len(label_real[0][0]), len(arr_label))
        # ========================
        hasil_hammingloss.append(nilai_hl)
#        print('Hamming Loss K ke', i+1 ,': ' , nilai_hl)
    n_rata_hl = np.mean(hasil_hammingloss)
#    print('Hamming Loss Keseluruhan : ' , n_rata_hl)

    save_info_kfold(label_pred, label_real, indeks_tes, hasil_hammingloss, output_path)
    return n_rata_hl