def hamming_loss(label_pred, label_real, len_data, jum_label):
    count_salah = 0
    for i, pred in enumerate(label_pred):
        for j, angka in enumerate(pred):
            if label_pred[i][j] == label_real[i][j]:
                continue
            else:
                count_salah += 1
    hamming_loss = (1/len_data)*(1/jum_label)*count_salah
    return hamming_loss