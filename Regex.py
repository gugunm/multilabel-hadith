# -*- coding: utf-8 -*-
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tag import CRFTagger

ct = CRFTagger()
ct.set_model_file('./input/tagger/indonesian_tagger')

# memasukkan fitur ke dictionary
def insert_dict(kata, fitur):
    if kata not in fitur:
        fitur[kata] = 1
    else:
        fitur[kata] += 1
    return fitur

# berikan tag pada
def beri_tag(kata):
    kata_tag = ct.tag_sents([word_tokenize(kata)])
    return kata_tag

def tf_baseline(kalimat):
    fitur = {}
    clean_punct = RegexpTokenizer(r'\w+') 
    arr = [clean_punct.tokenize(kalimat.lower())]
    hasil = ct.tag_sents(arr)
    for sentence in hasil:
        for word_tag in sentence:
            if(word_tag[0] not in fitur):
                fitur[word_tag[0]] = 1
            else:
                fitur[word_tag[0]] += 1
    return fitur

# main regex postagging
def tf_regex_pos(kalimat):
    fitur = {}
    clean_punct = RegexpTokenizer(r'\w+') 
    arr = [clean_punct.tokenize(kalimat.lower())]
    hasil = ct.tag_sents(arr)
    informasi = ['siapa', 'adalah', 'niscaya', 'sesungguhnya', 'bahwa', 'apabila', 'maka', 'barangsiapa', 'barang'] # me - kan sebagai VB
    lah_in_word = ['sabiilillah','baitullah','ailah','nabiyullah','jumlah','rahimahullah','atlah','alhamdulillah','ubaidullah','walfadliilah','subhaanallah','tudalah','kitabullah', 'kepadamulah','tibalah','istilah','sejumlah','alaallah','illallah','insyaallah', 'rasulullah', 'allah', 'adalah', 'abdullah', 'tuhanmulah']

    for i, sentence in enumerate(hasil):
        for j, word_tag in enumerate(sentence):
            # informasi
            if (word_tag[0] in informasi):
                tag_of_word = word_tag[1]
                if(word_tag[0] == 'barang'):
                    word_after = sentence[j+1][0]
                    if (word_after == 'siapa'):
                        insert_dict(word_tag[0]+word_after, fitur)
                else:
                    insert_dict(word_tag[0], fitur)
            # Verb + lah
            elif (word_tag[0][-3:] == 'lah' and len(word_tag[0]) > 3): # kata NN yang akhirannya 'lah' (makanlah)
                tag_of_word = beri_tag(word_tag[0][:-3])
                # print(tag_of_word)
                if(word_tag[0] not in lah_in_word): # allah tag SC, al tag JJ
                    if(tag_of_word[0][0][1] == 'JJ' or tag_of_word[0][0][1] == 'RB' or tag_of_word[0][0][1] == 'NEG' or tag_of_word[0][0][1] == 'VB'):
                        insert_dict(word_tag[0], fitur)
            # %anjur% atau %wajib% sebagai VB
            elif ('anjur' in word_tag[0] or 'wajib' in word_tag[0]):
                tag_of_word = word_tag[1] # ambil tag dari kata itu
                if(tag_of_word == 'VB' or tag_of_word == 'NN'):
                    insert_dict(word_tag[0], fitur)
            # %hendak% kecuali dia bertag VB(menghendaki)
            elif ('hendak' in word_tag[0]):
                tag_of_word = word_tag[1]
                if(tag_of_word != 'VB'):
                    insert_dict(word_tag[0], fitur)
            # %sebaik-baik% tag 'RB' dan 'CC'
            elif ('sebaikbaik' in word_tag[0]):
                tag_of_word = word_tag[1]
                if(tag_of_word == 'CC' or tag_of_word == 'RB'):
                    insert_dict(word_tag[0], fitur)
            # taat dan mentaatiku - NN
            elif ('taat' in word_tag[0]):
                tag_of_word = word_tag[1]
                if(tag_of_word == 'NN'):
                    insert_dict(word_tag[0], fitur)
            # %larang% sbg VB or NN
            elif ('larang' in word_tag[0]):
                tag_of_word = word_tag[1]
                if(tag_of_word == 'VB' or tag_of_word == 'NN'):
                    insert_dict(word_tag[0], fitur)
            # %haram% sbg VB, NN -> SC, NN -> VB, NN -> RB, NN -> IN | VB -> haram(NN)
            elif ('haram' in word_tag[0]):
                tag_of_word = word_tag[1]
                if(j < len(sentence)-1 and tag_of_word == 'NN'):
                    tag_after = sentence[j+1][1]
                    if(tag_after == 'SC' or tag_after == 'VB' or tag_after == 'RB' or tag_after == 'IN'):
                        insert_dict(word_tag[0], fitur)
                elif(j != 0 and tag_of_word == 'NN'):
                    tag_before = sentence[j-1][0]
                    if(tag_before == 'VB'):
                        insert_dict(word_tag[0], fitur)
                elif(tag_of_word == 'VB'):
                    insert_dict(word_tag[0], fitur)
            # %jangan%, %durhaka%, membangkang
            elif ('jangan' in word_tag[0] or 'membangkang' in word_tag[0] or 'membangkangku' in word_tag[0] or 'durhaka' in word_tag[0]):
                insert_dict(word_tag[0], fitur)
            # tidak -> VB
            elif ('tidak' in word_tag[0]):
                tag_of_word = word_tag[1]
                if(j < len(sentence)-2):
                    # print(len(sentence))
                    word_after = sentence[j+1][0]
                    tag_after2 = sentence[j+2][1]
                    if(word_after == 'pernah' and tag_after2 == 'VB'):
                        insert_dict(word_tag[0], fitur)
            # me-kan sbg VB
            elif(word_tag[0][0:2] == 'me' and word_tag[0][-3:] == 'kan'):
                tag_of_word = word_tag[1]
                if(tag_of_word == 'VB'):
                    insert_dict(word_tag[0], fitur)
    return fitur

def idf_regex_pos(fitur_1doc, fitur_alldoc):
    for i, fitur in enumerate(fitur_1doc):
        if fitur not in fitur_alldoc:
            fitur_alldoc[fitur] = 1
        else:
            fitur_alldoc[fitur] += 1
    return fitur_alldoc

# Read Data
def load_fitur_postag(path_data):
    input_path = path_data      # Baca data csv
    fitur_alldoc = {}           # fitur untuk seluruh dokumen ()
    fitur_onedoc = []           # array, isinya sebanyak data yang digunakan. dan setiap isinya mengandung fitur tiap dokumen (TF)
    with open (input_path, 'r', encoding = "ISO-8859-1") as inputan :
        reader = inputan.read().split("\n")
        for row in reader:
            fitur_perdoc = tf_regex_pos(row)         # Tipe dict isinya fitur dan frekuensi per dokumen | katanya udah unik
            # fitur_perdoc = tf_baseline(row)         # Tipe dict isinya fitur dan frekuensi per dokumen | katanya udah unik
            fitur_onedoc.append(fitur_perdoc)        # dict dimasukin ke array bernama fitur_onedoc
            fitur_alldoc = idf_regex_pos(fitur_perdoc, fitur_alldoc) # fitur perdoc dimasukin ke fitur alldoc dalam bentuk dict
    return fitur_onedoc, fitur_alldoc
