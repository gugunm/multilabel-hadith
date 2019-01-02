from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factori = StemmerFactory()
stemmer = factori.create_stemmer()
 
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

def praproses_data(input_path, output_path): #, stopword=stopword, stemmer=stemmer):
    arr_praproses =[]
    with open (input_path, 'r',  encoding='utf-8') as input :
        reader = input.read().split("\n")               #Akses data per baris
        for indeks in range(len(reader)):
            lowcase_word = reader[indeks].lower()       #case folding lowcase data perbaris
            stopw = stopword.remove(lowcase_word)       #uncomment jika pakai stopword removal
            stemming = stemmer.stem(stopw)              #uncomment jika pakai stemming
            tokenizer = RegexpTokenizer(r'\w+')         #remove punctuatuion
            tokens = tokenizer.tokenize(stemming)       #Tokenisasi Kalimat, tergantung proses terakhirnya, stemming atau stopword atau hanya casefolding
            output = []       
            for kata in tokens:
                output.append(kata)                     #proses stemming per-kata dalam 1 kalimat
            arr_praproses.append(output)                #tampung kalimat hasil stemm ke arr_praproses

    out = open(output_path, 'w')       #Open file .csv nampung nampung data
    for i, elemen in enumerate(arr_praproses):
        if i == len(arr_praproses)-1:
            out.write(" ".join(elemen) + '')   
        else:
            out.write(" ".join(elemen) + '\n')
    out.close()
    return output_path
