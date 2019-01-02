from sklearn import svm
#from sklearn.neighbors import KNeighborsClassifier

def klasifikasi_svm(X_train, y_train, X_test):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    clf_predictions = clf.predict(X_test)
    
    return clf_predictions


'''
def klasifikasi_data(data_tfidf, multilabel):
    data = pd.read_csv(data_tfidf)
    label = pd.read_csv(multilabel)

    anjuran = label['anjuran'].values.tolist()
    larangan = label['larangan'].values.tolist()
    informasi = label['informasi'].values.tolist()
    arr_label = [anjuran, larangan, informasi]

    label_pred = []
    label_real = []

    for label in arr_label:
        X = data.iloc[:,:].values  
        y = label

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  
        
        # SVM Prediction   
        clf = svm.SVC(kernel='linear', C=1)
        # clf = svm.SVC(kernel='rbf', C = 1.0, gamma=0.02)
        clf.fit(X_train, y_train)
        clf_predictions = clf.predict(X_test)
        
        label_real.append(y_test)
        label_pred.append(clf_predictions)
        # print(clf_predictions)
        # print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))

    return label_pred, label_real, len(data), len(arr_label)
'''