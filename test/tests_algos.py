import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ..src.methods import KernelSVCLilian, KernelSVCBen
from ..src.kernels import KernelSpectrum, KernelMismatch

def test_algo(k, choix, verbose):
    # Compare algo maison et clf de sklearn

    # get data ---------------------------------------------
    current_dir = os.getcwd()

    data_dir = current_dir + '/data/'

    filename = data_dir + 'Xtr0.csv'
    labelname = data_dir + 'Ytr0.csv'

    # filename = data_dir + 'Xtr1.csv'
    # labelname = data_dir + 'Ytr1.csv'

    # filename = data_dir + 'Xtr2.csv'
    # labelname = data_dir + 'Ytr2.csv'


    X = pd.read_csv(filename, index_col=0)
    Y = pd.read_csv(labelname, index_col=0)

    X = np.array(X).squeeze()
    Y = np.array(Y).squeeze()
        
    # subset -----------------------------------------------
    N = 500
    X = X[:N]
    Y = Y[:N]
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.9, random_state=42)
        
    # instantiate kernel spectrum ----------------------------


    if choix == 'spectrum':
        ks = KernelSpectrum(k=k)
    else :
        ks = KernelMismatch(k=k)

    #---------- algo scikit -----------------------------------
    clf = SVC(kernel='precomputed')

    print(f"Running scikit model on {filename} with {N} samples, kernel {choix} avec k = {k}")

    print(f"Computing Gram matrix on X_train")
    gram = ks.k_matrix(X_train, X_train, verbose=True)

    print(f"Fitting scikit model")
    clf.fit(gram, Y_train)

    print(f"Computing Gram matrix on X_test")
    if verbose is False:
        gramt = ks.k_matrix(X_val, X_train)
    else :
        gramt = ks.k_matrix(X_val, X_train, verbose=True)

    print(f"Computing prediction and accuracy")
    y_pred = clf.predict(gramt)

    print(f"Accuracy = {accuracy_score(y_pred, Y_val)*100:.1f}%")
    
    #---------- algo maison -----------------------------------
    
    kernel = ks.k_matrix
    
    clf_maison = KernelSVCBen(C=1.0, kernel=kernel)
    print(f"Running maison model on {filename} with {N} samples, kernel {choix} avec k = {k}")
    
    print(f"Fitting modèle maison")
    clf_maison.fit(X_train, Y_train)
    
    print(f"Computing prediction and accuracy")
    y_pred_maison = clf_maison.predict(X_val)
    
    print(f"Accuracy = {accuracy_score(y_pred_maison, Y_val)*100:.1f}%")
    
    
if __name__ == '__main__':
    k = 3
    choix = 'spectrum'
    verbose = True
    test_algo(k, choix, verbose) 
