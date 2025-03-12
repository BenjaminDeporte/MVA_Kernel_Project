import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from methods import KernelSVCLilian, KernelSVCBen
from kernels import KernelSpectrum, KernelMismatch
import pickle
import timeit

def recast_y(y):
    return 2*y-1

def test_algo(k, choix, verbose):
    
        # get training files -------------------------------------------
        current_dir = os.getcwd()
        data_dir = current_dir + '/data/'

        filename = data_dir + 'Xtr0.csv'
        labelname = data_dir + 'Ytr0.csv'

        # filename = data_dir + 'Xtr1.csv'
        # labelname = data_dir + 'Ytr1.csv'

        # filename = data_dir + 'Xtr2.csv'
        # labelname = data_dir + 'Ytr2.csv'

        X = np.array(pd.read_csv(filename, index_col=0)).squeeze()
        Y = np.array(pd.read_csv(labelname, index_col=0)).squeeze()
              
        # get subset
        N = 200
        X = X[:N]
        Y = Y[:N]
        
        # split
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
        
        # recast targets into [-1,1]
        Y_train = recast_y(Y_train)
        Y_val = recast_y(Y_val)
        
        # instantiate kernel
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
        print(f"Prédictions scikit = {np.unique(y_pred, return_counts=True)}")
        
        #---------- algo maison -----------------------------------
        
        kernel = ks.k_matrix
        
        clf_maison = KernelSVCLilian(C=1.0, kernel=kernel)
        print(f"Running maison model on {filename} with {N} samples, kernel {choix} avec k = {k}")
        
        print(f"Fitting modèle maison")
        clf_maison.fit(X_train, Y_train)
        
        print(f"Computing prediction and accuracy")
        y_pred_maison = clf_maison.predict(X_val)
        
        print(f"Accuracy = {accuracy_score(y_pred_maison, Y_val)*100:.1f}%")
        print(f"Prédictions maison = {np.unique(y_pred_maison, return_counts=True)}")
        
    
if __name__ == '__main__':
    choix = 'spectrum'
    verbose = True
    
    start = timeit.default_timer()
    results = grid_search()
    end = timeit.default_timer()
    print(f"Temps d'exécution : {end-start:.1f} secondes")
    
    results_file = os.getcwd() + "/rapport/liste_resultats.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results_file, f)
        
    # print(results)
    # for k in [3]:
    #     print(f"---------------------------------------------------------------------")
    #     test_algo(k, choix, verbose) 
