import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from methods import KernelSVCLilian, KernelSVCBen
from kernels import KernelSpectrum, KernelMismatch
import pickle

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
    N = 2000
    X = X[:N]
    Y = Y[:N]
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
    
    def recast_y(y):
        return 2*y-1
    
    Y_train = recast_y(Y_train)
    Y_val = recast_y(Y_val)
        
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
    print(f"Prédictions scikit = {np.unique(y_pred, return_counts=True)}")
    
    #---------- algo maison -----------------------------------
    
    kernel = ks.k_matrix
    
    clf_maison = KernelSVCBen(C=1.0, kernel=kernel)
    print(f"Running maison model on {filename} with {N} samples, kernel {choix} avec k = {k}")
    
    print(f"Fitting modèle maison")
    clf_maison.fit(X_train, Y_train)
    
    print(f"Computing prediction and accuracy")
    y_pred_maison = clf_maison.predict(X_val)
    
    print(f"Accuracy = {accuracy_score(y_pred_maison, Y_val)*100:.1f}%")
    print(f"Prédictions maison = {np.unique(y_pred_maison, return_counts=True)}")


def recast_y(y):
    return 2*y-1






def grid_search(choix='spectrum'):
    # Grid search
    
    # get training files -------------------------------------------
    current_dir = os.getcwd()
    data_dir = current_dir + '/data/'

    filename1 = data_dir + 'Xtr0.csv'
    labelname1 = data_dir + 'Ytr0.csv'

    filename2 = data_dir + 'Xtr1.csv'
    labelname2 = data_dir + 'Ytr1.csv'

    filename3 = data_dir + 'Xtr2.csv'
    labelname3 = data_dir + 'Ytr2.csv'

    X1 = np.array(pd.read_csv(filename1, index_col=0)).squeeze()
    Y1 = np.array(pd.read_csv(labelname1, index_col=0)).squeeze()
    
    X2 = np.array(pd.read_csv(filename2, index_col=0)).squeeze()
    Y2 = np.array(pd.read_csv(labelname2, index_col=0)).squeeze()
    
    X3 = np.array(pd.read_csv(filename3, index_col=0)).squeeze()
    Y3 = np.array(pd.read_csv(labelname3, index_col=0)).squeeze()
    
    Xs = [X1, X2, X3]
    Ys = [Y1, Y2, Y3]

    # subset
    N = 2000
    test_ratio = 0.1
    
    # Hyperparameters
    ks = [7]
    Cs = [2.0]
    
    id_test = 0
    
    # Reporting général
    hyperparams = f"Hyperparameters : k = {ks} - C = {Cs} - N = {N}"
    
    # fichier resultats
    res_file = os.getcwd() + "/rapport/results_grid_search.txt"
    
    def ecrit(msg):
        with open(res_file, 'a') as f:
            f.write(msg + '\n')
    
    # loop over datasets
    for k in ks:
        for C in Cs:
            
            news = f"\n k={k} - C={C} - N={N} (train {int(N*(1-test_ratio))}, test {int(N*test_ratio)}) -------------------------------------------"
            print(news)
            ecrit(news)
            
            for i, (X, Y) in enumerate(zip(Xs,Ys)):
                
                id_test += 1
                
                # get subset
                X = X[:N]
                Y = Y[:N]
                
                # split
                X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_ratio, random_state=42)
                
                # recast targets into [-1,1]
                Y_train = recast_y(Y_train)
                Y_val = recast_y(Y_val)
                
                # instantiate kernel spectrum
                ks = KernelSpectrum(k=k)
                
                # reporting = f"Test {id_test} - k = {k} - C = {C} - dataset {i}"
                
                #---------- algo scikit -----------------------------------
                clf = SVC(kernel='precomputed')
                
                res_scikit = f"Running scikit model - dataset {i} - "
                
                # print(f"Computing Gram matrix on X_train")
                gram = ks.k_matrix(X_train, X_train, verbose=False)
                
                # print(f"Fitting scikit model")
                clf.fit(gram, Y_train)
                
                # print(f"Computing Gram matrix on X_test")
                if verbose is False:
                    gramt = ks.k_matrix(X_val, X_train)
                else :
                    gramt = ks.k_matrix(X_val, X_train, verbose=False)
                
                # print(f"Computing prediction and accuracy")
                y_pred = clf.predict(gramt)
                
                res_scikit += f"Accuracy = {accuracy_score(y_pred, Y_val)*100:.1f}% - "
                # print(f"Accuracy = {accuracy_score(y_pred, Y_val)*100:.1f}%")
                res_scikit += f"Prédictions scikit = {np.unique(y_pred, return_counts=True)}"
                
                print(res_scikit)
                ecrit(res_scikit)
                
                #---------- algo maison -----------------------------------
                
                kernel = ks.k_matrix
                
                clf_maison = KernelSVCBen(C=C, kernel=kernel)
                
                res_maison = f"Running maison model - dataset {i} - "
                # print(f"Running maison model on {filename} with {N} samples, kernel {choix} avec k = {k}")
                
                # print(f"Fitting modèle maison")
                clf_maison.fit(X_train, Y_train)
                
                # print(f"Computing prediction and accuracy")
                y_pred_maison = clf_maison.predict(X_val)
                
                res_maison += f"Accuracy = {accuracy_score(y_pred_maison, Y_val)*100:.1f}% - "
                # print(f"Accuracy = {accuracy_score(y_pred_maison, Y_val)*100:.1f}%")
                res_maison += f"Prédictions maison = {np.unique(y_pred_maison, return_counts=True)}"
                # print(f"Prédictions maison = {np.unique(y_pred_maison, return_counts=True)}")
                
                print(res_maison)
                ecrit(res_maison)
                
                # sauve modèle entrainé
                savepath = current_dir + f"/src/model_{id_test}.pkl"
                with open(savepath, 'wb') as f:
                    pickle.dump(clf_maison, f)
                print(f"sauvegarde modèle entrainé sur dataset Xtr{id_test}")
                
                
                
                
                
                
                
                
                
                
                
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
    
    grid_search()
    # for k in [3]:
    #     print(f"---------------------------------------------------------------------")
    #     test_algo(k, choix, verbose) 
