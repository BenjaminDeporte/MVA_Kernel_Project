from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.data_loader import load_dataset
from src.methods import KernelLR, KernelSVCLilian, KernelSVCBen
from src.kernels import KernelSpectrum, FastKernelMismatch, KernelMismatch
from src.utils import recast_y

def test_algo(nb_dataset, k, choix, verbose):
    # Compare algo maison et clf de sklearn

    # get data ---------------------------------------------
    X, Y, _ = load_dataset(nb_dataset)


    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    Y_train = recast_y(Y_train )
    Y_val = recast_y(Y_val)

    # instantiate kernel spectrum ----------------------------

    if choix == 'spectrum':
        ks = KernelSpectrum(k=k)
    else :
        ks = FastKernelMismatch(k=k, m=1)
        #ks = KernelMismatch(k=k, m=1)

    #---------- algo scikit -----------------------------------
    clf = SVC(kernel='precomputed')

    print(f"Running scikit model on dataset {nb_dataset} with 2000 samples, kernel {choix} avec k = {k}")

    print(f"Computing Gram matrix on X_train")
    gram = ks.k_matrix(X_train, X_train)

    print(f"Fitting scikit model")
    clf.fit(gram, Y_train)

    print(f"Computing Gram matrix on X_test")
    if verbose is False:
        gramt = ks.k_matrix(X_val, X_train)
    else :
        gramt = ks.k_matrix(X_val, X_train)

    print(f"Computing prediction and accuracy")
    y_pred = clf.predict(gramt) 

    print(f"Accuracy = {accuracy_score(y_pred, Y_val)*100:.1f}%")
    
    #---------- algo maison -----------------------------------
    
    kernel = ks.k_matrix
    
    #clf_maison = KernelSVCLilian(C=2.0, kernel=kernel)
    clf_maison = KernelLR(kernel=kernel, lmbda=0.01, iters=1000, tol=1.e-5)
    print(f"Running maison model on dataset {nb_dataset} with 2000 samples, kernel {choix} avec k = {k}")
    
    print(f"Fitting modèle maison")
    clf_maison.fit(X_train, Y_train)
    
    print(f"Computing prediction and accuracy")
    y_pred_maison = clf_maison.predict(X_val)
    
    print(f"Accuracy = {accuracy_score(y_pred_maison, Y_val)*100:.1f}%")
    
    
if __name__ == '__main__':
    nb_dataset = 2
    k = 7
    choix = 'spectrum'
    verbose = True
    test_algo(nb_dataset, k, choix, verbose) 
