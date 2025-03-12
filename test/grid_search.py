import timeit
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.data_loader import load_dataset
from src.methods import KernelLR, KernelSVCLilian, KernelSVCBen
from src.kernels import KernelSpectrum, KernelMismatch
from src.utils import recast_y

def grid_search():
    # Grid search
    
    # get training files -------------------------------------------
    test_ratio = 0.2
    
    # Hyperparameters
    ks = [8, 9, 10, 11]
    Cs = [0.1]
    
    id_test = 0
    
    # fichier resultats
    res_file = "./rapport/results_grid_search.txt"
    
    # overall results
    results = []
    
    def ecrit(msg):
        with open(res_file, 'a') as f:
            f.write(msg + '\n')
    
    # loop over datasets
    for k in ks:
        for C in Cs:
            
            news = f"\n k={k} - C={C} - N=2000 (train {int(2000*(1-test_ratio))}, test {int(2000*test_ratio)}) -------------------------------------------"
            print(news)
            ecrit(news)
            
            kC_results = []
            
            for i in range(3):

                X, Y, _ = load_dataset(i)
                
                # split
                X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_ratio, random_state=42)
                
                # recast targets into [-1,1]
                Y_train = recast_y(Y_train)
                Y_val = recast_y(Y_val)
                
                # instantiate kernel spectrum
                kernel = KernelSpectrum(k=k)
                
                #clf_maison = KernelSVCBen(C=C, kernel=kernel)
                clf_maison = KernelLR(kernel=kernel.k_matrix, lmbda=k, iters=1000, tol=1.e-5)
                
                res_maison = f"Running maison model - dataset {i} - "
                
                clf_maison.fit(X_train, Y_train)
                
                y_pred_maison = clf_maison.predict(X_val)
                
                res_maison += f"Accuracy = {accuracy_score(y_pred_maison, Y_val)*100:.1f}% - "
                res_maison += f"Prédictions maison = {np.unique(y_pred_maison, return_counts=True)}"
                
                print(res_maison)
                ecrit(res_maison)
                
                results.append([id_test, k, C, i, accuracy_score(y_pred_maison, Y_val)])
                
                kC_results.append(accuracy_score(y_pred_maison, Y_val))
                
            kC_results_avg = np.mean(kC_results)
            print(f"Average accuracy for k={k} and C={C} : {kC_results_avg*100:.1f}%")
            ecrit(f"Average accuracy for k={k} and C={C} : {kC_results_avg*100:.1f}%")
    return results

if __name__ == '__main__':
    start = timeit.default_timer()
    results = grid_search()
    end = timeit.default_timer()
    print(f"Temps d'exécution : {end-start:.1f} secondes")
