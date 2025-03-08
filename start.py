#--------------------------------------------------------
#
#       RUN CUSTOM KERNELS ON SCIKIT CODE
#
#       TO SEE WHICH CUSTOM KERNEL IS BEST
#
#--------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.kernels import KernelSpectrum, KernelMismatch
from sklearn.metrics import accuracy_score
import os

# Test
from sklearn.svm import SVC

# get data
current_dir = os.getcwd()

data_dir = current_dir + '/data/'

filename = data_dir + 'Xtr0.csv'
labelname = data_dir + 'Ytr0.csv'

# filename = data_dir + 'Xtr1.csv'
# labelname = data_dir + 'Ytr1.csv'

# filename = data_dir + 'Xtr2.csv'
# labelname = data_dir + 'Ytr2.csv'

with open(filename,'r') as f:
    X = pd.read_csv(f, index_col=0)
       
with open(labelname, 'r') as g:
    y = pd.read_csv(g, index_col=0)

# instantiate kernel
k = 3

# choix = 'spectrum'
choix = 'mismatch'

if choix == 'spectrum':
    ks = KernelSpectrum(k=k)
else :
    ks = KernelMismatch(k=k)

kernel = ks.k_value

# go Forest, go
clf = SVC(kernel='precomputed')

N = 200
X = X[:N]
y = y[:N]

print(f"Running model on {filename} with {N} samples, kernel {choix} avec k = {k}")

id_train = int(N * .9)
X_train = np.array(X).squeeze()[:id_train]
y_train = np.array(y).squeeze()[:id_train]
X_test = np.array(X).squeeze()[id_train:]
y_test = np.array(y).squeeze()[id_train:]

print(f"Computing Gram matrix on X_train")
gram = ks.k_matrix(X_train, X_train, verbose=True)

print(f"Fitting model")
clf.fit(gram, y_train)

print(f"Computing Gram matrix on X_test")
if choix == 'spectrum':
    gramt = ks.k_matrix(X_test, X_train)
else :
    gramt = ks.k_matrix(X_test, X_train, verbose=True)

print(f"Computing prediction and accuracy")
y_pred = clf.predict(gramt)

print(f"Accuracy = {accuracy_score(y_pred, y_test)*100:.1f}%")