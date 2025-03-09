# file for Kernels classes
import numpy as np
from tqdm import tqdm

# DNA alphabet
dna_alphabet = ['A','G','C','T']

# import for cartesian product
from itertools import product
# import counter for counting uplets
from collections import Counter

#----------------------------------------------
#   KERNEL SPECTRUM FOR TEST
#----------------------------------------------

class KernelSpectrumT():
    
    dna_alphabet = ['A','G','C','T']
    
    def __init__(self,k=None,verbose=True):
        # default value for k
        if k is None:
            self.k=3
        else:
            self.k=k
        # create list of k-uplets for faster computation
        # product create an iterator of tuples of cartesian products
        iter_tuples = product(self.dna_alphabet, repeat=self.k)
        # change from tuples of k characters to strings
        self.all_kuplets = [ ''.join(t) for t in iter_tuples]
        self.verbose = verbose
        
    def k_value(self,x1,x2, verbose=True):
        """Compute K(x,y)

        Args:
            x1 (_type_): string, or array of one string
            x2 (_type_): string, or array of one string

        Raises:
            NameError: if not string in inputs

        Returns:
            _type_: kernel_spectrum(x1,x2)
        """
        # type check and recast
        if isinstance(x1, np.ndarray):
            x1 = x1.squeeze()
            x1 = x1[0]
        if isinstance(x2, np.ndarray):
            x2 = x2.squeeze()
            x2 = x2[0]
        if isinstance(x1, str) is False or isinstance(x2, str) is False:
            raise NameError('Can not compute a kernel on data not string')
        
        if verbose is True:
            print(f'Computing kernel between x1={x1} and x2={x2}, longueur strings : k = {self.k}')
            
        # list all k-uplets in x1
        x1_kuplets = [ x1[i:i+self.k] for i in range(len(x1)-self.k+1) ]
        if verbose is True:
            print(f'liste brute des x1_kuplets={x1_kuplets}')
        c1 = Counter()
        for uplet in x1_kuplets:
            c1[uplet] += 1
        if verbose is True:
            print(f'liste nette des k-uplets x1 avec comptes ={c1}')
        # list all k-uplets in x2
        x2_kuplets = [ x2[i:i+self.k] for i in range(len(x2)-self.k+1) ]
        if verbose is True:
            print(f'liste brute des x2_kuplets={x2_kuplets}')
        c2 = Counter()
        for uplet in x2_kuplets:
            c2[uplet] += 1
        if verbose is True:
            print(f'liste nette des k-uplets x2 avec comptes ={c2}')
        # compute kernel value
        kernel = 0
        for uplet, occurences_in_x1 in c1.items():
            occurences_in_x2 = c2.get(uplet, 0)
            if verbose is True:
                print(f'Test du k-uplet de x1 ={uplet}, occurences_in_x1={occurences_in_x1}, occurences_in_x2={occurences_in_x2} => kernel += {occurences_in_x1 * occurences_in_x2}')
            kernel += occurences_in_x1 * occurences_in_x2
            
        return kernel
    
    def k_matrix(self, xs, ys, verbose=True):
        """compute and return Gram matrix K(x_i, y_j) 
        for i in range(xs), j in range(xs)

        Args:
            xs (_type_): array of strings
            ys (_type_): array of strings
        """
        x_data = xs
        y_data = ys
        if isinstance(xs, list) is True:
            x_data = np.array(xs)
        if isinstance(ys, list) is True:
            y_data = np.array(ys)
                
        if isinstance(x_data, np.ndarray) is False or isinstance(y_data, np.ndarray) is False:
            raise NameError('can not compute design matrix - input is not an array')
        
        nx = x_data.shape[0]
        ny = y_data.shape[0]
        gram = np.zeros((nx, ny))
        
        if self.verbose is True:
            print(f'Computing Gram matrix between x1s={x_data} and x2s={y_data}, longueur strings : k = {self.k}')
        
        if verbose is True:
            for i in tqdm(range(nx)):
                x_i = x_data[i]
                for j in range(ny):
                    y_j = y_data[j]
                    gram[i,j] = self.k_value(x_i, y_j, verbose=False)
                    if self.verbose is True:
                        print(f'Computing kernel between x1={x_i} and x2={y_j}, kernel={gram[i,j]}')
        else:
            for i in tqdm(range(nx)):
                x_i = x_data[i]
                for j in range(ny):
                    y_j = y_data[j]
                    gram[i,j] = self.k_value(x_i, y_j, verbose=False)
                    
        # if self.verbose is True:
        #     print(f'Gram matrix computed : {gram}')
    
        return gram
    
    
#-------------------------------------------------------------------------
#   TESTS
#-------------------------------------------------------------------------

def test_kernel_spectrum():
    
    print(f"----------------------------------------------------------")
    print(f"Tests Calcul Kernel Spectrum")
    print(f"----------------------------------------------------------")
    
    # tests
    tests = [
        ['AGGCTTCGAC', 'CGGATGAGG', 1],   # common k_uplets : AGG (1,1), => k_value = 1
        ['AGGCTTCGAC', 'CCGATGAGG', 2],   # common k_uplets : AGG (1,1), CGA (1,1 )=> k_value = 2
        ['AAAAAAAAA', 'AAACGTGCAAA', 14]    # common k_uplets : AAA (7 in 1, 2 in 2) => k_value = 14
    ]
    
    ks = KernelSpectrumT(k=3)
    
    for i, (x1, x2, target_value) in enumerate(tests):
        print("\n")
        k = ks.k_value(x1, x2)
        print(f"Test {i+1} : x1={x1}, x2={x2}, target_value={target_value}, kernel calculé ={k}")
        assert k == target_value, f"Test {i+1} failed"
        assert ks.k_value(x1,x2,verbose=False) == ks.k_value(x2,x1,False), f"Kernel non symétrique"
        
    ks = KernelSpectrumT(k=4)
    for i, (x1, x2, target_value) in enumerate(tests):
        print("\n")
        k = ks.k_value(x1, x2)
        print(f"Test {i+1} : x1={x1}, x2={x2}, kernel calculé ={k}")
        assert ks.k_value(x1,x2,verbose=False) == ks.k_value(x2,x1,verbose=False), f"Kernel non symétrique"
        
    ks = KernelSpectrumT(k=2)
    for i, (x1, x2, target_value) in enumerate(tests):
        print("\n")
        k = ks.k_value(x1, x2)
        print(f"Test {i+1} : x1={x1}, x2={x2}, kernel calculé ={k}")
        assert ks.k_value(x1,x2,verbose=False) == ks.k_value(x2,x1,verbose=False), f"Kernel non symétrique"
        
    # test matrix
    xs = ['AGGCTTCGAC', 'CGGATGAGG', 'AAAAAAAAA', 'CGGATGAGG'] #, 'CCGATGAGG', 'AAACGTGCAAA']
    
    print("\n")
    print(f"----------------------------------------------------------")
    print(f"Tests Calcul Matrice Gram")
    print(f"----------------------------------------------------------")
    
    ks = KernelSpectrumT(k=3)
    gram = ks.k_matrix(xs, xs)
    print(gram)
    
    ks = KernelSpectrumT(k=4)
    gram = ks.k_matrix(xs, xs)
    print(gram)
    
    ks = KernelSpectrumT(k=2)
    gram = ks.k_matrix(xs, xs)
    print(gram)
        
if __name__ == '__main__':
    test_kernel_spectrum()