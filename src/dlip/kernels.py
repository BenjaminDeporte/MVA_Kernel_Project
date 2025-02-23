# file for Kernels classes
import numpy as np

# DNA alphabet
dna_alphabet = ['A','G','C','T']

# import for cartesian product
from itertools import product
# import counter for counting uplets
from collections import Counter

#----------------------------------------------
#   KERNEL SPECTRUM 
#----------------------------------------------

class KernelSpectrum():
    
    dna_alphabet = ['A','G','C','T']
    
    def __init__(self,k=None):
        # default value for k
        if k is None:
            self.k=3
        else:
            self.k=k
        # create list of k-uplets for faster computation
        # product create an iterator of tuples of cartesian products
        iter_tuples = product(self.dna_alphabet, repeat=k)
        # change from tuples of k characters to strings
        self.all_kuplets = [ ''.join(t) for t in iter_tuples]
        
    def k_value(self,x1,x2):
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
            
        # list all k-uplets in x1
        x1_kuplets = [ x1[i:i+self.k] for i in range(len(x1)-self.k+1) ]
        c1 = Counter()
        for uplet in x1_kuplets:
            c1[uplet] += 1
        # list all k-uplets in x2
        x2_kuplets = [ x2[i:i+self.k] for i in range(len(x2)-self.k+1) ]
        c2 = Counter()
        for uplet in x2_kuplets:
            c2[uplet] += 1
        # compute kernel value
        kernel = 0
        for uplet, occurences_in_x1 in c1.items():
            occurences_in_x2 = c2.get(uplet, 0)
            kernel += occurences_in_x1 * occurences_in_x2
            
        return kernel
    
    def k_matrix(self, xs, ys):
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
        
        for i in range(nx):
            x_i = x_data[i]
            for j in range(ny):
                y_j = y_data[j]
                gram[i,j] = self.k_value(x_i, y_j)
    
        return gram


#----------------------------------------
# MISMATCH
#----------------------------------------

class KernelMismatch():
    pass

#----------------------------------------
# SUBSTRING
#----------------------------------------

class KernelSubstring():
    pass


#------------------------------------------
# TESTS
#-----------------------------------------

def test_kernel_spectrum():
    
    # tests
    tests = [
        ['AGGCTTCGAC', 'CGGATGAGG', 1],   # common k_uplets : AGG (1,1), => k_value = 1
        ['AGGCTTCGAC', 'CCGATGAGG', 2],   # common k_uplets : AGG (1,1), CGA (1,1 )=> k_value = 2
        ['AAAAAAAAA', 'AAACGTGCAAA', 14]    # common k_uplets : AAA (7 in 1, 2 in 2) => k_value = 14
    ]
    
    ks = KernelSpectrum(k=3)
    
    for i,test in enumerate(tests):
        x1 = test[0]
        x2 = test[1]
        target = test[2]
        k_value = ks.k_value(x1,x2)
        assert k_value == target, f'Revoir code KernelSpectrum : test {i}'
        
    tests2 = [
        ['AGGCTTCGAC', 'CGGATGAGG', 'AAAAAAAAA', 'AAACGTGCAAA']
    ]
        
    for xs in tests2:
        gram = ks.k_matrix(xs,xs)
        print(gram)
        
    print(f"Tests passés avec succès")
    
if __name__ == '__main__':
    test_kernel_spectrum()