# file for Kernels classes

# DNA alphabet
dna_alphabet = ['A','G','C','T']

# import for cartesian product
from itertools import product
# import counter for counting uplets
from collections import Counter

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

class KernelMismatch():
    pass

class KernelSubstring():
    pass

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
        
    print(f"Tests passés avec succès")
    
if __name__ == '__main__':
    test_kernel_spectrum()