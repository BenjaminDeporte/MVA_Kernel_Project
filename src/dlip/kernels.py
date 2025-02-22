# file for Kernels classes

# DNA alphabet
dna_alphabet = ['A','G','C','T']

# import for cartesian product
from itertools import product

class KernelSpectrum():
    
    dna_alphabet = ['A','G','C','T']
    
    def __init__(self,k=None):
        # default value for k
        if k is None:
            k=3
        # create list of k-uplets for faster computation
        # product create an iterator of tuples of cartesian products
        iter_tuples = product(self.dna_alphabet, repeat=k)
        # change from tuples of k characters to strings
        self.all_kuplets = [ ''.join(t) for t in iter_tuples]
        
    def k(self,x1,x2):
        # list all k-uplets in x1
        x1_kuplets = [ x1[i:i+self.k] for i in range(len(x1)-self.k+1) ]
        # list all k-uplets in x2
        x2_kuplets = [ x2[i:i+self.k] for i in range(len(x2)-self.k+1) ]
        # compute kernel value
        k = 0
        # for kuplet in x1_kuplets:
        #     if kuplet in x2_kuplets:
        #         k += 1
        
        

class KernelMismatch():
    pass

class KernelSubstring():
    pass