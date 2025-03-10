import numpy as np
from tqdm import tqdm
from itertools import product
from collections import Counter

#----------------------------------------------
#   KERNEL SPECTRUM 
#----------------------------------------------

class KernelSpectrum():
    
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
            for i in range(nx):
                x_i = x_data[i]
                for j in range(ny):
                    y_j = y_data[j]
                    gram[i,j] = self.k_value(x_i, y_j, verbose=False)
                    
        # if self.verbose is True:
        #     print(f'Gram matrix computed : {gram}')
    
        return gram

#----------------------------------------
# MISMATCH
#----------------------------------------

class KernelMismatch():
    
    dna_alphabet = ['A','G','C','T']
    
    def __init__(self,k=None):
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
        
    def _mismatches(self, kuplet, mismatches=1):
        """Compute all possible mismatches of k-uplet, with at most m mismatches

        Args:
            kuplet (string): input k-uplet
            m (int, optional): maximum number of allowed mismatches. Defaults to 1.
        """
        mismatches_kuplets = []
        
        for alphabet_kuplet in self.all_kuplets:
            nb_mismatches = np.sum([kuplet[i] != alphabet_kuplet[i] for i in range(self.k)])
            if nb_mismatches <= mismatches:
                mismatches_kuplets.append(alphabet_kuplet)
        
        return mismatches_kuplets
        
    def k_value(self,x1,x2, mismatches=1, verbose=False):
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
        # compute dictionnary of unique kuplets in x1 with number of occurences
        c1 = Counter()
        for uplet in x1_kuplets:
            c1[uplet] += 1
            
        # list all k-uplets in x2
        x2_kuplets = [ x2[i:i+self.k] for i in range(len(x2)-self.k+1) ]
        # compute dictionnary of unique kuplets in x2 with number of occurences
        c2 = Counter()
        for uplet in x2_kuplets:
            c2[uplet] += 1
        
        kernel = 0
        # loop over unique kuplets in x1
        for uplet, occurences in c1.items():
            # what are all possible mismatches of this kuplet
            mismatches_kuplet = self._mismatches(uplet, mismatches=mismatches)
            for mismatch in mismatches_kuplet:
                # how many times does this mismatched kuplet appear in x2
                occurences_in_x2 = c2.get(mismatch, 0)
                kernel += occurences * occurences_in_x2
                if occurences_in_x2 > 0 and verbose is True:
                    print(f"uplet in x1 = {uplet}, mismatch in x1 occuring in x2 = {mismatch}, number of occurences_in_x2 = {occurences_in_x2}")
                
        return kernel
    
    def k_matrix(self, xs, ys, verbose=False):
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
        
        if verbose is True:
            for i in tqdm(range(nx)):
                x_i = x_data[i]
                for j in range(ny):
                    y_j = y_data[j]
                    gram[i,j] = self.k_value(x_i, y_j)
        else:
            for i in range(nx):
                x_i = x_data[i]
                for j in range(ny):
                    y_j = y_data[j]
                    gram[i,j] = self.k_value(x_i, y_j)
    
        return gram
        

#----------------------------------------
# SUBSTRING
#----------------------------------------

class KernelSubstring():
    pass
