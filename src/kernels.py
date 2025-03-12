import numpy as np
from tqdm import tqdm
from itertools import product
from collections import Counter
from functools import lru_cache

#----------------------------------------------
#   KERNEL SPECTRUM 
#----------------------------------------------

class KernelSpectrum():
    def __init__(self, k=3, verbose=False):
        self.k=k
        self.verbose = verbose
        
    def k_value(self,x1,x2, verbose=False):
        """Compute K(x,y)

        Args:
            x1 (_type_): string, or array of one string
            x2 (_type_): string, or array of one string

        Raises:
            NameError: if not string in inputs

        Returns:
            _type_: kernel_spectrum(x1,x2)
        """
        if isinstance(x1, str) is False or isinstance(x2, str) is False:
            raise NameError('Can not compute a kernel on data not string')
        
        if verbose is True:
            print(f'Computing kernel between x1={x1} and x2={x2}, longueur strings : k = {self.k}')
            
        x1_kuplets = [ x1[i:i+self.k] for i in range(len(x1)-self.k+1) ]
        if verbose is True:
            print(f'liste brute des x1_kuplets={x1_kuplets}')
        c1 = Counter(x1_kuplets)
        if verbose is True:
            print(f'liste nette des k-uplets x1 avec comptes ={c1}')
            
        x2_kuplets = [ x2[i:i+self.k] for i in range(len(x2)-self.k+1) ]
        if verbose is True:
            print(f'liste brute des x2_kuplets={x2_kuplets}')
        c2 = Counter(x2_kuplets)
        if verbose is True:
            print(f'liste nette des k-uplets x2 avec comptes ={c2}')

        kernel = sum(count * c2.get(uplet, 0) for uplet, count in c1.items())

        return kernel
    
    def k_matrix(self, xs, ys):
        """compute and return Gram matrix K(x_i, y_j) 
        for i in range(xs), j in range(xs)

        Args:
            xs (_type_): array of strings
            ys (_type_): array of strings
        """
        x_data = np.array(xs) if isinstance(xs, list) else xs
        y_data = np.array(ys) if isinstance(ys, list) else ys

        if not isinstance(x_data, np.ndarray) or not isinstance(y_data, np.ndarray):
            raise NameError('Cannot compute Gram matrix - input is not an array')

        nx, ny = x_data.shape[0], y_data.shape[0]
        gram = np.zeros((nx, ny))
        
        for i in tqdm(range(nx)):
            x_i = x_data[i]
            for j in range(ny):
                y_j = y_data[j]
                gram[i,j] = self.k_value(x_i, y_j)
    
        return gram

#----------------------------------------
# MISMATCH
#----------------------------------------

class KernelMismatch():
    def __init__(self, k=3, m=1):
        self.k=k
        self.m=m
        self.dna_alphabet = ['A','G','C','T']

        # create list of k-uplets for faster computation
        # product create an iterator of tuples of cartesian products
        iter_tuples = product(self.dna_alphabet, repeat=self.k)
        # change from tuples of k characters to strings
        self.all_kuplets = [''.join(t) for t in iter_tuples]
        
    def _mismatches(self, kuplet):
        """Compute all possible mismatches of k-uplet, with at most m mismatches

        Args:
            kuplet (string): input k-uplet
            m (int, optional): maximum number of allowed mismatches. Defaults to 1.
        """
        mismatches_kuplets = []
        
        for alphabet_kuplet in self.all_kuplets:
            nb_mismatches = sum(kuplet[i] != alphabet_kuplet[i] for i in range(self.k))
            if nb_mismatches <= self.m:
                mismatches_kuplets.append(alphabet_kuplet)
        
        return mismatches_kuplets

    def k_value(self, x1, x2):
        """Compute K(x,y)

        Args:
            x1 (_type_): string, or array of one string
            x2 (_type_): string, or array of one string

        Raises:
            NameError: if not string in inputs

        Returns:
            _type_: kernel_spectrum(x1,x2)
        """
        if not isinstance(x1, str) or not isinstance(x2, str):
            raise NameError('Can not compute a kernel on data not string')
        
        # Extract k-mers and their mismatches
        c1 = Counter(mismatch for i in range(len(x1) - self.k + 1) for mismatch in self._mismatches(x1[i:i+self.k]))
        c2 = Counter(mismatch for i in range(len(x2) - self.k + 1) for mismatch in self._mismatches(x2[i:i+self.k]))

        return sum(count * c2.get(uplet, 0) for uplet, count in c1.items())

    def k_matrix(self, xs, ys):
        """compute and return Gram matrix K(x_i, y_j) 
        for i in range(xs), j in range(xs)

        Args:
            xs (_type_): array of strings
            ys (_type_): array of strings
        """
        x_data = np.array(xs) if isinstance(xs, list) else xs
        y_data = np.array(ys) if isinstance(ys, list) else ys

        if not isinstance(x_data, np.ndarray) or not isinstance(y_data, np.ndarray):
            raise NameError('Cannot compute Gram matrix - input is not an array')

        nx, ny = x_data.shape[0], y_data.shape[0]
        gram = np.zeros((nx, ny))
        
        for i in tqdm(range(nx)):
            x_i = x_data[i]
            for j in range(ny):
                y_j = y_data[j]
                gram[i,j] = self.k_value(x_i, y_j)
    
        return gram


class FastKernelMismatch():
    def __init__(self, k=3, m=1):
        self.k=k
        self.m=m
        self.dna_alphabet = ['A','G','C','T']
        
    @lru_cache(None)
    def _mismatches(self, kuplet):
        """Generate all k-mers within m mismatches from the given kuplet."""
        def generate(kmer, mismatches_left, index):
            if mismatches_left == 0 or index == len(kmer):
                return {kmer}  # No more changes allowed

            mismatch_set = generate(kmer, mismatches_left, index + 1)  # Continue without change

            # Introduce mismatches
            for letter in self.dna_alphabet:
                if letter != kmer[index]:  # Only modify if different
                    new_kmer = kmer[:index] + letter + kmer[index + 1:]
                    mismatch_set |= generate(new_kmer, mismatches_left - 1, index + 1)

            return mismatch_set

        return generate(kuplet, self.m, 0)  # Start recursive generation

    def k_value(self, x1, x2):
        """Compute K(x,y)

        Args:
            x1 (_type_): string, or array of one string
            x2 (_type_): string, or array of one string

        Raises:
            NameError: if not string in inputs

        Returns:
            _type_: kernel_spectrum(x1,x2)
        """
        if not isinstance(x1, str) or not isinstance(x2, str):
            raise NameError('Can not compute a kernel on data not string')
        
        # Extract k-mers and their mismatches
        c1 = Counter(mismatch for i in range(len(x1) - self.k + 1) for mismatch in self._mismatches(x1[i:i+self.k]))
        c2 = Counter(mismatch for i in range(len(x2) - self.k + 1) for mismatch in self._mismatches(x2[i:i+self.k]))

        return sum(count * c2.get(uplet, 0) for uplet, count in c1.items())

    
    def k_matrix(self, xs, ys):
        """compute and return Gram matrix K(x_i, y_j) 
        for i in range(xs), j in range(xs)

        Args:
            xs (_type_): array of strings
            ys (_type_): array of strings
        """
        x_data = np.array(xs) if isinstance(xs, list) else xs
        y_data = np.array(ys) if isinstance(ys, list) else ys

        if not isinstance(x_data, np.ndarray) or not isinstance(y_data, np.ndarray):
            raise NameError('Cannot compute Gram matrix - input is not an array')

        nx, ny = x_data.shape[0], y_data.shape[0]
        gram = np.zeros((nx, ny))
        
        for i in tqdm(range(nx)):
            x_i = x_data[i]
            for j in range(ny):
                y_j = y_data[j]
                gram[i,j] = self.k_value(x_i, y_j)
    
        return gram

#----------------------------------------
# SUM
#----------------------------------------

class SumKernel:
    def __init__(self, *kernels, verbose=False):
        """
        Initialize with multiple kernel instances.

        Args:
            *kernels: Any number of kernel instances with a k_value method.
            verbose (bool): Whether to print additional info.
        """
        self.kernels = kernels
        self.verbose = verbose

    def k_value(self, x1, x2):
        """
        Compute the sum of kernel values from all provided kernels.

        Args:
            x1, x2: Input sequences or feature vectors.

        Returns:
            Sum of all kernel values.
        """
        return sum(kernel.k_value(x1, x2) for kernel in self.kernels)

    def k_matrix(self, xs, ys):
        """
        Compute and return Gram matrix K(x_i, y_j)
        for i in range(xs), j in range(xs).

        Args:
            xs (_type_): array of strings
            ys (_type_): array of strings

        Returns:
            Gram matrix of summed kernel values.
        """
        x_data = np.array(xs) if isinstance(xs, list) else xs
        y_data = np.array(ys) if isinstance(ys, list) else ys

        if not isinstance(x_data, np.ndarray) or not isinstance(y_data, np.ndarray):
            raise NameError('Cannot compute Gram matrix - input is not an array')

        nx, ny = x_data.shape[0], y_data.shape[0]
        gram = np.zeros((nx, ny))

        if self.verbose:
            print(f'Computing Gram matrix with {len(self.kernels)} kernels')

        for i in tqdm(range(nx)):
            x_i = x_data[i]
            for j in range(ny):
                y_j = y_data[j]
                gram[i, j] = sum(kernel.k_value(x_i, y_j) for kernel in self.kernels)

        return gram

