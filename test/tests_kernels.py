from src.kernels import KernelSpectrum

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
    
    ks = KernelSpectrum(k=3)
    
    for i, (x1, x2, target_value) in enumerate(tests):
        print("\n")
        k = ks.k_value(x1, x2)
        print(f"Test {i+1} : x1={x1}, x2={x2}, target_value={target_value}, kernel calculé ={k}")
        assert k == target_value, f"Test {i+1} failed"
        assert ks.k_value(x1,x2,verbose=False) == ks.k_value(x2,x1,False), f"Kernel non symétrique"
        
    ks = KernelSpectrum(k=4)
    for i, (x1, x2, target_value) in enumerate(tests):
        print("\n")
        k = ks.k_value(x1, x2)
        print(f"Test {i+1} : x1={x1}, x2={x2}, kernel calculé ={k}")
        assert ks.k_value(x1,x2,verbose=False) == ks.k_value(x2,x1,verbose=False), f"Kernel non symétrique"
        
    ks = KernelSpectrum(k=2)
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
    
    ks = KernelSpectrum(k=3)
    gram = ks.k_matrix(xs, xs)
    print(gram)
    
    ks = KernelSpectrum(k=4)
    gram = ks.k_matrix(xs, xs)
    print(gram)
    
    ks = KernelSpectrum(k=2)
    gram = ks.k_matrix(xs, xs)
    print(gram)
        
if __name__ == '__main__':
    test_kernel_spectrum()

#def test_kernel_spectrum():
#    
#    # tests
#    # tests = [
#    #     ['AGGCTTCGAC', 'CGGATGAGG', 1],   # common k_uplets : AGG (1,1), => k_value = 1
#    #     ['AGGCTTCGAC', 'CCGATGAGG', 2],   # common k_uplets : AGG (1,1), CGA (1,1 )=> k_value = 2
#    #     ['AAAAAAAAA', 'AAACGTGCAAA', 14]    # common k_uplets : AAA (7 in 1, 2 in 2) => k_value = 14
#    # ]
#    
#    # ks = KernelSpectrum(k=3)
#    
#    # for i,test in enumerate(tests):
#    #     x1 = test[0]
#    #     x2 = test[1]
#    #     target = test[2]
#    #     k_value = ks.k_value(x1,x2)
#    #     assert k_value == target, f'Revoir code KernelSpectrum : test {i}'
#        
#    # tests2 = [
#    #     ['AGGCTTCGAC', 'CGGATGAGG', 'AAAAAAAAA', 'AAACGTGCAAA']
#    # ]
#        
#    # for xs in tests2:
#    #     gram = ks.k_matrix(xs,xs)
#    #     print(gram)
#        
#    # print(f"Tests passés avec succès")
#    
#    # tests = [
#    #     ['AAG', 'TGC', 1],  # with m=1, k=2 : only match is 'TG'
#    #     ['AGGT', 'CGC', 4]  # with m=1, k=2 : matches are : (CG in x2 + AG in x1), (GC in x2 et GG in x1), (CG in x2 at GG in x1), (GC in x2 et GT in x1)
#    #     ]
#    
#    # ks = KernelMismatch(k=2)
#    
#    # for i,test in enumerate(tests):
#    #     x1 = test[0]
#    #     x2 = test[1]
#    #     target = test[2]
#    #     k_value = ks.k_value(x1,x2, mismatches=1)
#    #     assert k_value == target, f'Revoir code KernelSpectrum : test {i}'
#        
#    # print(f"Tests ok avec m=1")
#    
#    ks = KernelMismatch(k=3)
#    
#    tests = [
#        ['AGGCTTCGAC', 'CGGATGAGG', 1],   # common k_uplets : AGG (1,1), => k_value = 1
#        # ['AGGCTTCGAC', 'CCGATGAGG', 2],   # common k_uplets : AGG (1,1), CGA (1,1 )=> k_value = 2
#        # ['AAAAAAAAA', 'AAACGTGCAAA', 14]    # common k_uplets : AAA (7 in 1, 2 in 2) => k_value = 14
#    ]
#        
#    for i,test in enumerate(tests):
#        x1 = test[0]
#        x2 = test[1]
#        target = test[2]
#        k_value = ks.k_value(x1,x2, mismatches=1, verbose=True)
#        print(f"mismatches = 1, x1 = {x1}, x2 = {x2}, k_value = {k_value}")
#        k_value = ks.k_value(x1,x2, mismatches=2, verbose=True)
#        print(f"mismatches = 2, x1 = {x1}, x2 = {x2}, k_value = {k_value}")
#    
#    print(f"Gram matrix example")   
#    tests2 = [
#        ['AGGCTTCGAC', 'CGGATGAGG', 'AAAAAAAAA', 'AAACGTGCAAA']
#    ]
#        
#    for xs in tests2:
#        gram = ks.k_matrix(xs,xs)
#        print(gram)

