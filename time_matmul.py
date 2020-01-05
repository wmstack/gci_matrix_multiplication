from matmul_custom import matmul_jit, matmul_no_jit

#for generating the arrays
import numpy  as np

#for timing
from functools import partial
from timeit import Timer

#for writing json
from json import dumps

#argparse
import argparse

def main(use_jit, use_numpy, maxsize):

    if use_jit:
        matmul = matmul_jit
    elif use_numpy:
        matmul = np.matmul
    else:
        matmul = matmul_no_jit

    times = []
    try:
        for n in range(10, maxsize+1, 10):
            #get random matrix
            mat = np.random.random([n,n])
            
            #apply the matrix twice to matmul
            pmatmul = partial(matmul, mat, mat)
            
            #time the function
            time = Timer(pmatmul).timeit(number=1)
            times.append([n, time])

            print('done with {}'.format(n))
    except KeyboardInterrupt:
        pass
    finally:
        #dump the data to custom_matmul.txt
        with open('matmul'+'_jit'*use_jit + '_numpy'*use_numpy+'.txt','w') as file:
            file.write(dumps(times))

def parse_args():
    
    parser = argparse.ArgumentParser(description='Timing matrix multiplication functions. Output saved to matmul.txt, matmul_jit.txt, matmul_numpy.txt')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--jit','-j', action='store_true', help='Time the numba jit optimized version of my matmul')
    group.add_argument('--numpy','-n', action='store_true', help='Time the numpy version of matmul')
    
    parser.add_argument('max', nargs='?',type=int,default=500, help='loop from 10 with a step of 10 till this number, default is 500')

    return parser.parse_args()

if __name__ == "__main__":
    parse = parse_args()
    
    main(parse.jit, parse.numpy, parse.max)
