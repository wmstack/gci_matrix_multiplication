#display the data in matmul_jit.txt, matmul.txt, and matmul_numpy
from json import loads
from matplotlib import pyplot as plt
import matplotlib as mpl
import argparse

with open('matmul_jit.txt') as matmul_jit:
        matmul_jit_function = loads(matmul_jit.read())

with open('matmul.txt') as matmul:
    matmul_function = loads(matmul.read())

with open('matmul_numpy.txt') as matmul_numpy:
    matmul_numpy_function = loads(matmul_numpy.read())

def plot_all(numpy, jit, normal):

    xdata_m, ydata_m = zip(*matmul_function)
    xdata_mj, ydata_mj = zip(*matmul_jit_function)
    xdata_mn, ydata_mn = zip(*matmul_numpy_function)
    
    mpl.style.use('seaborn')
    if normal:
        plt.plot(xdata_m,ydata_m,'ro', ms=4, label='My matrix multiplication')
    if jit:
        plt.plot(xdata_mj,ydata_mj,'bo', ms=4, label='My matrix mul optimized with jit')
    if numpy:
        plt.plot(xdata_mn, ydata_mn,'yo',ms=4, label='Numpy matrix mul')

    plt.legend()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare times vs input for matrix multiplication')
    parser.add_argument('--jit','-j', action='store_true', help='Display jit')
    parser.add_argument('--numpy','-n', action='store_true', help='Display numpy')
    parser.add_argument('--normal','-c', action='store_true', help='Display my matrix multiplication')
    flags = parser.parse_args()

    plot_all(jit=flags.jit, numpy=flags.numpy, normal=flags.normal)