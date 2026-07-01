"""
Testing the factorization of the Hadamard transform
"""

import torch
from scipy.linalg import hadamard
from src.GB_factorization import *
from src.GB_operators import *
import numpy as np
import time


if __name__ == "__main__":
    n = 13
    N = 2 ** n
    matrix = torch.from_numpy(hadamard(N)).reshape(1,1,N,N)
    matrix = matrix.float()
    gb_params = [(2 ** i, 2, 2, 2 ** (n - i - 1), 1, 1) for i in range(n)]
    #orders = np.random.permutation(n - 1)
    orders = [i for i in range(n - 1)]
    print(orders)

    begin = time.time()
    factor_list = GBfactorize(matrix, gb_params, orders, False)
    factor_list = [f.factor for f in factor_list]
    end = time.time()
    print("Error: ", torch.norm(matrix - densification(factor_list, gb_params)))
    print("Running time: ", end - begin)

    begin = time.time()
    factor_list = GBfactorize(matrix, gb_params, orders, True)
    factor_list = [f.factor for f in factor_list]
    end = time.time()
    print(torch.norm(matrix - densification(factor_list, gb_params)))
    print("Running time: ", end - begin)